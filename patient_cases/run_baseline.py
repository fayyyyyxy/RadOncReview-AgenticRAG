"""Run development inputs through the existing RAG; dry-run by default."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import io
import json
import re
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from load_scenarios import DEFAULT_WORKBOOK, VIGNETTE, QUESTION, read_records, validate_records
from config import LLMConfig
from core.case_labels import SETTING_GROUPS, CaseLabels

# Model for every engine LLM call: the web app's, so a run measures what the app runs.
# ⚠️ Read from config.py rather than written out again; two copies drifted apart before (Llama 3 vs 3.1)
WRITER_MODEL = LLMConfig.model_id


def make_query(record):
    return f"Patient vignette:\n{record['vignette']}\n\nClinical question:\n{record['clinical_question']}"


def load_inputs(path, workbook):
    """Check every input against the current workbook, not only its case ID."""
    records = read_records(workbook)
    validate_records(records)
    eligible = {
        r["Case ID"]: r for r in records
        if r["Split"] == "Development" and r["Worked-Example Eligible"] == "Yes"
    }
    inputs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen = set()
    for item in inputs:
        if not isinstance(item, dict) or set(item) != {"case_id", "vignette", "clinical_question"}:
            raise ValueError("Baseline inputs must contain only case_id, vignette, clinical_question.")
        case_id = item["case_id"]
        if not isinstance(case_id, str) or case_id in seen or case_id not in eligible:
            raise ValueError("Duplicate or non-development Case ID in baseline inputs.")
        seen.add(case_id)
        source = eligible[case_id]
        if item["vignette"] != source[VIGNETTE] or item["clinical_question"] != source[QUESTION]:
            raise ValueError(f"Input text does not match the current workbook for {case_id}.")
    if not inputs:
        raise ValueError("Baseline input file is empty.")
    return inputs


def labels_for(case_ids, workbook):
    """Workbook labels standing in for a physician's drop-down choices; they never enter the query text."""
    records = {r["Case ID"]: r for r in read_records(workbook)}
    return {
        case_id: CaseLabels(
            disease_site=records[case_id]["Disease Site"],
            setting_group=SETTING_GROUPS[records[case_id]["Setting"]],
            prior_rt=records[case_id]["Prior RT"],
            encounter_type=records[case_id]["Encounter Type"],
        )
        for case_id in case_ids
    }


# The engine's trace line giving the size and reasoning of the first-round plan (written in core/agent.py)
PLAN_LINE = re.compile(r"\] PLAN: Created (\d+) sub-tasks: (.*)", re.S)


def saved_plans(run_name, selected):
    """Each selected case's first-round plan from a saved run: its analysis and the sub-tasks searched first."""
    from core.models import QueryAnalysis, QueryIntent, RetrievalStrategy, SubTask, TaskPlan
    path = ROOT / "outputs" / f"{run_name}_answers" / "results.jsonl"
    if not path.exists():
        raise ValueError(f"No saved run {path.parent.name} to replay plans from.")
    records = {r["case_id"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    plans = {}
    for item in selected:
        record = records.get(item["case_id"], {})
        response = record.get("response") or {}
        saved = response.get("task_plan")
        if not saved:
            raise ValueError(f"{path.parent.name} has no plan for {item['case_id']}.")
        if record["query"] != make_query(item):
            raise ValueError(f"{path.parent.name} asked about different input text for {item['case_id']}.")
        # ⚠️ Saved sub-tasks hold the first round and any re-plan together. Only the first round is replayed:
        # a re-plan answers that run's own draft, so it is made again for the new draft
        found = [m for line in response["reasoning_trace"] for m in [PLAN_LINE.search(line)] if m]
        if len(found) != 1:
            raise ValueError(f"Cannot tell the first-round plan of {item['case_id']} in {path.parent.name}.")
        size, reasoning = int(found[0].group(1)), found[0].group(2)
        plans[item["case_id"]] = TaskPlan(
            # ⚠️ Not the saved reasoning: a re-plan overwrites it (35 of 40 cases in gpt56terra-run3), and the
            # re-plan prompt shows it, so a replay would hand the new re-plan the old run's second thoughts
            plan_id=saved["plan_id"], original_query=saved["original_query"], reasoning=reasoning,
            analysis=QueryAnalysis(**{**saved["analysis"], "intent": QueryIntent(saved["analysis"]["intent"])}),
            # New task objects, so nothing the saved run retrieved is carried over
            sub_tasks=[SubTask(task_id=t["task_id"], description=t["description"], query=t["query"],
                               retrieval_strategy=RetrievalStrategy(t["retrieval_strategy"]), top_k=t["top_k"])
                       for t in saved["sub_tasks"][:size]],
        )
    return plans


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported output type: {type(value).__name__}")


def run_case(engine, item, max_length, labels=None, case_context=None, plan=None):
    """Save blocked/error cases explicitly; never truncate clinical input."""
    query = make_query(item)
    labels = labels or CaseLabels()
    result = {"case_id": item["case_id"], "query": query, "case_labels": vars(labels)}
    if len(query) > max_length:
        return {**result, "status": "skipped_input_too_long", "characters": len(query), "limit": max_length}
    started = time.monotonic()
    log = io.StringIO()
    try:
        engine.reset_conversation()
        with redirect_stdout(log), redirect_stderr(log):
            extra = {key: value for key, value in (("case_context", case_context), ("plan", plan)) if value is not None}
            response = engine.query(query, labels, **extra)
        retrieval = response.metadata.get("retrieval", {})
        status = "response_generated"
        if not retrieval:
            status = "blocked_or_no_retrieval_metadata"
        elif retrieval.get("tasks_completed", 0) < retrieval.get("tasks_total", 0):
            status = "retrieval_incomplete"
        elif not retrieval.get("total_chunks", 0):
            status = "no_evidence_retrieved"
        elif not response.citations:
            status = "response_without_citations"
        result.update(status=status, response=asdict(response))
    except Exception as exc:
        result.update(status="error", error_type=type(exc).__name__, error=str(exc))
    result.update(elapsed_seconds=round(time.monotonic() - started, 3), engine_log=log.getvalue())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/processed/baseline_inputs.jsonl")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--model", default=WRITER_MODEL, help="Bedrock model ID for every engine LLM call.")
    parser.add_argument("--execute", action="store_true", help="Make real AWS retrieval and paid model calls.")
    parser.add_argument("--judge", action="store_true", help="Enable additional paid LLM judge calls.")
    parser.add_argument("--name", help="Readable run name such as site-filter; answers go to outputs/<name>_answers.")
    parser.add_argument("--site-filter", action="store_true", help="Limit evidence to each case's disease site.")
    parser.add_argument("--no-trial-papers", action="store_true", help="Search review notes only, not the trial-paper Knowledge Base.")
    parser.add_argument("--no-citation-check", action="store_true", help="Skip the citation check, as runs made before it did.")
    parser.add_argument("--institutional-examples", action="store_true", help="Include up to two institutional examples in synthesis.")
    parser.add_argument("--past-case-index", type=Path, help="Existing local past-case index folder (does not build it).")
    parser.add_argument("--replay-plans", metavar="NAME",
                        help="Search with the first-round plans saved in outputs/NAME_answers instead of planning anew, "
                             "so runs compared case by case retrieve the same first-round passages.")
    parser.add_argument("--repair-batch", action="store_true",
                        help="Turn on the whole repair batch: no re-plan, the critic accepts a named gap, and the "
                             "citation check only flags (docs/rag_improvement_plan_v2.md, Part 1).")
    args = parser.parse_args(argv)
    try:
        for flag, value in (("--name", args.name), ("--replay-plans", args.replay_plans)):
            if value is not None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", value):
                raise ValueError(f"{flag} may contain only letters, digits, '-' and '_'.")
        # Checked before any paid call, so a clashing name never costs a run
        named = ROOT / "outputs" / f"{args.name}_answers" if args.name else None
        if named is not None and named.exists():
            raise ValueError(f"{named.name} already exists; choose another --name.")
        inputs = load_inputs(args.inputs, args.workbook)
        if not 1 <= args.limit <= len(inputs):
            raise ValueError(f"--limit must be between 1 and {len(inputs)}.")
        selected = inputs[:args.limit]
        labels = labels_for([r["case_id"] for r in selected], args.workbook)
        # Read in dry-run mode too, so a missing or mismatched plan shows before a paid run
        plans = saved_plans(args.replay_plans, selected) if args.replay_plans else {}
        # No engine import, model download or AWS calls in dry-run mode.
        if not args.execute:
            print(json.dumps({
                "mode": "dry_run", "aws_calls": 0, "model": args.model, "site_filter": args.site_filter,
                "trial_papers": not args.no_trial_papers, "citation_check": not args.no_citation_check,
                "institutional_examples": args.institutional_examples, "replayed_plans": args.replay_plans,
                "repair_batch": args.repair_batch,
                "example_index_validation": "not_performed_in_dry_run" if args.institutional_examples else "not_requested",
                "output_folder": named.name if named else "baseline_<timestamp>_<id>",
                "validated_development_inputs": len(inputs),
                "selected": [{"case_id": r["case_id"], "query_characters": len(make_query(r))} for r in selected],
                "note": "No attending plans passed; no answers generated. Use --execute for a real run.",
            }, indent=2))
            return 0

        from config import Config, validate_config
        from core.agent import AgenticRAGEngine
        validate_config()
        config = Config()
        config.llm.temperature = 0.0
        # ⚠️ One shared client serves the analyzer, planner, writer and self-critique, so all of them switch
        config.llm.model_id = args.model
        config.features.site_filter = args.site_filter
        config.features.citation_check = not args.no_citation_check
        config.features.institutional_examples = args.institutional_examples
        if args.repair_batch:
            # 💡 One flag for the batch, so a run never carries half of it
            config.features.replan = False
            config.features.critic_named_gap = True
            config.features.citation_flag_only = True
        if args.past_case_index is not None:
            config.features.past_case_index_path = str(args.past_case_index.resolve())
        case_contexts = {}
        if args.institutional_examples:
            from core.institutional_examples import CaseContext
            workbook_records = read_records(args.workbook)
            records = {r["Case ID"]: r for r in workbook_records}
            export_path = ROOT / "data" / "processed" / "development_examples.jsonl"
            config.features.past_case_source_sha256 = hashlib.sha256(export_path.read_bytes()).hexdigest()
            case_contexts = {item["case_id"]: CaseContext(
                vignette=item["vignette"], clinical_question=item["clinical_question"],
                case_id=item["case_id"], patient_id=records[item["case_id"]]["Patient ID"],
            ) for item in selected}
        # The planner still makes a trial-paper task per topic; with the Knowledge Base off it returns nothing
        config.aws.knowledge_base_enabled = not args.no_trial_papers
        # Log all settings, including the opt-in judge difference from the web app.
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
        # ⚠️ Tracked files only: untracked notes and drawings outside the code marked every run dirty
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                 capture_output=True, text=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        output = named or ROOT / "outputs" / ("baseline_" + run_id)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": run_id, "mode": "evidence_with_institutional_examples" if args.institutional_examples else "existing_evidence_only_rag",
            "git_commit": revision.stdout.strip(), "working_tree_dirty": bool(changes.stdout.strip()),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
            "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
            "case_ids": [r["case_id"] for r in selected], "judge_enabled": args.judge,
            "patient_example_retrieval": args.institutional_examples, "repair_batch": args.repair_batch,
            "config": asdict(config),
            "replayed_plans": {
                "run": args.replay_plans,
                "results_sha256": hashlib.sha256(
                    (ROOT / "outputs" / f"{args.replay_plans}_answers" / "results.jsonl").read_bytes()).hexdigest(),
            } if args.replay_plans else None,
            "evidence_snapshot": "Existing live AWS resources; not frozen or clinically validated",
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        statuses = []
        try:
            # Store startup warnings, including any reranker model download failure.
            with (output / "startup.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                engine = AgenticRAGEngine(config=config, enable_judge=args.judge)
            if args.institutional_examples:
                manifest["institutional_example_index"] = engine.example_provider.provenance()
                (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            for item in selected:
                result = run_case(engine, item, config.guardrail.max_input_length, labels[item["case_id"]],
                                  case_contexts.get(item["case_id"]), plans.get(item["case_id"]))
                with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, default=json_value) + "\n")
                statuses.append(result["status"])
                # ⚠️ A model that rejects temperature drops it on its first call, so the manifest is corrected to
                # what was sent (null) instead of keeping the configured 0.0
                sent = engine.llm.sampling_sent
                if any(manifest["config"]["llm"][key] != value for key, value in sent.items()):
                    manifest["config"]["llm"].update(sent)
                    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                print(f"{item['case_id']}: {result['status']}", flush=True)
        except Exception as exc:
            (output / "startup_error.json").write_text(json.dumps({"type": type(exc).__name__, "error": str(exc)}), encoding="utf-8")
            raise
        print(f"Local results: {output}")
        return 0 if all(s == "response_generated" for s in statuses) else 1
    except Exception as exc:
        print(f"Baseline runner failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
