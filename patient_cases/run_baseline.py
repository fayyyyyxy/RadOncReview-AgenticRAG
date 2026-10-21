"""Run development inputs through the existing RAG; dry-run by default."""

import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from load_scenarios import DEFAULT_WORKBOOK, VIGNETTE, QUESTION, read_records, validate_records

# Model for every engine LLM call in evaluation runs. Llama 3 70B's 8k context overflowed on
# long evidence; the web app keeps config.py's default until a site is declared GO.
WRITER_MODEL = "meta.llama3-1-70b-instruct-v1:0"


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


def json_value(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported output type: {type(value).__name__}")


def run_case(engine, item, max_length):
    """Save blocked/error cases explicitly; never truncate clinical input."""
    query = make_query(item)
    result = {"case_id": item["case_id"], "query": query}
    if len(query) > max_length:
        return {**result, "status": "skipped_input_too_long", "characters": len(query), "limit": max_length}
    started = time.monotonic()
    log = io.StringIO()
    try:
        engine.reset_conversation()
        with redirect_stdout(log), redirect_stderr(log):
            response = engine.query(query)
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
    args = parser.parse_args(argv)
    try:
        inputs = load_inputs(args.inputs, args.workbook)
        if not 1 <= args.limit <= len(inputs):
            raise ValueError(f"--limit must be between 1 and {len(inputs)}.")
        selected = inputs[:args.limit]
        # No engine import, model download or AWS calls in dry-run mode.
        if not args.execute:
            print(json.dumps({
                "mode": "dry_run", "aws_calls": 0, "model": args.model,
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
        # Log all settings, including the opt-in judge difference from the web app.
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True)
        changes = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
        output = ROOT / "outputs" / ("baseline_" + run_id)
        output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "run_id": run_id, "mode": "existing_evidence_only_rag",
            "git_commit": revision.stdout.strip(), "working_tree_dirty": bool(changes.stdout.strip()),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
            "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
            "case_ids": [r["case_id"] for r in selected], "judge_enabled": args.judge,
            "patient_example_retrieval": False, "config": asdict(config),
            "evidence_snapshot": "Existing live AWS resources; not frozen or clinically validated",
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        statuses = []
        try:
            # Store startup warnings, including any reranker model download failure.
            with (output / "startup.log").open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
                engine = AgenticRAGEngine(config=config, enable_judge=args.judge)
            for item in selected:
                result = run_case(engine, item, config.guardrail.max_input_length)
                with (output / "results.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False, default=json_value) + "\n")
                statuses.append(result["status"])
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
