"""Repeat fresh clinical retrieval on Development inputs; never writes an answer."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))
from load_scenarios import DEFAULT_WORKBOOK
from run_baseline import load_inputs, labels_for, make_query, json_value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=json_value).encode()).hexdigest()


def summarize(runs):
    """Compare stages separately; an identical empty/failed retrieval cannot pass."""
    baseline = set(tuple(p) for p in runs[0]["passages"])
    overlaps = []
    for run in runs[1:]:
        current = set(tuple(p) for p in run["passages"])
        overlaps.append(len(current & baseline) / len(current | baseline) if current | baseline else None)
    result = {"case_id": runs[0]["case_id"], "repeats": len(runs),
              "healthy": all(r["healthy"] for r in runs),
              **{f"same_{key}": len({r[key] for r in runs}) == 1 for key in
                 ("query_plan_hash", "candidate_pool_hash", "ordered_passage_hash")},
              "passage_counts": [len(r["passages"]) for r in runs], "jaccard_vs_first": overlaps}
    result["passed"] = result["healthy"] and all(result[f"same_{key}"] for key in
                                               ("query_plan_hash", "candidate_pool_hash", "ordered_passage_hash"))
    return result


def retrieve_once(engine, config, item, labels, repeat, mode):
    from core.models import QueryIntent, TaskStatus
    from core.retrieval_identity import passage_identity
    from core.site_routing import EvidenceScope, scope_for

    query = make_query(item)
    scope = scope_for(labels.disease_site) if config.features.site_filter else EvidenceScope()
    started, log = time.monotonic(), io.StringIO()
    with redirect_stdout(log), redirect_stderr(log):
        analysis = engine.query_analyzer.analyze(query)
        if mode == "llm" and (analysis.is_complex or analysis.confidence < 0.7):
            analysis = engine.query_analyzer.analyze_with_llm(query)
        plan = engine.task_planner.plan(analysis)
        analysis = plan.analysis
        for task in plan.sub_tasks:
            engine.retriever.execute_task(task, scope)
        if analysis.is_complex or analysis.intent == QueryIntent.COMPARISON:
            topics = engine._collect_multi_topic_chunks(plan)
            final = [chunk for chunks in topics.values() for chunk in chunks]
        else:
            topics = {query: engine._collect_all_chunks(plan)}
            final = topics[query]
    semantic_plan = [{"query": t.query, "description": t.description,
                      "strategy": t.retrieval_strategy.value, "top_k": t.top_k} for t in plan.sub_tasks]
    raw_pools = [sorted(passage_identity(c) for c in t.retrieved_chunks) for t in plan.sub_tasks]
    ordered = [passage_identity(c) for c in final]
    diagnostics = [line for line in log.getvalue().splitlines() if any(marker in line.lower() for marker in
                   ("failed", "error fetching", "warning:"))]
    # Intentionally disabled sources are excluded; empty active sources are not a healthy check.
    empty_tasks = [t.description for t in plan.sub_tasks if not t.retrieved_chunks and not (
        (t.retrieval_strategy.value == "dense" and scope.review_note_categories == frozenset()) or
        (t.retrieval_strategy.value == "knowledge_base" and scope.trial_paper_folders == frozenset()))]
    reranker_loaded = engine.retriever._cross_encoder._model is not None
    return {"case_id": item["case_id"], "input": item, "labels": asdict(labels), "repeat": repeat,
            "analysis": asdict(analysis), "plan": asdict(plan),
            "query_plan_hash": digest(semantic_plan), "candidate_pool_hash": digest(raw_pools),
            "ordered_passage_hash": digest(ordered), "passages": ordered,
            "selected_chunks": final, "selected_by_topic": topics,
            "healthy": bool(final) and bool(plan.sub_tasks) and reranker_loaded and not diagnostics
                       and plan.planning.get("cache_status") != "fallback_not_cached"
                       and not empty_tasks and all(t.status == TaskStatus.COMPLETED for t in plan.sub_tasks),
            "diagnostics": diagnostics, "empty_tasks": empty_tasks, "reranker_loaded": reranker_loaded,
            "seconds": round(time.monotonic() - started, 3), "log": log.getvalue()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=ROOT / "data/processed/baseline_inputs.jsonl")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case-id")
    selection.add_argument("--all-inputs", action="store_true", help="Interleave repeats over every supplied input.")
    parser.add_argument("--mode", choices=("llm", "versioned"), default="versioned")
    parser.add_argument("--plan-cache", type=Path, help="Local cache of versioned plans, never passages.")
    parser.add_argument("--site-filter", action="store_true", help="Apply the existing disease-site routing table.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    inputs = load_inputs(args.inputs, args.workbook)
    items = inputs if args.all_inputs else [next((r for r in inputs if r["case_id"] == args.case_id), None)
                                          if args.case_id else inputs[0]]
    if any(r is None for r in items) or not 2 <= args.repeats <= 10:
        raise ValueError("Choose eligible inputs and 2–10 repetitions.")
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "case_ids": [r["case_id"] for r in items], "planning": args.mode,
                          "repeats": args.repeats, "site_filter": args.site_filter, "aws_calls": 0,
                          "note": "Execute repeats live search; no saved passages or answers are replayed."}, indent=2))
        return 0

    from config import Config, validate_config
    from core.agent import AgenticRAGEngine
    from core.site_routing import table_fingerprint
    validate_config()
    config = Config()
    config.llm.temperature = 0.0
    config.features.versioned_planning = args.mode == "versioned"
    if args.plan_cache is not None:
        config.features.plan_cache_path = str(args.plan_cache.resolve())
    config.features.institutional_examples = False
    config.features.site_filter = args.site_filter
    for item in items:
        if len(make_query(item)) > config.guardrail.max_input_length:
            raise ValueError("Input exceeds the engine's input limit; no truncation is allowed.")
    output = ROOT / "outputs" / ("retrieval-consistency-" + args.mode + "-" +
                                 datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"planning_mode": args.mode, "requested_repeats": args.repeats,
                "case_ids": [r["case_id"] for r in items], "config": asdict(config),
                "input_sha256": digest(items), "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
                "site_routing_version": table_fingerprint(),
                "code_sha256": {str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted([ROOT.parent / "config.py", Path(__file__),
                                                 *list((ROOT.parent / "core").glob("*.py"))])},
                "no_passage_replay": True, "writer_calls": 0,
                "note": "FAISS loads once as in the app; DynamoDB/KB searched anew; live sources not frozen."}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Local report: {output}", flush=True)
    with (output / "startup.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
        engine = AgenticRAGEngine(config, enable_judge=False)
    labels = labels_for([r["case_id"] for r in items], args.workbook)
    for item in items:
        if not engine.guardrails.check_input(make_query(item)).passed:
            raise ValueError(f"Input blocked by the engine's guardrail: {item['case_id']}")
    runs = []
    for repeat in range(1, args.repeats + 1):
        for item in items:
            record = retrieve_once(engine, config, item, labels[item["case_id"]], repeat, args.mode)
            with (output / "runs.jsonl").open("a") as handle:
                handle.write(json.dumps(record, default=json_value) + "\n")
            runs.append(record)
            print(f"{item['case_id']} repeat {repeat}: {len(record['passages'])} passages; "
                  f"healthy={record['healthy']}; {record['seconds']}s", flush=True)
        cases = [summarize([r for r in runs if r["case_id"] == item["case_id"]]) for item in items]
        summary = {"planning_mode": args.mode, "repeats_completed": repeat, "case_count": len(items),
                   "healthy_cases": sum(c["healthy"] for c in cases),
                   "passed_cases": sum(c["passed"] for c in cases) if repeat >= 2 else None, "cases": cases}
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if all(c["passed"] for c in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
