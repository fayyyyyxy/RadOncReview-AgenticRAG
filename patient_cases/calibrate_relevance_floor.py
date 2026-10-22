"""Set the passage relevance floor on development cases outside the evaluation set.

Each sampled case is analysed, planned and retrieved exactly as the engine does with the site filter on, but no
answer is written. The cross-encoder then scores every candidate passage against its own case and against
passages retrieved for cases of other disease sites. The floor is a high percentile of those mismatched scores:
a passage that scores no better than evidence for an unrelated case is not evidence.

Prints counts and percentiles only; per-case scores stay in the ignored outputs folder.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import redirect_stdout
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from run_baseline import DEFAULT_WORKBOOK, WRITER_MODEL, labels_for, make_query

DEVELOPMENT = ROOT / "data" / "processed" / "development_examples.jsonl"
EVALUATION = ROOT / "data" / "processed" / "evaluation_set.json"
OUTPUT = ROOT / "outputs" / "relevance_floor"
# These sites search every review-note category, so their passages may belong to any site
BROAD_SITES = {"Palliative", "Benign & HO Prophylaxis"}


def sample_cases(per_site, seed):
    """Development cases outside the evaluation set, up to per_site from each disease site."""
    evaluation = {c["case_id"] for c in json.loads(EVALUATION.read_text())["cases"]}
    cases = [json.loads(line) for line in DEVELOPMENT.read_text(encoding="utf-8").splitlines() if line.strip()]
    tuning = sorted((c for c in cases if c["case_id"] not in evaluation), key=lambda c: c["case_id"])
    labels = labels_for([c["case_id"] for c in tuning], DEFAULT_WORKBOOK)
    by_site = defaultdict(list)
    for case in tuning:
        by_site[labels[case["case_id"]].disease_site].append(case)
    rng = random.Random(seed)
    chosen = [c for site in sorted(by_site) for c in rng.sample(by_site[site], min(per_site, len(by_site[site])))]
    return chosen, labels


def rerank_calls(engine, query, labels):
    """Every rerank the engine makes for one query: (rerank query, candidate passages, top_k).

    Mirrors steps 1-4 of AgenticRAGEngine.query; keep the two in step.
    """
    from core.models import QueryIntent
    from core.site_routing import scope_for

    check = engine.guardrails.check_input(query)
    if not check.passed:
        return []
    query = check.filtered_text or query
    analysis = engine.query_analyzer.analyze(query)
    if analysis.is_complex or analysis.confidence < 0.7:
        analysis = engine.query_analyzer.analyze_with_llm(query)
    plan = engine.task_planner.plan(analysis)
    scope = scope_for(labels.disease_site)
    for task in plan.sub_tasks:
        engine.retriever.execute_task(task, scope)

    calls, original = [], engine.retriever.rerank
    # 💡 Wrapping rerank records exactly what the cross-encoder would see, including the per-topic query
    def record(rerank_query, chunks, top_k=None):
        calls.append({"query": rerank_query, "passages": [(c.text, c.source_type) for c in chunks],
                      "top_k": top_k or engine.config.retriever.rerank_top_k})
        return original(rerank_query, chunks, top_k)
    engine.retriever.rerank = record
    try:
        if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
            engine._collect_multi_topic_chunks(plan)
        else:
            engine._collect_all_chunks(plan)
    finally:
        engine.retriever.rerank = original
    return calls


def score(model, query, texts):
    return model.predict([[query, t] for t in texts], show_progress_bar=False).tolist() if texts else []


def kept(scores, top_k, floor):
    """Passages the engine would pass on: the top_k by score, minus any below the floor."""
    return sum(s >= floor for s in sorted(scores, reverse=True)[:top_k])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-site", type=int, default=5, help="Development cases per disease site.")
    parser.add_argument("--others", type=int, default=3, help="Other-site cases whose passages each case is scored against.")
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Make the analyser, planner and retrieval calls.")
    args = parser.parse_args(argv)
    try:
        cases, labels = sample_cases(args.per_site, args.seed)
        sites = {c["case_id"]: labels[c["case_id"]].disease_site for c in cases}
        if not args.execute:
            print(json.dumps({"mode": "dry_run", "cases": len(cases), "by_site": dict(Counter(sites.values()))}, indent=2))
            return 0

        from config import Config, validate_config
        from core.agent import AgenticRAGEngine
        validate_config()
        config = Config()
        config.llm.temperature, config.llm.model_id = 0.0, WRITER_MODEL
        config.features.site_filter = True
        OUTPUT.mkdir(parents=True, exist_ok=True)
        # The engine logs task descriptions drawn from the case text, so its output goes to the ignored folder
        with (OUTPUT / "engine.log").open("w", encoding="utf-8") as log, redirect_stdout(log):
            engine = AgenticRAGEngine(config=config, enable_judge=False)
            calls = {}
            for case in cases:
                calls[case["case_id"]] = rerank_calls(engine, make_query(case), labels[case["case_id"]])
                print(f"{case['case_id']}: {len(calls[case['case_id']])} rerank calls", file=sys.stderr, flush=True)
        model = engine.retriever._cross_encoder._model

        rng = random.Random(args.seed)
        own, mismatched, rows = [], [], []
        for case_id, case_calls in calls.items():
            # ⚠️ Broad sites draw notes from every site, so neither side of a mismatched pair may be one
            others = [c for c in calls if sites[c] != sites[case_id] and sites[c] not in BROAD_SITES]
            foreign = sorted({text for other in rng.sample(others, min(args.others, len(others)))
                              for call in calls[other] for text, _ in call["passages"]})
            case_scores = []
            for call in case_calls:
                scores = score(model, call["query"], [text for text, _ in call["passages"]])
                own += scores
                case_scores.append({"top_k": call["top_k"], "scores": scores,
                                    "sources": [source for _, source in call["passages"]]})
                if sites[case_id] not in BROAD_SITES:
                    mismatched += score(model, call["query"], foreign)
            rows.append({"case_id": case_id, "disease_site": sites[case_id], "calls": case_scores})

        floor = float(np.percentile(mismatched, args.percentile))
        selected = lambda f: [sum(kept(c["scores"], c["top_k"], f) for c in r["calls"]) for r in rows]
        today = selected(float("-inf"))
        with_floor = selected(floor)
        top_scores = [s for r in rows for c in r["calls"] for s in sorted(c["scores"], reverse=True)[:c["top_k"]]]
        summary = {
            "cases": len(rows), "by_site": dict(Counter(sites.values())), "rerank_calls": sum(len(r["calls"]) for r in rows),
            "mismatched_pairs": len(mismatched), "own_pairs": len(own),
            "mismatched_percentiles": {p: round(float(np.percentile(mismatched, p)), 4) for p in (50, 90, 95, 99)},
            "own_percentiles": {p: round(float(np.percentile(own, p)), 4) for p in (50, 90, 95, 99)},
            "floor": round(floor, 4), "percentile": args.percentile,
            "selected_passages_below_floor": round(sum(s < floor for s in top_scores) / len(top_scores), 3),
            "passages_per_case_today": dict(sorted(Counter(today).items())),
            "passages_per_case_with_floor": dict(sorted(Counter(with_floor).items())),
            "cases_left_without_passages": sorted(r["case_id"] for r, n in zip(rows, with_floor) if n == 0),
            "median_passages_with_floor_by_site": {
                site: float(np.median([n for r, n in zip(rows, with_floor) if r["disease_site"] == site]))
                for site in sorted(set(sites.values()))},
        }
        (OUTPUT / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (OUTPUT / "mismatched_scores.json").write_text(json.dumps(mismatched), encoding="utf-8")
        (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"Relevance floor calibration failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
