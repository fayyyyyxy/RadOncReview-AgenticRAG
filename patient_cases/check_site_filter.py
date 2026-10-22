"""Check that a --site-filter run kept all evidence inside each case's disease site; no model calls.

Reads results.jsonl from an answers folder. Prints case IDs and counts only.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from core.case_labels import CaseLabels
from core.site_routing import scope_for


def check_case(result):
    """Evidence counts for one answered case, and how much of it falls outside the case's site."""
    labels = CaseLabels(**result["case_labels"])
    scope = scope_for(labels.disease_site)
    response = result["response"]
    passages = response.get("context_passages") or []
    # Everything the searches returned, not only what reached the model
    retrieved = [chunk for task in (response.get("task_plan") or {}).get("sub_tasks", [])
                 for chunk in task.get("retrieved_chunks", [])]
    return {
        "case_id": result["case_id"], "disease_site": labels.disease_site,
        "site_filter_on": bool((response.get("routing") or {}).get("site_filter")),
        "passages": len(passages),
        "review_notes": sum(p.get("source_type") != "clinical_trial" for p in passages),
        "trial_papers": sum(p.get("source_type") == "clinical_trial" for p in passages),
        "outside_scope": sum(not scope.allows(c) for c in passages + retrieved),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Answers folder, e.g. outputs/site-filter_answers")
    args = parser.parse_args(argv)
    try:
        lines = (args.run / "results.jsonl").read_text(encoding="utf-8").splitlines()
        results = [json.loads(line) for line in lines if line.strip()]
        rows = [check_case(r) for r in results if r["status"] == "response_generated"]
        if not rows:
            raise ValueError("No answered cases to check.")
        summary = {
            "answered": len(rows), "not_answered": len(results) - len(rows),
            "site_filter_off": [r["case_id"] for r in rows if not r["site_filter_on"]],
            "evidence_outside_site": {r["case_id"]: r["outside_scope"] for r in rows if r["outside_scope"]},
            "passages_median": statistics.median(r["passages"] for r in rows),
            "no_passages": [r["case_id"] for r in rows if not r["passages"]],
            "no_review_notes_by_site": dict(Counter(r["disease_site"] for r in rows if not r["review_notes"])),
            "passages_by_source": {"review_notes": sum(r["review_notes"] for r in rows),
                                   "trial_papers": sum(r["trial_papers"] for r in rows)},
        }
        print(json.dumps(summary, indent=2))
        passed = not summary["site_filter_off"] and not summary["evidence_outside_site"]
        print("Site filter check:", "passed" if passed else "FAILED")
        return 0 if passed else 1
    except Exception as exc:
        print(f"Site filter check failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
