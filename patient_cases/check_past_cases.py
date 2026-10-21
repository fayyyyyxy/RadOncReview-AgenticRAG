"""Check past-case selection on every indexed case and write a local review page; no model calls.

Each case queries the index with its own workbook labels, as if chosen from the drop-down lists, and its stored
vector. The terminal shows counts only; the review page holds clinical text and stays in the ignored outputs folder.
"""

import argparse
from collections import Counter
from html import escape
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from core.case_labels import CaseLabels
from core.past_cases import NumpyPastCaseIndex, select_examples
from build_past_case_index import INDEX_FOLDER, PROCESSED

OUTPUT = ROOT / "outputs" / "past_cases_check"


def labels_of(record):
    return CaseLabels(record["disease_site"], record["setting_group"], record["prior_rt"], record["encounter_type"])


def check(index, min_similarity=None):
    """Selection for every case, with any rule it breaks."""
    rows = []
    for i, query in enumerate(index.records):
        hits = select_examples(index, index.vectors[i], labels_of(query), query["case_id"], query["patient_id"],
                               min_similarity=min_similarity)
        chosen = [h.record for h in hits]
        # What each tier could offer: same site, other patients, then narrowed by setting and prior RT
        same_site = [r for r in index.records
                     if r["disease_site"] == query["disease_site"] and r["patient_id"] != query["patient_id"]]
        same_setting = [r for r in same_site if r["setting_group"] == query["setting_group"]]
        same_both = [r for r in same_setting if r["prior_rt_group"] == query["prior_rt_group"]]
        tier = lambda r: ("setting + prior RT" if r in same_both else "setting" if r in same_setting else "site only")
        # With a floor, a tier can only be expected to offer the cases above it
        if min_similarity is not None:
            above = {h.record["case_id"] for h in index.search(index.vectors[i], {"disease_site": query["disease_site"]},
                                                                {"patient_id": {query["patient_id"]}}, len(index.records))
                     if h.similarity >= min_similarity}
            same_both = [r for r in same_both if r["case_id"] in above]
            same_setting = [r for r in same_setting if r["case_id"] in above]
        problems = [name for name, broken in (
            ("chose itself", any(r["case_id"] == query["case_id"] for r in chosen)),
            ("chose the same patient", any(r["patient_id"] == query["patient_id"] for r in chosen)),
            ("chose another site", any(r["disease_site"] != query["disease_site"] for r in chosen)),
            ("more than two", len(chosen) > 2),
            ("chose a case twice", len({r["case_id"] for r in chosen}) != len(chosen)),
            ("chose a case below the similarity floor",
             min_similarity is not None and any(h.similarity < min_similarity for h in hits)),
            ("skipped a same-setting and prior-RT case", sum(r in same_both for r in chosen) < min(2, len(same_both))),
            ("skipped a same-setting case", sum(r in same_setting for r in chosen) < min(2, len(same_setting))),
        ) if broken]
        rows.append({
            "case_id": query["case_id"], "disease_site": query["disease_site"], "problems": problems,
            "examples": [{"case_id": h.record["case_id"], "similarity": round(h.similarity, 3), "tier": tier(h.record),
                          "same_setting": h.record["setting_group"] == query["setting_group"],
                          "same_encounter": h.record["encounter_type"] == query["encounter_type"]} for h in hits],
        })
    return rows


def summarise(rows, index, min_similarity=None):
    firsts = sorted(r["examples"][0]["similarity"] for r in rows if r["examples"])
    return {
        "index_version": index.version(), "min_similarity": min_similarity, "queries": len(rows),
        "rule_violations": dict(Counter(p for r in rows for p in r["problems"])),
        "examples_per_query": dict(Counter(len(r["examples"]) for r in rows)),
        "examples_by_tier": dict(Counter(e["tier"] for r in rows for e in r["examples"])),
        "queries_using_site_only_fallback": sum(any(not e["same_setting"] for e in r["examples"]) for r in rows),
        "fallback_by_site": dict(Counter(r["disease_site"] for r in rows
                                         if any(not e["same_setting"] for e in r["examples"])).most_common()),
        "same_encounter_share": round(sum(e["same_encounter"] for r in rows for e in r["examples"])
                                      / max(1, sum(len(r["examples"]) for r in rows)), 3),
        "first_example_similarity": {"min": firsts[0], "median": firsts[len(firsts) // 2], "max": firsts[-1]},
    }


def case_card(record, title, extra=""):
    labels = " · ".join(escape(str(record[k])) for k in ("disease_site", "setting", "prior_rt", "encounter_type"))
    return f"""<div class="card"><h3>{escape(title)} {extra}</h3><p class="labels">{labels}</p>
<p><b>Vignette.</b> {escape(record["vignette"])}</p><p><b>Question.</b> {escape(record["clinical_question"])}</p>
<details><summary>Attending plan · technique: {escape(str(record["technique"]))}</summary>
<p>{escape(record["attending_plan"])}</p></details></div>"""


def review_page(index, rows, case_ids):
    by_id = {r["case_id"]: r for r in index.records}
    sections = []
    for case_id in case_ids:
        row = next(r for r in rows if r["case_id"] == case_id)
        examples = []
        for n, example in enumerate(row["examples"]):
            tier = example["tier"]
            rating = "".join(f'<label><input type="radio" name="{case_id}|{example["case_id"]}" value="{v}"> {v}</label>'
                             for v in ("similar", "partly", "not similar"))
            examples.append(case_card(by_id[example["case_id"]], f"Example {'AB'[n]}: {example['case_id']}",
                                      f'<span class="meta">similarity {example["similarity"]} · {tier}</span>')
                            + f'<div class="rating">{rating}</div>')
        sections.append(f'<section><div class="query">{case_card(by_id[case_id], f"Query: {case_id}")}</div>'
                        f'<div class="examples">{"".join(examples) or "<p>No examples.</p>"}</div></section>')
    return PAGE.replace("$sections", "".join(sections))


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Past-case review</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body { font-family: -apple-system, system-ui, sans-serif; margin: 0; background: #f7f7f8; color: #1c1c1e; }
header { padding: 16px 24px; background: #1c1c1e; color: #fff; position: sticky; top: 0; }
main { max-width: 1300px; margin: 0 auto; padding: 16px; }
section { display: grid; grid-template-columns: 1fr 2fr; gap: 12px; margin-bottom: 28px; }
.examples { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.card { background: #fff; border: 1px solid #e2e2e4; border-radius: 10px; padding: 12px 16px; font-size: 14px; line-height: 1.45; }
.query .card { border-color: #1c1c1e; }
h3 { margin: 0 0 4px; font-size: 15px; } .labels, .meta { color: #6b6b70; font-size: 12px; font-weight: 400; }
.rating { margin-top: 6px; font-size: 13px; } .rating label { margin-right: 10px; }
button { font-size: 14px; padding: 6px 12px; margin-left: 12px; }
@media (max-width: 900px) { section, .examples { grid-template-columns: 1fr; } }
</style></head><body>
<header><strong>Past-case review</strong> — are the examples clinically similar to the query?
<button onclick="copyRatings()">Copy ratings</button></header>
<main>$sections</main>
<script>
function copyRatings() {
  const out = {};
  document.querySelectorAll("input[type=radio]:checked").forEach(r => { out[r.name] = r.value; });
  navigator.clipboard.writeText(JSON.stringify(out, null, 1));
  alert(Object.keys(out).length + " ratings copied");
}
</script></body></html>"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=20, help="Tuning cases on the review page.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-similarity", type=float, default=None, help="Drop examples below this similarity.")
    args = parser.parse_args(argv)
    try:
        index = NumpyPastCaseIndex.load(INDEX_FOLDER)
        rows = check(index, args.min_similarity)
        summary = summarise(rows, index, args.min_similarity)
        # A floor writes to its own folder, so the unfloored review page is never overwritten
        output = OUTPUT if args.min_similarity is None else OUTPUT.with_name(f"{OUTPUT.name}_floor-{args.min_similarity}")
        evaluation = {c["case_id"] for c in json.loads((PROCESSED / "evaluation_set.json").read_text())["cases"]}
        # Reviewed cases come from outside the evaluation set, so any change they prompt is not tuned on it
        tuning = sorted(r["case_id"] for r in index.records if r["case_id"] not in evaluation)
        sample = sorted(random.Random(args.seed).sample(tuning, args.sample))
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps({**summary, "review_sample": sample}, indent=2), encoding="utf-8")
        (output / "selections.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (output / "review.html").write_text(review_page(index, rows, sample), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        print(f"Review page: {output / 'review.html'}")
        return 0 if not summary["rule_violations"] else 1
    except Exception as exc:
        print(f"Past-case check failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
