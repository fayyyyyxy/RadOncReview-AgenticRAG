"""Score past-case selection against the attending plans it never sees; no model calls.

Selection may use only what is known when a question is asked: the labels and the vignette plus question. The
attending plan is the answer, so it only grades the choice: does a chosen case share the query's technique or dose
regimen? Each way of choosing is compared with random choice from the same pool (layered baselines).

Tuning runs development queries against development cases only, so evaluation-case plans never shape the
similarity floor. The terminal and summary.json show metrics only.
"""

import argparse
import json
from pathlib import Path
import random
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))

from core.case_labels import CaseLabels
from core.past_cases import MIN_SIMILARITY, NumpyPastCaseIndex, PastCaseHit, PastCaseIndex, select_examples
from build_past_case_index import INDEX_FOLDER, PROCESSED
from check_past_cases import labels_of

OUTPUT = ROOT / "outputs" / "past_case_quality"
UNKNOWN_TECHNIQUE = "EBRT – technique not specified"
METRICS = ("technique", "regimen")

# 2 x 2 design: whether the other labels filter the pool, and whether text similarity orders it; site always does
WAYS = {
    "site, random": (False, False),
    "site, similarity": (False, True),
    "labels, random": (True, False),
    "labels, similarity (current rule)": (True, True),
}
# Each contrast changes one factor, so it isolates what that factor adds
CONTRASTS = {
    "labels, in random order": ("labels, random", "site, random"),
    "labels, in similarity order": ("labels, similarity (current rule)", "site, similarity"),
    "similarity, site only": ("site, similarity", "site, random"),
    "similarity, with labels": ("labels, similarity (current rule)", "labels, random"),
}
FLOORS = (None, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7)
BIN_EDGES = (0.4, 0.5, 0.6, 0.7, 0.8)

NUMBER = r"\d+(?:\.\d+)?"
# "50 Gy in 25 fractions", "5000 cGy/25 fx", "70/56 Gy in 35 fractions" (SIB), "40 Gy in 5 daily fractions"
DOSE_IN_FRACTIONS = re.compile(
    rf"(?P<doses>(?:{NUMBER}\s*/\s*)*{NUMBER})\s*(?P<unit>c?Gy)\s*(?:in|/|over|,)?\s*"
    rf"(?:(?P<count>\d+)\s*(?:\w+\s+)?(?:fx|fxs|fractions?)|(?:a\s+)?(?P<single>single)\s+fraction)\b", re.I)
# "8 Gy x 1" and "5 x 5 Gy": dose per fraction times the number of fractions
TIMES_AFTER = re.compile(rf"(?P<dose>{NUMBER})\s*(?P<unit>c?Gy)\s*[x×]\s*(?P<count>\d+)\b", re.I)
TIMES_BEFORE = re.compile(rf"\b(?P<count>\d+)\s*[x×]\s*(?P<dose>{NUMBER})\s*(?P<unit>c?Gy)\b", re.I)
PRIOR_COURSE = re.compile(r"\b(?:prior|previous(?:ly)?|received|had|history of|initial course|first course)\b", re.I)


def gray(value, unit):
    return float(value) / 100 if unit.lower() == "cgy" else float(value)


def earlier_course(text, start):
    """True when the clause before a dose names an earlier course ("prior 60 Gy in 30 fractions")."""
    # ⚠️ Cut at the clause start, but not at a decimal point: "1.8 Gy" must not end the clause
    clause = re.split(r"[;\n]|\.(?!\d)", text[max(0, start - 40):start])[-1]
    return bool(PRIOR_COURSE.search(clause))


def regimens(text):
    """(total dose in Gy, fractions) pairs prescribed in a plan.

    A dose with no stated number of fractions ("20 Gy to the lesion", "total dose 54 Gy") is left out rather than
    guessed, and so is any dose given for an earlier course.
    """
    found = set()
    for match in DOSE_IN_FRACTIONS.finditer(text):
        if earlier_course(text, match.start()):
            continue
        count = 1 if match["single"] else int(match["count"])
        # 💡 A simultaneous boost lists several dose levels over one number of fractions: one pair per level
        for dose in re.split(r"\s*/\s*", match["doses"]):
            found.add((round(gray(dose, match["unit"]), 2), count))
    for pattern in (TIMES_AFTER, TIMES_BEFORE):
        for match in pattern.finditer(text):
            if not earlier_course(text, match.start()):
                count = int(match["count"])
                found.add((round(gray(match["dose"], match["unit"]) * count, 2), count))
    return frozenset(found)


class ShuffledIndex(PastCaseIndex):
    """The same filters and exclusions as `index` in random order: choice without text similarity."""

    def __init__(self, index, rng):
        self.index, self.rng = index, rng

    def add(self, records, vectors):
        raise NotImplementedError("A baseline index is read-only.")

    def search(self, query, filters, exclude, k):
        hits = self.index.search(query, filters, exclude, len(self.index.records))
        # ⚠️ Replace the similarity, not only the order: select_examples re-sorts its window by similarity
        shuffled = [PastCaseHit(h.record, self.rng.random()) for h in hits]
        return sorted(shuffled, key=lambda h: -h.similarity)[:k]

    def version(self):
        return self.index.version()


def subset(index, keep):
    """A copy of `index` holding only the records `keep` accepts."""
    rows = [i for i, r in enumerate(index.records) if keep(r)]
    part = NumpyPastCaseIndex(index.dimension)
    part.records, part.vectors = [index.records[i] for i in rows], index.vectors[rows]
    return part


def choose(index, position, use_labels, ranked, rng=None, min_similarity=None):
    """Examples for the indexed case at `position`, chosen one of the WAYS."""
    query = index.records[position]
    labels = labels_of(query) if use_labels else CaseLabels(query["disease_site"])
    search = index if ranked else ShuffledIndex(index, rng)
    hits = select_examples(search, index.vectors[position], labels, query["case_id"], query["patient_id"],
                           min_similarity=min_similarity)
    return [h.record for h in hits]


def gradable(query, plan_regimens):
    """Which metrics the query's own plan can grade: it must state a technique, or a regimen."""
    return {"technique": query["technique"] != UNKNOWN_TECHNIQUE, "regimen": bool(plan_regimens[query["case_id"]])}


def match_flags(query, examples, plan_regimens):
    """Per metric, whether each example shares the query's technique or one of its regimens.

    An example whose plan states neither counts as no match: it shows the physician nothing about that choice.
    """
    wanted = plan_regimens[query["case_id"]]
    return {"technique": [e["technique"] == query["technique"] for e in examples],
            "regimen": [bool(plan_regimens[e["case_id"]] & wanted) for e in examples]}


def per_query(index, positions, repeats, seed, plan_regimens):
    """Share of matching examples per query and way; random ways average `repeats` draws."""
    rng = random.Random(seed)
    table = {way: {metric: {} for metric in METRICS} for way in WAYS}
    for position in positions:
        query = index.records[position]
        can = gradable(query, plan_regimens)
        for way, (use_labels, ranked) in WAYS.items():
            draws = [match_flags(query, choose(index, position, use_labels, ranked, rng), plan_regimens)
                     for _ in range(1 if ranked else repeats)]
            for metric in METRICS:
                shares = [np.mean(d[metric]) for d in draws if d[metric]]
                if can[metric] and shares:
                    table[way][metric][query["case_id"]] = float(np.mean(shares))
    return table


def bootstrap(values, patients, draws, seed):
    """Means of `values` over resampled patients.

    A patient's encounters are drawn together: they resemble each other, so treating them as independent cases
    would make the interval too narrow.
    """
    groups = {p: i for i, p in enumerate(dict.fromkeys(patients))}
    group_of = np.array([groups[p] for p in patients])
    rng = np.random.default_rng(seed)
    # 💡 Drawing patients with replacement = giving each patient a count; a query weighs its patient's count
    counts = np.stack([np.bincount(rng.integers(0, len(groups), len(groups)), minlength=len(groups))
                       for _ in range(draws)])
    weights = counts[:, group_of]
    return (weights * np.asarray(values, dtype=float)).sum(axis=1) / weights.sum(axis=1)


def interval(values, patients, draws, seed):
    means = bootstrap(values, patients, draws, seed)
    return {"mean": round(float(np.mean(values)), 3), "low": round(float(np.percentile(means, 2.5)), 3),
            "high": round(float(np.percentile(means, 97.5)), 3), "queries": len(values)}


def compare_ways(table, patient_of, draws, seed):
    """Each way's mean and each contrast's paired difference, with 95% patient-bootstrap intervals."""
    result = {}
    for metric in METRICS:
        ways = {}
        for way in WAYS:
            ids = sorted(table[way][metric])
            ways[way] = interval([table[way][metric][i] for i in ids], [patient_of[i] for i in ids], draws, seed)
        contrasts = {}
        for name, (after, before) in CONTRASTS.items():
            # Paired: the same queries under both ways, so the difference cancels how hard each query is
            ids = sorted(set(table[after][metric]) & set(table[before][metric]))
            differences = [table[after][metric][i] - table[before][metric][i] for i in ids]
            contrasts[name] = interval(differences, [patient_of[i] for i in ids], draws, seed)
        result[metric] = {"ways": ways, "contrasts": contrasts}
    return result


def bin_name(similarity):
    lower = [edge for edge in BIN_EDGES if similarity >= edge]
    if not lower:
        return f"< {BIN_EDGES[0]:.2f}"
    upper = [edge for edge in BIN_EDGES if edge > lower[-1]]
    return f"{lower[-1]:.2f}–{upper[0]:.2f}" if upper else f"≥ {lower[-1]:.2f}"


def calibration(index, positions, plan_regimens):
    """Match rate of every same-site candidate by its similarity to the query: does similarity predict the plan?"""
    names = [bin_name(0.0)] + [bin_name(edge) for edge in BIN_EDGES]
    counts = {name: {"pairs": 0, **{m: [0, 0] for m in METRICS}} for name in names}
    for position in positions:
        query = index.records[position]
        can = gradable(query, plan_regimens)
        similarities = index.vectors @ index.vectors[position]
        candidates = [j for j, r in enumerate(index.records)
                      if r["disease_site"] == query["disease_site"] and r["patient_id"] != query["patient_id"]]
        flags = match_flags(query, [index.records[j] for j in candidates], plan_regimens)
        for n, j in enumerate(candidates):
            row = counts[bin_name(float(similarities[j]))]
            row["pairs"] += 1
            for metric in METRICS:
                if can[metric]:
                    row[metric][0] += flags[metric][n]
                    row[metric][1] += 1
    return {name: {"pairs": row["pairs"],
                   **{m: round(row[m][0] / row[m][1], 3) if row[m][1] else None for m in METRICS}}
            for name, row in counts.items()}


def floor_sweep(index, positions, plan_regimens):
    """The current rule under each similarity floor: how many queries keep examples, and how good they are.

    `match_rate` averages over queries shown at least one example; `useful` is the share of all gradable queries
    shown at least one matching example, so it also falls when a floor leaves a query with nothing.
    """
    rows = []
    for floor in FLOORS:
        shown = {"2": 0, "1": 0, "0": 0}
        rates = {m: [] for m in METRICS}
        useful = {m: [] for m in METRICS}
        for position in positions:
            query = index.records[position]
            examples = choose(index, position, True, True, min_similarity=floor)
            shown[str(len(examples))] += 1
            can = gradable(query, plan_regimens)
            flags = match_flags(query, examples, plan_regimens)
            for metric in METRICS:
                if can[metric]:
                    useful[metric].append(any(flags[metric]))
                    if examples:
                        rates[metric].append(np.mean(flags[metric]))
        rows.append({"floor": floor, "queries_with_examples": shown,
                     **{m: {"match_rate": round(float(np.mean(rates[m])), 3) if rates[m] else None,
                            "useful": round(float(np.mean(useful[m])), 3)} for m in METRICS}})
    return rows


def evaluation_ids():
    cases = json.loads((PROCESSED / "evaluation_set.json").read_text(encoding="utf-8"))["cases"]
    return {c["case_id"] for c in cases}


def percent(value):
    return "—" if value is None else f"{value:.1%}"


def report(summary):
    """Terminal tables; metrics only."""
    print(f"Queries: {summary['queries']} ({summary['split']}); candidates: {summary['candidates']}; "
          f"95% intervals resample patients")
    for metric in METRICS:
        print(f"\n{metric.capitalize()} match (share of chosen examples)")
        for way, row in summary[metric]["ways"].items():
            print(f"  {way:36s} {percent(row['mean']):>6s}  [{percent(row['low'])}, {percent(row['high'])}]  "
                  f"n={row['queries']}")
        print("  What each factor adds (paired difference, points):")
        for name, row in summary[metric]["contrasts"].items():
            print(f"  {name:36s} {100 * row['mean']:+6.1f}  [{100 * row['low']:+.1f}, {100 * row['high']:+.1f}]")
    print("\nSimilarity vs match rate, every same-site candidate")
    for name, row in summary["calibration"].items():
        print(f"  {name:12s} pairs {row['pairs']:5d}   technique {percent(row['technique']):>6s}   "
              f"regimen {percent(row['regimen']):>6s}")
    print("\nSimilarity floor on the current rule (examples per query 2/1/0; match rate; useful)")
    for row in summary["floors"]:
        shown = row["queries_with_examples"]
        floor = "off" if row["floor"] is None else f"{row['floor']:.2f}"
        chosen = "  ← in use" if row["floor"] == MIN_SIMILARITY else ""
        print(f"  floor {floor:4s}  {shown['2']:3d}/{shown['1']:3d}/{shown['0']:3d}   "
              + "   ".join(f"{m} {percent(row[m]['match_rate'])} / {percent(row[m]['useful'])}" for m in METRICS)
              + chosen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=50, help="Random draws per query for the random ways.")
    parser.add_argument("--draws", type=int, default=2000, help="Bootstrap resamples of patients.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", choices=("development", "evaluation"), default="development",
                        help="development tunes on the cases outside the evaluation set; evaluation reports on it.")
    args = parser.parse_args(argv)
    try:
        full = NumpyPastCaseIndex.load(INDEX_FOLDER)
        evaluation = evaluation_ids()
        if args.split == "development":
            # ⚠️ Tuning searches development cases only, so no evaluation plan grades a tuning query
            index = subset(full, lambda r: r["case_id"] not in evaluation)
            positions = range(len(index.records))
        else:
            # Reporting searches every case, as real use would; each query still excludes itself and its patient
            index = full
            positions = [i for i, r in enumerate(full.records) if r["case_id"] in evaluation]
        plan_regimens = {r["case_id"]: regimens(r["attending_plan"]) for r in index.records}
        patient_of = {r["case_id"]: r["patient_id"] for r in index.records}
        table = per_query(index, positions, args.repeats, args.seed, plan_regimens)
        summary = {
            "index_version": full.version(), "split": args.split, "queries": len(positions),
            "candidates": len(index.records), "repeats": args.repeats, "draws": args.draws, "seed": args.seed,
            **compare_ways(table, patient_of, args.draws, args.seed),
            "calibration": calibration(index, positions, plan_regimens),
            "floors": floor_sweep(index, positions, plan_regimens),
        }
        OUTPUT.mkdir(parents=True, exist_ok=True)
        path = OUTPUT / f"{args.split}.json"
        path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        report(summary)
        print(f"\nWrote {path}")
        return 0
    except Exception as exc:
        print(f"Quality check failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
