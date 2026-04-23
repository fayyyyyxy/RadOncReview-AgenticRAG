"""
calibration.py — Isotonic Regression Calibration
=================================================
Takes ONE input CSV with both raw judge scores and human annotator scores, fits isotonic regression per criteria,
outputs calibration CSVs.

Input CSV — one row per annotated question (columns for human evaluation may be reduced if can't be assessed):
  question
  faithfulness_score          ← raw judge score (from llm_judge_v3 CSV)
  correctness_score
  answer_quality_score
  retrieval_adequacy_score
  human_faithfulness          ← human annotator score (1-5)
  human_correctness
  human_answer_quality
  human_retrieval_adequacy

Output: calibration/ folder with one CSV per criteria:
  calibration/faithfulness.csv
  ...

Each output CSV example (5 rows):
  raw_score,calibrated_score
  1,1.0
  2,2.1
  3,2.8
  4,3.4
  5,4.3
"""

import csv
import os
import argparse
from typing import List, Dict, Tuple


CRITERIA = {
    "faithfulness":       ("faithfulness_score",      "human_faithfulness"),
    "correctness":        ("correctness_score",        "human_correctness"),
    "answer_quality":     ("answer_quality_score",     "human_answer_quality"),
    "retrieval_adequacy": ("retrieval_adequacy_score", "human_retrieval_adequacy"),
}


# Isotonic regression
def isotonic_regression(
    x: List[float], y: List[float]
) -> List[Tuple[float, float]]:
    """
    Pool Adjacent Violators algorithm.
    Fits a monotone non-decreasing mapping from x to y.
    Returns sorted list of (x_val, calibrated_y_val) pairs.
    """
    pairs  = sorted(zip(x, y), key=lambda p: p[0])
    xs     = [p[0] for p in pairs]
    ys     = [p[1] for p in pairs]

    blocks = [[ys[i], 1, [xs[i]]] for i in range(len(ys))]

    changed = True
    while changed:
        changed    = False
        new_blocks = []
        i          = 0
        while i < len(blocks):
            if i + 1 < len(blocks):
                mean_i    = blocks[i][0]   / blocks[i][1]
                mean_next = blocks[i+1][0] / blocks[i+1][1]
                if mean_i > mean_next:
                    merged = [
                        blocks[i][0]   + blocks[i+1][0],
                        blocks[i][1]   + blocks[i+1][1],
                        blocks[i][2]   + blocks[i+1][2],
                    ]
                    new_blocks.append(merged)
                    i      += 2
                    changed = True
                    continue
            new_blocks.append(blocks[i])
            i += 1
        blocks = new_blocks

    result = []
    for block in blocks:
        mean = block[0] / block[1]
        for x_val in block[2]:
            result.append((x_val, mean))

    return sorted(result, key=lambda p: p[0])


def fit_calibration_map(
    raw_scores: List[int], human_scores: List[float]
) -> Dict[int, float]:
    """
    Groups by integer raw score (1-5), computes mean human score per group,
    runs isotonic regression to ensure monotonicity.
    Returns {1: cal, 2: cal, 3: cal, 4: cal, 5: cal}.
    """
    groups: Dict[int, List[float]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for raw, human in zip(raw_scores, human_scores):
        key = max(1, min(5, int(raw)))
        groups[key].append(float(human))

    means_x, means_y = [], []
    for score in range(1, 6):
        if groups[score]:
            means_x.append(float(score))
            means_y.append(sum(groups[score]) / len(groups[score]))

    if len(means_x) < 2:
        return {s: float(s) for s in range(1, 6)}

    iso_pairs = isotonic_regression(means_x, means_y)
    iso_map   = {int(round(p[0])): round(p[1], 4) for p in iso_pairs}

    # Fill any missing integer scores
    full_map = {}
    for score in range(1, 6):
        if score in iso_map:
            full_map[score] = iso_map[score]
        else:
            nearest         = min(iso_map.keys(), key=lambda k: abs(k - score))
            full_map[score] = iso_map[nearest]

    return full_map


def run_calibration(input_csv: str, output_dir: str):

    if not os.path.exists(input_csv):
        print(f"ERROR: Input file not found: {input_csv}")
        return

    with open(input_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from {input_csv}\n")

    os.makedirs(output_dir, exist_ok=True)

    for criteria, (raw_col, human_col) in CRITERIA.items():

        raw_scores   = []
        human_scores = []
        for row in rows:
            raw   = row.get(raw_col, "").strip()
            human = row.get(human_col, "").strip()
            if raw and human:
                try:
                    raw_scores.append(int(float(raw)))
                    human_scores.append(float(human))
                except ValueError:
                    continue

        n = len(raw_scores)
        print(f"{'='*50}")
        print(f"Criteria: {criteria} ({n} labeled examples)")

        if n < 5:
            print(f"  WARNING: Only {n} examples — need at least 5. Skipping.\n")
            continue

        mapping  = fit_calibration_map(raw_scores, human_scores)

        out_path = os.path.join(output_dir, f"{criteria}.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["raw_score", "calibrated_score"])
            writer.writeheader()
            for score in sorted(mapping.keys()):
                writer.writerow({
                    "raw_score":        score,
                    "calibrated_score": mapping[score],
                })

        print(f"  Mapping (raw → calibrated):")
        for score in sorted(mapping.keys()):
            bar = "█" * round(mapping[score]) + "░" * (5 - round(mapping[score]))
            print(f"    {score} → {mapping[score]:.2f}  [{bar}]")
        print(f"  Saved: {out_path}\n")

    print(f"Calibration complete.")
    print(f"CSVs written to: {os.path.abspath(output_dir)}/")
    print(f"Place this folder next to llm_judge_v3.py to enable calibration.")


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isotonic regression calibration")
    parser.add_argument("--input",      required=True,
                        help="Path to human_labels.csv")
    parser.add_argument("--output_dir", default="calibration",
                        help="Output directory (default: calibration/)")
    args = parser.parse_args()

    run_calibration(input_csv=args.input, output_dir=args.output_dir)
