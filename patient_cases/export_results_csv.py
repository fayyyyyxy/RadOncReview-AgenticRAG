"""Export baseline/pilot answers and rubric scores to CSV for spreadsheet review."""

import argparse
import csv
import json
from pathlib import Path

from load_scenarios import DEFAULT_WORKBOOK, QUESTION, VIGNETTE, read_records, validate_records


SCORE_COLUMNS = (
    "accuracy",
    "completeness",
    "guideline_adherence",
    "harm",
    "hallucination",
    "citations_verifiable",
    "appropriate_abstention",
)


def read_jsonl(path):
    if not path.exists():
        return {}
    records = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                records[row["case_id"]] = row
    return records


def read_scores(path):
    if not path:
        return {}
    scores_path = Path(path)
    if scores_path.is_dir():
        scores_path = scores_path / "scores.jsonl"
    return read_jsonl(scores_path)


def response_for(result):
    if not result:
        return "missing_run"
    return result.get("status", "missing_status")


def answer_for(result):
    if not result:
        return ""
    response = (result.get("response") or {})
    return response.get("answer", "")


def retrieval_for(result):
    if not result:
        return "missing_run", "", "", ""
    retrieval = (result.get("response") or {}).get("metadata", {}).get("retrieval", {})
    tasks_completed = retrieval.get("tasks_completed", "")
    tasks_total = retrieval.get("tasks_total", "")
    total_chunks = retrieval.get("total_chunks", "")
    response = (result.get("response") or {})
    task_plan = response.get("task_plan") or {}
    tasks = task_plan.get("sub_tasks", []) if isinstance(task_plan, dict) else []
    failed_tasks = sum(task.get("status") == "failed" for task in tasks)
    return tasks_completed, tasks_total, total_chunks, failed_tasks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Baseline *_answers directory")
    parser.add_argument("--pilot", type=Path, required=True, help="Pilot *_answers directory")
    parser.add_argument("--baseline-scores", type=Path, help="Baseline scores.jsonl or *_scores directory")
    parser.add_argument("--pilot-scores", type=Path, help="Pilot scores.jsonl or *_scores directory")
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--output", type=Path, help="Output CSV path")
    args = parser.parse_args(argv)

    records = read_records(args.workbook)
    validate_records(records)
    workbook = {row["Case ID"]: row for row in records if row["Split"] == "Development"}

    baseline = read_jsonl(args.baseline / "results.jsonl")
    pilot = read_jsonl(args.pilot / "results.jsonl")
    baseline_scores = read_scores(args.baseline_scores)
    pilot_scores = read_scores(args.pilot_scores)

    case_ids = sorted(set(workbook) & (set(baseline) | set(pilot)))
    output = args.output or args.pilot.parent / "pilot_comparison.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "case_id", "disease_site", "vignette", "clinical_question",
        "baseline_status", "baseline_tasks_completed", "baseline_tasks_total",
        "baseline_total_chunks", "baseline_failed_tasks", "baseline_answer",
        "pilot_status", "pilot_tasks_completed", "pilot_tasks_total",
        "pilot_total_chunks", "pilot_failed_tasks", "pilot_answer",
    ]
    columns += [f"baseline_{column}" for column in SCORE_COLUMNS]
    columns += [f"pilot_{column}" for column in SCORE_COLUMNS]
    columns += ["baseline_score_flags", "pilot_score_flags"]

    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for case_id in case_ids:
            row = workbook[case_id]
            baseline_result = baseline.get(case_id)
            pilot_result = pilot.get(case_id)
            baseline_record = baseline_scores.get(case_id)
            pilot_record = pilot_scores.get(case_id)
            baseline_final = baseline_record.get("final", {}) if baseline_record else {}
            pilot_final = pilot_record.get("final", {}) if pilot_record else {}

            baseline_retrieval = retrieval_for(baseline_result)
            pilot_retrieval = retrieval_for(pilot_result)
            output_row = {
                "case_id": case_id,
                "disease_site": row["Disease Site"],
                "vignette": row[VIGNETTE],
                "clinical_question": row[QUESTION],
                "baseline_status": response_for(baseline_result),
                "baseline_tasks_completed": baseline_retrieval[0],
                "baseline_tasks_total": baseline_retrieval[1],
                "baseline_total_chunks": baseline_retrieval[2],
                "baseline_failed_tasks": baseline_retrieval[3],
                "baseline_answer": answer_for(baseline_result),
                "pilot_status": response_for(pilot_result),
                "pilot_tasks_completed": pilot_retrieval[0],
                "pilot_tasks_total": pilot_retrieval[1],
                "pilot_total_chunks": pilot_retrieval[2],
                "pilot_failed_tasks": pilot_retrieval[3],
                "pilot_answer": answer_for(pilot_result),
            }
            for column in SCORE_COLUMNS:
                output_row[f"baseline_{column}"] = baseline_final.get(column, "") if baseline_record else ""
                output_row[f"pilot_{column}"] = pilot_final.get(column, "") if pilot_record else ""
            output_row["baseline_score_flags"] = "; ".join(baseline_record.get("flags", [])) if baseline_record else ""
            output_row["pilot_score_flags"] = "; ".join(pilot_record.get("flags", [])) if pilot_record else ""
            writer.writerow(output_row)

    print(json.dumps({"output": str(output), "rows": len(case_ids)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
