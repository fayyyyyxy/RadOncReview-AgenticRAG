"""Validate the writer's example selection for every eligible development case, without AWS."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from config import Config
from core.case_labels import CaseLabels, SETTING_GROUPS
from core.institutional_examples import CaseContext, LocalInstitutionalExamples
from core.past_cases import LABEL, MIN_SIMILARITY
from load_scenarios import DEFAULT_WORKBOOK, VIGNETTE, QUESTION, development_examples, read_records, validate_records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--past-case-index", type=Path, default=Path(Config().features.past_case_index_path))
    args = parser.parse_args(argv)
    records = read_records(args.workbook)
    validate_records(records)
    development = [r for r in records if r["Split"] == "Development" and r["Worked-Example Eligible"] == "Yes"]
    source = "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in development_examples(records))
    config = Config()
    provider = LocalInstitutionalExamples(args.past_case_index, config.embedding.model_id, config.embedding.dimension,
                                         expected_source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest())
    violations, counts, selections = Counter(), Counter(), []

    def no_embedding(text):
        raise RuntimeError("Development checks must reuse existing vectors, without model calls.")

    for record in development:
        context = CaseContext(record[VIGNETTE], record[QUESTION], record["Case ID"], record["Patient ID"])
        labels = CaseLabels(record["Disease Site"], SETTING_GROUPS[record["Setting"]],
                            record["Prior RT"], record["Encounter Type"])
        examples = provider.select(context, labels, no_embedding)
        counts[str(len(examples))] += 1
        violations["more_than_two"] += len(examples) > 2
        for example in examples:
            indexed = provider.records[example["case_id"]]
            violations["query_case_selected"] += example["case_id"] == context.case_id
            violations["same_patient_selected"] += indexed["patient_id"] == context.patient_id
            violations["other_site_selected"] += indexed["disease_site"] != labels.disease_site
            violations["wrong_label"] += example["label"] != LABEL
            violations["below_similarity_floor"] += example["similarity"] < MIN_SIMILARITY
        selections.append({"case_id": context.case_id,
                           "examples": [{"case_id": e["case_id"], "similarity": e["similarity"]} for e in examples]})
    report = {**provider.provenance(), "workbook_sha256": hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
              "queries": len(development), "model_calls": 0, "examples_per_query": dict(counts),
              "rule_violations": {k: v for k, v in violations.items() if v}, "selections": selections}
    output = ROOT / "outputs/institutional_examples_preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "selections"}, indent=2))
    print(f"Local report: {output}")
    return 1 if report["rule_violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
