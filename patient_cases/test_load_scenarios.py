"""Regression checks for split isolation and reference-answer separation."""

from copy import deepcopy
import unittest

from load_scenarios import (
    CLINICAL_METADATA_FIELDS, PLAN, QUESTION, VIGNETTE, baseline_inputs,
    development_examples, evaluation_labels, evaluation_set, validate_records,
)


def labelled(case_id, patient_id, site, prior_rt="None", encounter="New start"):
    return {**record(case_id, patient_id), "Disease Site": site, "Prior RT": prior_rt, "Encounter Type": encounter}


def record(case_id, patient_id, split="Development"):
    return {
        "Case ID": case_id, "Patient ID": patient_id, "Split": split,
        "Worked-Example Eligible": "Yes" if split == "Development" else "No",
        "LLM Use": "Worked-example index" if split == "Development" else "Evaluation only – never index",
        VIGNETTE: "Synthetic patient description", QUESTION: "Synthetic question",
        PLAN: "REFERENCE PLAN MUST NOT ENTER BASELINE INPUT",
        "Disease Site": "Synthetic site", "Subsite": None, "Setting": "Synthetic setting",
        "Prior RT": "None", "Technique": None, "Encounter Type": "New start",
    }


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.rows = [record("D1", "P1"), record("D2", "P2"), record("T1", "P3", "Locked test")]

    def test_locked_records_and_plans_do_not_enter_baseline(self):
        validate_records(self.rows)
        examples = development_examples(self.rows)
        self.assertEqual({r["case_id"] for r in examples}, {"D1", "D2"})
        for example in examples:
            self.assertNotIn(example["attending_plan"], example["search_text"])
            self.assertEqual(set(example["metadata"]), set(CLINICAL_METADATA_FIELDS))
            self.assertNotIn("patient_id", example["metadata"])
            self.assertNotIn("Age", example["metadata"])
            self.assertNotIn("Sex", example["metadata"])
            self.assertEqual(example["patient_linkage"]["patient_id"], example["case_id"].replace("D", "P"))
            self.assertEqual(example["governance"]["split"], "Development")
            self.assertNotIn("attending_plan", example["provenance"])
        for item in baseline_inputs(examples, 2, 42):
            self.assertEqual(set(item), {"case_id", "vignette", "clinical_question"})

    def test_patient_overlap_is_rejected(self):
        self.rows[-1]["Patient ID"] = "P1"
        with self.assertRaisesRegex(ValueError, "Patient overlap"):
            validate_records(self.rows)

    def test_conflicting_test_eligibility_is_rejected(self):
        self.rows[-1]["Worked-Example Eligible"] = "Yes"
        with self.assertRaisesRegex(ValueError, "conflicts with Split"):
            validate_records(self.rows)

    def test_duplicate_case_and_missing_question_are_rejected(self):
        for field, value in [("Case ID", "D1"), (QUESTION, " ")]:
            with self.subTest(field=field):
                rows = deepcopy(self.rows)
                rows[1][field] = value
                with self.assertRaises(ValueError):
                    validate_records(rows)

    def test_sampling_is_independent_of_row_order(self):
        examples = development_examples(self.rows)
        self.assertEqual(baseline_inputs(examples, 1, 42), baseline_inputs(examples[::-1], 1, 42))
        with self.assertRaises(ValueError):
            baseline_inputs(examples, 3, 42)

    def test_evaluation_set_keeps_baseline_and_fills_quotas_by_priority(self):
        # Each priority pass has exactly one eligible candidate, so the result does not depend on the seed
        rows = [
            labelled("B1", "P1", "A"), labelled("B2", "P1", "A"),         # baseline pair, one patient
            labelled("A1", "P2", "A", "Re-irradiation (overlap)"),          # fills site A's third slot
            labelled("A2", "P3", "A", "Re-irradiation (overlap)"),          # site A is full
            labelled("C1", "P1", "C", "Re-irradiation (overlap)"),          # baseline patient again
            labelled("C2", "P4", "C"), labelled("C3", "P5", "C", "Prior RT – different site"),
            labelled("C4", "P6", "C", encounter="Boost"),
        ]
        examples = development_examples(rows)
        chosen = evaluation_set(examples, ["B1", "B2"], 42, size=5)
        self.assertEqual([r["case_id"] for r in chosen], ["A1", "B1", "B2", "C3", "C4"])
        self.assertEqual(chosen, evaluation_set(examples[::-1], ["B1", "B2"], 42, size=5))
        labels = evaluation_labels(chosen, ["B1", "B2"], 42)
        self.assertEqual(labels["unique_patients"], 4)
        self.assertNotIn("vignette", labels["cases"][0])
        self.assertTrue(labels["cases"][0]["needs_cumulative_dose"])
        with self.assertRaisesRegex(ValueError, "site quotas"):
            evaluation_set(examples, ["B1", "B2"], 42, size=7)


if __name__ == "__main__":
    unittest.main()
