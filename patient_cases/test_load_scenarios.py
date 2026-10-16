"""Regression checks for split isolation and reference-answer separation."""

from copy import deepcopy
import unittest

from load_scenarios import (
    PLAN, QUESTION, VIGNETTE, baseline_inputs, development_examples,
    validate_records,
)


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


if __name__ == "__main__":
    unittest.main()
