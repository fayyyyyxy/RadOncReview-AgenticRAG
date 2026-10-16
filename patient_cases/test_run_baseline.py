"""Baseline input isolation and per-case error checks without AWS."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from load_scenarios import QUESTION, VIGNETTE
from run_baseline import load_inputs, make_query, run_case
from test_load_scenarios import record


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.input = {"case_id": "D1", "vignette": "Synthetic patient description", "clinical_question": "Synthetic question"}
        self.records = [record("D1", "P1"), record("T1", "P2", "Locked test")]

    def load(self, item):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "inputs.jsonl"
            path.write_text(json.dumps(item) + "\n", encoding="utf-8")
            with patch("run_baseline.read_records", return_value=self.records):
                return load_inputs(path, Path("unused.xlsx"))

    def test_valid_development_input(self):
        self.assertEqual(self.load(self.input), [self.input])
        self.assertNotIn("REFERENCE PLAN", make_query(self.input))

    def test_reference_plan_field_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load({**self.input, "attending_plan": "Do not send"})

    def test_locked_test_id_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load({**self.input, "case_id": "T1"})

    def test_changed_text_is_rejected(self):
        with self.assertRaises(ValueError):
            self.load({**self.input, "clinical_question": "Changed question"})

    def test_oversize_query_does_not_call_engine(self):
        result = run_case(None, self.input, 1)
        self.assertEqual(result["status"], "skipped_input_too_long")

    def test_engine_failure_is_recorded(self):
        class FailingEngine:
            def reset_conversation(self):
                pass

            def query(self, query):
                raise RuntimeError("Synthetic failure")

        result = run_case(FailingEngine(), self.input, 2000)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
