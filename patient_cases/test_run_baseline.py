"""Baseline input isolation and per-case error checks without AWS."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from load_scenarios import QUESTION, VIGNETTE
from run_baseline import labels_for, load_inputs, main, make_query, run_case
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

    def test_bad_or_taken_name_fails_before_any_input_is_read(self):
        with tempfile.TemporaryDirectory() as folder:
            (Path(folder) / "outputs" / "stage1_answers").mkdir(parents=True)
            with patch("run_baseline.ROOT", Path(folder)), patch("run_baseline.load_inputs") as load:
                self.assertEqual(main(["--name", "stage 1"]), 1)
                self.assertEqual(main(["--name", "stage1"]), 1)
                load.assert_not_called()

    def test_dry_run_reports_which_evidence_sources_are_on(self):
        def dry_run(*flags):
            out = io.StringIO()
            with patch("run_baseline.load_inputs", return_value=[self.input]), \
                 patch("run_baseline.labels_for", return_value={}), redirect_stdout(out):
                self.assertEqual(main(list(flags)), 0)
            return json.loads(out.getvalue())
        self.assertTrue(dry_run()["trial_papers"])
        self.assertFalse(dry_run("--no-trial-papers")["trial_papers"])

    def test_workbook_labels_reach_the_engine_but_not_the_query(self):
        rows = [{**record("D1", "P1"), "Disease Site": "Breast", "Setting": "Consolidative",
                 "Prior RT": "None", "Encounter Type": "Boost"}]
        with patch("run_baseline.read_records", return_value=rows):
            labels = labels_for(["D1"], Path("unused.xlsx"))["D1"]
        self.assertEqual((labels.disease_site, labels.setting_group), ("Breast", "Definitive"))
        seen = {}

        class Engine:
            def reset_conversation(self):
                pass

            def query(self, query, labels=None):
                seen.update(query=query, labels=labels)
                raise RuntimeError("stop after the call")

        result = run_case(Engine(), self.input, 2000, labels)
        self.assertIs(seen["labels"], labels)
        self.assertNotIn("Breast", seen["query"])
        self.assertEqual(result["case_labels"]["encounter_type"], "Boost")

    def test_engine_failure_is_recorded(self):
        class FailingEngine:
            def reset_conversation(self):
                pass

            def query(self, query, labels=None):
                raise RuntimeError("Synthetic failure")

        result = run_case(FailingEngine(), self.input, 2000)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "RuntimeError")


if __name__ == "__main__":
    unittest.main()
