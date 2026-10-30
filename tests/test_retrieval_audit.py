"""Audit failures must remain visible even when final passages look stable."""
import copy
import unittest

from patient_cases.check_retrieval_consistency import summarize
from patient_cases.review_retrieval_relevance import build_packet, validate_screen, render


class ConsistencyAuditTests(unittest.TestCase):
    def run_record(self):
        return {"case_id": "SYN-1", "healthy": True, "passages": [("notes", "1", "text")],
                "query_plan_hash": "q", "candidate_pool_hash": "pool", "ordered_passage_hash": "ordered"}

    def test_changed_candidates_fail_even_when_selected_passages_match(self):
        first = self.run_record()
        second = {**first, "candidate_pool_hash": "different"}
        result = summarize([first, second])
        self.assertFalse(result["passed"])
        self.assertTrue(result["same_ordered_passage_hash"])
        self.assertEqual(result["jaccard_vs_first"], [1.0])

    def test_failed_backend_is_not_a_successful_consistency_result(self):
        first = self.run_record()
        result = summarize([first, {**first, "healthy": False}])
        self.assertFalse(result["passed"])
        self.assertFalse(result["healthy"])

    def test_same_passages_with_different_order_do_not_pass(self):
        first = self.run_record()
        result = summarize([first, {**first, "ordered_passage_hash": "different order"}])
        self.assertFalse(result["passed"])


class RelevanceAuditTests(unittest.TestCase):
    def setUp(self):
        self.passages = [{"id": "P1", "text": "Synthetic source text about study population."}]
        self.screen = {"passages": [{"id": "P1", "label": "partial", "reason": "Background only",
                                    "quote": "source text", "flags": []}],
                       "coverage": [{"aspect": "Requested question", "status": "partial",
                                     "passage_ids": ["P1"], "reason": "Some background"}]}

    def test_fabricated_supporting_quote_is_rejected(self):
        self.screen["passages"][0]["quote"] = "An invented claim absent from the source"
        with self.assertRaises(ValueError):
            validate_screen(self.screen, self.passages)

    def test_missing_assessment_cannot_silently_inflate_relevance(self):
        with self.assertRaises(ValueError):
            validate_screen(self.screen, self.passages + [{"id": "P2", "text": "Other source"}])

    def test_unsupported_coverage_cannot_reference_unknown_passages(self):
        self.screen["coverage"][0]["passage_ids"] = ["P99"]
        with self.assertRaises(ValueError):
            validate_screen(self.screen, self.passages)

    def test_union_keeps_changed_passages_and_does_not_count_repeats_twice(self):
        first = {"case_id": "SYN-1", "input": {"vignette": "<script>bad()</script>", "clinical_question": "Question"},
                 "labels": {"disease_site": "Synthetic"}, "repeat": 1,
                 "selected_chunks": [{"faiss_id": 1, "text": "First", "path": "one"}]}
        second = copy.deepcopy(first)
        second["repeat"] = 2
        second["selected_chunks"].append({"faiss_id": 2, "text": "Changed result", "path": "two"})
        packet = build_packet([first, second])
        self.assertEqual(len(packet[0]["passages"]), 2)
        self.assertEqual(len(packet[0]["passages"][0]["appearances"]), 2)
        page = render(packet, {}, {})
        self.assertNotIn("<script>", page)
        self.assertIn("&lt;script&gt;", page)


if __name__ == "__main__":
    unittest.main()
