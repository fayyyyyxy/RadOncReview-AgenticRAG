"""Judge reply checks, merging, blinding and isolation without AWS."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import judge_baseline as jb
from load_scenarios import PLAN, QUESTION, VIGNETTE
from run_baseline import make_query
from test_load_scenarios import record

RUBRIC = {"reference": "synthetic reference", "anchors": {name: f"{name} anchors" for name, *_ in jb.RUBRIC}}
CLINICAL_OK = {
    "accuracy": {"reason": "r", "score": 4}, "completeness": {"reason": "r", "score": 3},
    "guideline_adherence": {"reason": "r", "score": 4}, "harm": {"reason": "r", "quote": "", "score": 1},
    "appropriate_abstention": {"reason": "r", "value": "NA"},
}
GROUNDING_OK = {
    "hallucination": {"reason": "r", "fabricated": [], "value": "N"},
    "citations_verifiable": {"reason": "r", "failing_citations": [], "value": "Y"},
    "unsupported_claims": [],
}


class FakeJudge:
    """Returns canned replies in order, separately for clinical and grounding prompts."""

    def __init__(self, clinical, grounding):
        self.replies = {"clinical": list(clinical), "grounding": list(grounding)}
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        reply = self.replies["grounding" if "Passages the AI was given" in prompt else "clinical"].pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


def case(case_id="D1", answer="Synthetic answer [1]."):
    return {
        "case_id": case_id, "disease_site": "Synthetic site", "vignette": "Synthetic patient description",
        "question": "Synthetic question", "plan": "Synthetic plan", "answer": answer,
        "passages": [{"citation_id": n, "source_path": f"source {n}", "text": f"passage {n}", "topic": "T"}
                     for n in (1, 2, 3)],
    }


def scores(acc=4, comp=4, adh=4, harm=0, halluc="N", cites="Y", abstain="NA"):
    return {
        "accuracy": {"score": acc}, "completeness": {"score": comp}, "guideline_adherence": {"score": adh},
        "harm": {"score": harm}, "hallucination": {"value": halluc},
        "citations_verifiable": {"value": cites}, "appropriate_abstention": {"value": abstain},
    }


class JudgeTests(unittest.TestCase):
    def test_bad_reply_is_retried_then_left_out_of_means(self):
        off_scale = {**CLINICAL_OK, "harm": {"reason": "r", "quote": "", "score": 4}}
        broken = FakeJudge(["not json", off_scale, CLINICAL_OK], [GROUNDING_OK, GROUNDING_OK])
        failed = jb.judge_case(broken, case("D1"), RUBRIC)
        self.assertEqual(failed["status"], "judge_error")
        self.assertEqual(len(broken.prompts), 5)  # one retry, then the second pass still runs

        good = FakeJudge([CLINICAL_OK] * 2, [GROUNDING_OK] * 2)
        scored = jb.judge_case(good, case("D2"), RUBRIC)
        summary = jb.summarize([failed, scored], {})
        self.assertEqual(summary["judge_errors"], ["D1"])
        self.assertEqual(summary["means"]["accuracy"], 4.0)

    def test_harm_2_or_more_needs_a_quote(self):
        answer = "Give the synthetic dose."
        reply = lambda quote: json.dumps({**CLINICAL_OK, "harm": {"reason": "r", "quote": quote, "score": 2}})
        keys = [jb.KEY[n] for n in jb.CLINICAL]
        with self.assertRaises(ValueError):
            jb.parse_reply(reply(""), keys, answer)
        self.assertTrue(jb.parse_reply(reply("Give the  synthetic dose."), keys, answer)["harm"]["quote_found"])
        # A paraphrase is kept and flagged, so a harmful answer is never dropped.
        self.assertFalse(jb.parse_reply(reply("Give a dose."), keys, answer)["harm"]["quote_found"])

    def write_results(self, folder, case_id, response=None):
        source = record(case_id, "P1" if case_id == "D1" else "P2", "Development" if case_id == "D1" else "Locked test")
        result = {"case_id": case_id, "status": "response_generated", "response": response,
                  "query": make_query({"vignette": source[VIGNETTE], "clinical_question": source[QUESTION]})}
        (folder / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")

    def test_locked_test_case_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            self.write_results(Path(name), "T1")
            with patch("judge_baseline.read_records", return_value=[record("D1", "P1"), record("T1", "P2", "Locked test")]):
                with self.assertRaises(ValueError):
                    jb.load_cases(Path(name), Path("unused.xlsx"))

    def test_judge_is_blinded(self):
        response = {
            "answer": "Synthetic answer [1].", "confidence_score": 0.613,
            "self_critique": {"feedback": "SELF-CRITIQUE-MARKER"},
            "citations": [{"citation_id": 1, "chunk_text": "passage 1"}],
            "context_passages": case()["passages"], "metadata": {"retrieval": {"total_chunks": 3}},
        }
        with tempfile.TemporaryDirectory() as name:
            run = Path(name) / "RUN-ID-MARKER"
            run.mkdir()
            self.write_results(run, "D1", response)
            with patch("judge_baseline.read_records", return_value=[record("D1", "P1")]):
                cases, skipped = jb.load_cases(run, Path("unused.xlsx"))
        self.assertEqual(skipped, {})
        clinical, grounding = jb.clinical_prompt(cases[0], RUBRIC), jb.grounding_prompt(cases[0], RUBRIC)
        for prompt in (clinical, grounding):
            for hidden in ("0.613", "SELF-CRITIQUE-MARKER", "RUN-ID-MARKER"):
                self.assertNotIn(hidden, prompt)
        self.assertIn(record("D1", "P1")[PLAN], clinical)
        self.assertNotIn(record("D1", "P1")[PLAN], grounding)

    def test_citation_number_without_passage_fails_in_code(self):
        fake = FakeJudge([CLINICAL_OK], [GROUNDING_OK])  # the model says Citations Verifiable = Y
        result = jb.judge_run(fake, case(answer="Synthetic answer [4]."), RUBRIC)
        cites = result["scores"]["citations_verifiable"]
        self.assertEqual((cites["value"], cites["failing_citations"], cites["model_value"]), ("N", [4], "Y"))

    def test_two_runs_merge_with_workbook_rules(self):
        final, flags = jb.merge(scores(acc=5, adh=3, harm=1, halluc="N", cites="Y", abstain="Y"),
                                scores(acc=3, adh=4, harm=2, halluc="Y", cites="N", abstain="NA"))
        self.assertEqual(final, {"accuracy": 4.0, "completeness": 4.0, "guideline_adherence": 3.5, "harm": 2,
                                 "hallucination": "Y", "citations_verifiable": "N", "appropriate_abstention": "Y"})
        self.assertEqual(flags, ["accuracy differs by 2+", "harm differs", "hallucination differs", "harm 2+",
                                 "citations differ", "abstention differs"])
        final, flags = jb.merge(scores(abstain="Y"), scores(abstain="N"))
        self.assertEqual((final["appropriate_abstention"], flags), ("N", ["abstention differs"]))
        self.assertEqual(jb.merge(scores(), scores())[1], [])

    def test_dry_run_makes_no_model_calls(self):
        judge = Mock(side_effect=AssertionError("dry run must not create a judge"))
        with patch("judge_baseline.load_rubric", return_value=RUBRIC), \
                patch("judge_baseline.load_cases", return_value=([case()], {})), \
                patch("judge_baseline.BedrockJudge", judge), redirect_stdout(io.StringIO()):
            self.assertEqual(jb.main(["unused-run"]), 0)
        judge.assert_not_called()


if __name__ == "__main__":
    unittest.main()
