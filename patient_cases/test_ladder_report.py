"""Ladder noise, comparison and registry checks on synthetic judge runs; no AWS."""

import json
from pathlib import Path
import tempfile
import unittest

from ladder_report import FROZEN, compare, ladder, noise, register, reject, render, sha256


def final(accuracy=3.0, harm=1, hallucination="N", citations="N", abstention="NA"):
    return {"accuracy": accuracy, "completeness": 3.0, "guideline_adherence": 3.0, "harm": harm,
            "hallucination": hallucination, "citations_verifiable": citations, "appropriate_abstention": abstention}


def values(**columns):
    """Per-case values for metrics given as lists; the rest are constant."""
    n = len(next(iter(columns.values())))
    base = {"accuracy": 3.0, "completeness": 3.0, "guideline_adherence": 3.0, "harm_2plus": 0.0,
            "hallucination": 0.0, "citations_verifiable": 0.0, "abstention_failure": 0.0}
    return {f"C{i}": {**base, **{k: float(v[i]) for k, v in columns.items()}} for i in range(n)}


# C3 is never answered unless a test says so, like a case that fails in every run
CASES = ("C1", "C2", "C3")


class NoiseTests(unittest.TestCase):
    def test_noise_uses_paired_differences_with_one_case_floor(self):
        same = noise(values(accuracy=[3, 3, 3, 3]), values(accuracy=[3, 3, 3, 3]))
        self.assertEqual(same["accuracy"], 0.25)
        # Differences -1, 1, 0, 0: sd = sqrt(2/3), so 2 * sd / sqrt(4) = sqrt(2/3)
        wobbly = noise(values(accuracy=[2, 4, 3, 3]), values(accuracy=[3, 3, 3, 3]))
        self.assertAlmostEqual(wobbly["accuracy"], (2 / 3) ** 0.5)
        # Every case one point lower: sd = 0, but the unchanged system still moved by a whole point
        shifted = noise(values(accuracy=[2, 2, 2, 2]), values(accuracy=[3, 3, 3, 3]))
        self.assertEqual(shifted["accuracy"], 1.0)

    def test_compare_respects_each_metric_direction(self):
        before = values(accuracy=[3, 3, 3, 3], hallucination=[0, 0, 0, 0], harm_2plus=[1, 1, 0, 0])
        after = values(accuracy=[2, 2, 2, 2], hallucination=[1, 1, 0, 0], harm_2plus=[0, 0, 0, 0])
        thresholds = noise(before, before)
        result = compare(before, after, thresholds)["metrics"]
        self.assertEqual(result["accuracy"]["verdict"], "worse")
        self.assertEqual(result["hallucination"]["verdict"], "worse")    # more hallucination is worse
        self.assertEqual(result["harm_2plus"]["verdict"], "better")      # fewer harmful answers is better
        self.assertEqual(result["citations_verifiable"]["verdict"], "within noise")
        # One flipped case equals the floor, so it is still noise
        one = compare(before, values(accuracy=[3, 3, 3, 3], hallucination=[1, 0, 0, 0], harm_2plus=[1, 1, 0, 0]), thresholds)
        self.assertFalse(one["stop"])


class RegistryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.inputs = self.root / "evaluation_inputs.jsonl"
        self.inputs.write_text("".join(json.dumps({"case_id": c}) + "\n" for c in CASES), encoding="utf-8")
        self.registry = self.root / "ladder.json"
        self.judge = {key: "frozen" for key in FROZEN}

    def tearDown(self):
        self.folder.cleanup()

    def run_folder(self, name, finals, input_sha=None, writer="llama", answers=None, **judge_changes):
        answers = answers or f"baseline_{name}"
        (self.root / answers).mkdir()
        llm = {"model_id": writer, "temperature": 0.0, "max_tokens": 2048, "top_p": 0.9}
        (self.root / answers / "manifest.json").write_text(json.dumps({
            "input_sha256": input_sha or sha256(self.inputs), "case_ids": list(CASES),
            "config": {"llm": llm}}), encoding="utf-8")
        judge = self.root / f"judge_{name}"
        judge.mkdir()
        # Cases left out of `finals` stand for failed generations, which the judge skips
        skipped = {case: "error" for case in CASES if case not in finals}
        (judge / "manifest.json").write_text(json.dumps({
            **self.judge, **judge_changes, "baseline_run": answers,
            "case_ids": sorted(finals), "skipped": skipped}), encoding="utf-8")
        (judge / "scores.jsonl").write_text("".join(json.dumps({
            "case_id": case, "disease_site": "Site", "status": "scored", "final": f,
            "runs": [{"scores": {"accuracy": {"reason": "SECRET-REASON", "score": 3}}}],
        }) + "\n" for case, f in finals.items()), encoding="utf-8")
        return judge

    def add(self, stage, judge):
        return register(stage, judge, self.registry, self.inputs)

    def test_stages_must_be_filled_in_order(self):
        first = self.run_folder("a", {"C1": final(), "C2": final()})
        self.add(0, first)
        with self.assertRaisesRegex(ValueError, "Stage 0 must be complete"):
            self.add(1, self.run_folder("b", {"C1": final(), "C2": final()}))
        with self.assertRaisesRegex(ValueError, "already registered"):
            self.add(0, first)
        self.add(0, self.run_folder("c", {"C1": final(), "C2": final(accuracy=2.0)}))
        with self.assertRaisesRegex(ValueError, "already has its runs"):
            self.add(0, self.run_folder("d", {"C1": final(), "C2": final()}))
        self.add(1, self.run_folder("e", {"C1": final(accuracy=1.0), "C2": final(accuracy=1.0)}))
        registry = json.loads(self.registry.read_text(encoding="utf-8"))
        built = ladder(registry, self.root, labels=self.root / "absent.json")
        self.assertTrue(built["comparisons"][1]["stop"])
        page = render(registry, built)
        self.assertIn("<strong>STOP</strong>", page)
        self.assertNotIn("SECRET-REASON", page)     # the report never reads the judge's reasons

    def test_unanswered_cases_count_against_a_stage(self):
        everything = {case: final() for case in CASES}
        self.add(0, self.run_folder("a", everything))
        self.add(0, self.run_folder("b", everything))
        # Stage 1 fails two of three cases; C1 scores the same, so the rubric metrics alone would not notice
        self.add(1, self.run_folder("c", {"C1": final()}))
        registry = json.loads(self.registry.read_text(encoding="utf-8"))
        built = ladder(registry, self.root, labels=self.root / "absent.json")
        self.assertEqual(built["comparisons"][1]["metrics"]["accuracy"]["verdict"], "within noise")
        self.assertEqual(built["comparisons"][1]["metrics"]["no_answer"]["verdict"], "worse")
        self.assertTrue(built["comparisons"][1]["stop"])
        self.assertIn('class="noanswer"', render(registry, built))

    def test_a_stage_not_adopted_is_skipped_by_later_comparisons(self):
        everything = {case: final() for case in CASES}
        self.add(0, self.run_folder("a", everything))
        self.add(0, self.run_folder("b", everything))
        self.add(1, self.run_folder("c", {case: final(accuracy=1.0) for case in CASES}))
        with self.assertRaisesRegex(ValueError, "after stage 0"):
            reject(0, "reason", self.registry)
        with self.assertRaisesRegex(ValueError, "after stage 0"):
            reject(2, "reason", self.registry)
        with self.assertRaisesRegex(ValueError, "reason"):
            reject(1, "  ", self.registry)
        reject(1, "Accuracy fell", self.registry)
        with self.assertRaisesRegex(ValueError, "already marked"):
            reject(1, "again", self.registry)
        # Stage 2 matches stage 0, so against stage 0 it is within noise; against dropped stage 1 it would look better
        self.add(2, self.run_folder("d", everything))
        with self.assertRaisesRegex(ValueError, "later stages"):
            reject(1, "late", self.registry)
        registry = json.loads(self.registry.read_text(encoding="utf-8"))
        built = ladder(registry, self.root, labels=self.root / "absent.json")
        self.assertEqual(built["comparisons"][1]["against"], 0)
        self.assertTrue(built["comparisons"][1]["stop"])
        self.assertEqual(built["comparisons"][2]["against"], 0)
        self.assertEqual(built["comparisons"][2]["metrics"]["accuracy"]["verdict"], "within noise")
        page = render(registry, built)
        self.assertIn("Not adopted: Accuracy fell", page)
        self.assertIn("Stage 0 → stage 2", page)

    def test_scores_folder_is_named_after_its_answers(self):
        judge = self.run_folder("a", {"C1": final(), "C2": final()}, answers="stage0_run-a_answers")
        registry = self.add(0, judge)
        self.assertFalse(judge.exists())
        self.assertTrue((self.root / "stage0_run-a_scores" / "scores.jsonl").exists())
        self.assertEqual(registry["runs"][0]["judge_run"], "stage0_run-a_scores")

    def test_changed_judge_or_cases_are_refused(self):
        self.add(0, self.run_folder("a", {"C1": final(), "C2": final()}))
        with self.assertRaisesRegex(ValueError, "Judge changed .*prompt_sha256"):
            self.add(0, self.run_folder("b", {"C1": final(), "C2": final()}, prompt_sha256="edited"))
        with self.assertRaisesRegex(ValueError, "did not answer the current evaluation set"):
            self.add(0, self.run_folder("c", {"C1": final(), "C2": final()}, input_sha="other"))
        with self.assertRaisesRegex(ValueError, "Answer-writing settings changed .*model_id"):
            self.add(0, self.run_folder("d", {"C1": final(), "C2": final()}, writer="other-model"))


if __name__ == "__main__":
    unittest.main()
