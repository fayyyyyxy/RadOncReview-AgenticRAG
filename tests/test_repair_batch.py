"""The Part 1 repair batch switches (re-plan off, named gap, citation flag-only) and first-draft logging, without AWS."""

import contextlib
import io
import unittest

from config import Config
from core.models import QueryAnalysis, QueryIntent, RetrievalStrategy, SelfCritiqueResult, SubTask, TaskPlan
from core.synthesizer import SelfCritic

QUESTION = "Synthetic question"


def make_engine(**features):
    from core.agent import AgenticRAGEngine
    config = Config()
    for name, value in features.items():
        setattr(config.features, name, value)
    with contextlib.redirect_stdout(io.StringIO()):
        return AgenticRAGEngine(config=config, use_mock=True, enable_judge=False)


def saved_plan():
    analysis = QueryAnalysis(original_query=QUESTION, intent=QueryIntent.TREATMENT_PROTOCOL, is_complex=True)
    tasks = [SubTask(task_id=f"task_{i}", description=f"[Guidelines] topic {i}", query=f"saved query {i}",
                     retrieval_strategy=RetrievalStrategy.DENSE, top_k=6) for i in (1, 2)]
    return TaskPlan(plan_id="plan_saved", original_query=QUESTION, analysis=analysis, sub_tasks=tasks,
                    reasoning="Saved reasoning")


class RetryTests(unittest.TestCase):
    """A critique that calls the draft unsafe forces one rewrite."""

    def run_query(self, **features):
        engine = make_engine(**features)
        self.searched, self.replanned, self.judged, self.writes = [], [], [], []
        search = engine.retriever.execute_task
        engine.retriever.execute_task = lambda task, scope=None: (self.searched.append(task.query),
                                                                 search(task, scope))[1]
        replan = engine.task_planner.replan
        engine.task_planner.replan = lambda plan, guidance: (self.replanned.append(guidance), replan(plan, guidance))[1]

        def critique(query, answer, analysis):
            self.judged.append(answer)
            return SelfCritiqueResult(is_complete=True, is_safe=False, concerns=["Synthetic concern"],
                                      confidence_score=0.8)
        engine.self_critic.critique = critique
        write = engine.synthesizer.synthesize_multi_topic

        def recording(query, analysis, chunks_by_topic, **kwargs):
            self.writes.append({"passages": sum(len(c) for c in chunks_by_topic.values()), **kwargs})
            return write(query, analysis, chunks_by_topic, **kwargs)
        engine.synthesizer.synthesize_multi_topic = recording
        with contextlib.redirect_stdout(io.StringIO()):
            return engine.query(QUESTION, plan=saved_plan())

    def test_replan_runs_by_default(self):
        self.run_query()
        self.assertEqual(len(self.replanned), 1)

    def test_replan_off_rewrites_from_the_first_round_passages(self):
        response = self.run_query(replan=False)

        self.assertEqual(self.replanned, [])
        self.assertEqual(self.searched, ["saved query 1", "saved query 2"])
        self.assertEqual(len(self.writes), 2)
        # Same passages, now with the critique's feedback and the draft it judged
        self.assertEqual(self.writes[1]["passages"], self.writes[0]["passages"])
        self.assertEqual(self.writes[1]["draft"], self.judged[0])
        self.assertIn("Synthetic concern", self.writes[1]["feedback"])
        self.assertEqual([t.query for t in response.task_plan.sub_tasks], ["saved query 1", "saved query 2"])

    def test_the_draft_the_critique_judged_is_kept(self):
        response = self.run_query(replan=False)
        self.assertEqual(response.metadata["first_draft"], self.judged[0])


class FlagOnlyTests(unittest.TestCase):
    """The citation check reports rejected citations; flag-only leaves them in the answer."""

    def run_query(self, **features):
        engine = make_engine(**features)
        engine.self_critic.critique = lambda query, answer, analysis: SelfCritiqueResult(
            is_complete=True, is_safe=True, confidence_score=0.9)
        engine.citation_checker.check = lambda answer, passages: ("CHECKED ANSWER", {
            "citations": 1, "items": 1, "unchecked": 0, "removed": 1, "attempts": 1, "error": None,
            "uncited_sentences": ["Synthetic sentence."], "rejected": [{"sentence": "Synthetic sentence.", "citation": 1}]})
        with contextlib.redirect_stdout(io.StringIO()):
            return engine.query(QUESTION, plan=saved_plan())

    def test_rejected_citations_are_removed_by_default(self):
        response = self.run_query()
        self.assertIn("CHECKED ANSWER", response.answer)
        self.assertFalse(response.metadata["citation_check"]["flag_only"])

    def test_flag_only_keeps_the_answer_and_lists_the_rejected_citations(self):
        response = self.run_query(citation_flag_only=True)
        check = response.metadata["citation_check"]

        self.assertNotIn("CHECKED ANSWER", response.answer)
        self.assertIn(check["answer_before"], response.answer)
        self.assertTrue(check["flag_only"])
        self.assertEqual(check["rejected"], [{"sentence": "Synthetic sentence.", "citation": 1}])


class RecordingLLM:
    def __init__(self):
        self.prompt = None

    def generate_structured(self, prompt, output_schema, system_prompt=None):
        self.prompt = prompt
        return {"checklist": [], "is_safe": True, "concerns": [], "confidence_score": 0.9, "suggestions": []}


class NamedGapTests(unittest.TestCase):
    def critique_prompt(self, **kwargs):
        llm = RecordingLLM()
        analysis = QueryAnalysis(original_query=QUESTION, intent=QueryIntent.TREATMENT_PROTOCOL)
        SelfCritic(llm, Config().agent, **kwargs).critique(QUESTION, "Synthetic answer.", analysis)
        return " ".join(llm.prompt.split())

    def test_a_named_gap_does_not_count_by_default(self):
        prompt = self.critique_prompt()
        self.assertIn("Saying that the sources do not cover it, giving general background, or answering for another "
                      "condition or setting does not count.", prompt)

    def test_with_the_switch_a_named_gap_counts_as_addressed(self):
        prompt = self.critique_prompt(named_gap=True)
        self.assertIn("Mark an item addressed if the answer gives a specific recommendation for this patient, or if it "
                      "says the provided evidence does not cover the item. General background, or an answer for "
                      "another condition or setting, does not count.", prompt)
        self.assertNotIn("Saying that the sources do not cover it", prompt)


if __name__ == "__main__":
    unittest.main()
