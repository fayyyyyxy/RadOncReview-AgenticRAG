"""After the critique: the rewrite and its re-plan, first-draft logging, the citation check's removals, and what the
critique counts as addressed, without AWS."""

import contextlib
import io
import unittest

from config import Config
from core.models import QueryAnalysis, QueryIntent, RetrievalStrategy, SelfCritiqueResult, SubTask, TaskPlan
from core.synthesizer import SelfCritic

QUESTION = "Synthetic question"


def make_engine():
    from core.agent import AgenticRAGEngine
    with contextlib.redirect_stdout(io.StringIO()):
        return AgenticRAGEngine(config=Config(), use_mock=True, enable_judge=False)


def saved_plan():
    analysis = QueryAnalysis(original_query=QUESTION, intent=QueryIntent.TREATMENT_PROTOCOL, is_complex=True)
    tasks = [SubTask(task_id=f"task_{i}", description=f"[Guidelines] topic {i}", query=f"saved query {i}",
                     retrieval_strategy=RetrievalStrategy.DENSE, top_k=6) for i in (1, 2)]
    return TaskPlan(plan_id="plan_saved", original_query=QUESTION, analysis=analysis, sub_tasks=tasks,
                    reasoning="Saved reasoning")


class RetryTests(unittest.TestCase):
    """A critique that calls the draft unsafe forces one rewrite."""

    def run_query(self):
        engine = make_engine()
        self.replanned, self.judged, self.writes = [], [], []
        replan = engine.task_planner.replan
        engine.task_planner.replan = lambda plan, guidance: (self.replanned.append(guidance), replan(plan, guidance))[1]

        def critique(query, answer, analysis):
            self.judged.append(answer)
            return SelfCritiqueResult(is_complete=True, is_safe=False, concerns=["Synthetic concern"],
                                      confidence_score=0.8)
        engine.self_critic.critique = critique
        write = engine.synthesizer.synthesize_multi_topic

        def recording(query, analysis, chunks_by_topic, **kwargs):
            self.writes.append(kwargs)
            return write(query, analysis, chunks_by_topic, **kwargs)
        engine.synthesizer.synthesize_multi_topic = recording
        with contextlib.redirect_stdout(io.StringIO()):
            return engine.query(QUESTION, plan=saved_plan())

    def test_a_rewrite_re_plans_and_gets_the_feedback_and_the_draft(self):
        self.run_query()
        self.assertEqual(len(self.replanned), 1)
        self.assertEqual(len(self.writes), 2)
        self.assertEqual(self.writes[1]["draft"], self.judged[0])
        self.assertIn("Synthetic concern", self.writes[1]["feedback"])

    def test_the_draft_the_critique_judged_is_kept(self):
        response = self.run_query()
        self.assertEqual(response.metadata["first_draft"], self.judged[0])


class CitationCheckTests(unittest.TestCase):
    def test_rejected_citations_are_removed_and_reported(self):
        engine = make_engine()
        engine.self_critic.critique = lambda query, answer, analysis: SelfCritiqueResult(
            is_complete=True, is_safe=True, confidence_score=0.9)
        engine.citation_checker.check = lambda answer, passages: ("CHECKED ANSWER", {
            "citations": 1, "items": 1, "unchecked": 0, "removed": 1, "attempts": 1, "error": None,
            "uncited_sentences": ["Synthetic sentence."], "rejected": [{"sentence": "Synthetic sentence.", "citation": 1}]})
        with contextlib.redirect_stdout(io.StringIO()):
            response = engine.query(QUESTION, plan=saved_plan())
        check = response.metadata["citation_check"]

        self.assertIn("CHECKED ANSWER", response.answer)
        self.assertEqual(check["rejected"], [{"sentence": "Synthetic sentence.", "citation": 1}])
        self.assertNotIn("CHECKED ANSWER", check["answer_before"])


class RecordingLLM:
    def __init__(self):
        self.prompt = None

    def generate_structured(self, prompt, output_schema, system_prompt=None):
        self.prompt = prompt
        return {"checklist": [], "is_safe": True, "concerns": [], "confidence_score": 0.9, "suggestions": []}


class AddressedRuleTests(unittest.TestCase):
    def test_a_named_gap_does_not_count_as_addressed(self):
        # Counting it marked every item addressed, and completeness fell (docs/rag_improvement_plan_v2.md, Part 1)
        llm = RecordingLLM()
        analysis = QueryAnalysis(original_query=QUESTION, intent=QueryIntent.TREATMENT_PROTOCOL)
        SelfCritic(llm, Config().agent).critique(QUESTION, "Synthetic answer.", analysis)
        self.assertIn("Saying that the sources do not cover it, giving general background, or answering for another "
                      "condition or setting does not count.", " ".join(llm.prompt.split()))


if __name__ == "__main__":
    unittest.main()
