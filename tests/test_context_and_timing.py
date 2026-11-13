"""Web context guardrails and checked-answer timing without live model calls."""
from contextlib import redirect_stdout
import io
import unittest
from unittest.mock import patch

from config import Config
from core.agent import AgenticRAGEngine
from core.case_labels import CaseLabels
from core.models import QueryAnalysis, QueryIntent
from core.site_routing import scope_for

EVALUATION = {"retrieval_adequacy": 4, "faithfulness": 4, "correctness": 4,
              "answer_quality": 4, "composite_score": 4.0, "abstention_triggered": False}


class ContextAndTimingTests(unittest.TestCase):
    def engine(self, judge=False):
        engine = AgenticRAGEngine(Config(), use_mock=True, enable_judge=judge)
        engine.query_analyzer.analyze = lambda q: QueryAnalysis(q, QueryIntent.GENERAL, confidence=0.9)
        return engine

    def test_bounded_user_context_reaches_retrieval_writer_and_evaluator_without_server_history(self):
        engine = self.engine(judge=True)
        stages = []
        scope = scope_for("Breast")
        scopes = []
        search = engine.retriever.execute_task
        engine.retriever.execute_task = lambda task, s: (scopes.append(s), search(task, s))[1]
        with patch("evaluation.llm_judge_v3.evaluate_single_query", return_value=EVALUATION) as evaluate, redirect_stdout(io.StringIO()):
            result = engine.query("What about toxicity?", CaseLabels(disease_site="Breast"),
                                  prior_questions=["Synthetic initial case", "Synthetic recent question"],
                                  evidence_scope=scope, progress_callback=stages.append, retain_history=False)
        effective_question = result.task_plan.original_query
        self.assertIn("Synthetic initial case", effective_question)
        self.assertIn("Current clinical question:\nWhat about toxicity?", effective_question)
        self.assertEqual(evaluate.call_args.kwargs["question"], effective_question)
        self.assertTrue(all(s == scope for s in scopes))
        self.assertEqual(result.query, "What about toxicity?")
        self.assertEqual(engine.conversation_history, [])
        self.assertEqual(result.metadata["context"]["earlier_questions"], 2)
        self.assertEqual(result.metadata["judge_evaluations"], [EVALUATION])
        self.assertLess(stages.index("synthesize"), stages.index("evaluate"))
        self.assertEqual(stages[-1], "complete")

    def test_unsafe_prior_input_is_rejected_before_any_inference(self):
        engine = self.engine()
        with redirect_stdout(io.StringIO()):
            result = engine.query("What about toxicity?", prior_questions=["MRN: ABC12345"])
        self.assertTrue(result.metadata["input_blocked"])
        self.assertEqual(engine.llm.call_count, 0)

    def test_edited_context_is_used_and_safety_checked_as_a_whole(self):
        engine = self.engine()
        with redirect_stdout(io.StringIO()):
            result = engine.query('What about toxicity?', conversation_context='Corrected stage: II', retain_history=False)
        self.assertIn('Corrected stage: II', result.task_plan.original_query)
        self.assertEqual(result.metadata['context']['case_context_chars'], 19)
        with redirect_stdout(io.StringIO()):
            blocked = engine.query('q', conversation_context='x' * 1995 + ' MRN: ABC12345')
        self.assertTrue(blocked.metadata['input_blocked'])
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, '4,000'):
            engine.query('q', conversation_context='x' * 4001)

    def test_more_than_two_prior_questions_is_rejected(self):
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "At most two"):
            self.engine().query("q", prior_questions=["a", "b", "c"])

    def test_final_latency_includes_evaluator_time(self):
        engine = self.engine(judge=True)
        clock = [100.0]

        def evaluate(**kwargs):
            clock[0] += 12
            return EVALUATION

        with patch("core.agent.time.time", side_effect=lambda: clock[0]), \
                patch("evaluation.llm_judge_v3.evaluate_single_query", side_effect=evaluate), redirect_stdout(io.StringIO()):
            result = engine.query("Synthetic question", retain_history=False)
        self.assertEqual(result.metadata["answer_ready_ms"], 0)
        self.assertEqual(result.latency_ms, 12000)


if __name__ == "__main__":
    unittest.main()
