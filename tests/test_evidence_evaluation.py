"""Always-on evidence flags and optional extra assessments, with no live model calls."""
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from config import Config
from core.agent import AgenticRAGEngine, create_agent
from core.models import QueryAnalysis, QueryIntent
from evaluation import llm_judge_v3 as judge_module


class EvidenceEvaluationTests(unittest.TestCase):
    def engine(self):
        engine = AgenticRAGEngine(Config(), use_mock=True, enable_judge=True)
        # Keep all retrieval/writing synthetic, but exercise the non-mock evidence boundary.
        engine.use_mock = False
        engine.query_analyzer.analyze = lambda q: QueryAnalysis(q, QueryIntent.GENERAL, confidence=0.9)
        return engine

    def test_normal_request_keeps_evidence_check_but_skips_optional_assessments(self):
        check = {"status": "completed", "judgment": "Should Have Abstained", "flagged": True,
                 "reason": "Synthetic passages do not support the conclusion."}
        with patch.object(judge_module, 'evaluate_abstention', return_value=check) as evidence, \
                patch.object(judge_module, 'evaluate_single_query') as full, redirect_stdout(io.StringIO()):
            result = self.engine().query('Synthetic question', run_evaluation=False)
        evidence.assert_called_once()
        full.assert_not_called()
        self.assertEqual(result.metadata['evidence_check'], check)
        self.assertEqual(result.metadata['quality_evaluation']['status'], 'skipped')
        self.assertTrue(result.answer)

    def test_failed_check_is_unknown_instead_of_silently_clear(self):
        with patch.object(judge_module, 'evaluate_abstention', side_effect=ValueError('Synthetic failure')), redirect_stdout(io.StringIO()):
            result = self.engine().query('Synthetic question', run_evaluation=False)
        self.assertEqual(result.metadata['evidence_check']['status'], 'unavailable')
        self.assertIsNone(result.metadata['evidence_check']['flagged'])

    def test_one_check_does_not_require_composite_calibration(self):
        judge = Mock()
        judge.judge_abstention.return_value = judge_module.AbstentionResult('Partial Abstention Preferred', 'Synthetic reason')
        with patch.object(judge_module, '_get_judge', return_value=judge), \
                patch.object(judge_module, '_get_calibration', side_effect=AssertionError('Must not load scores')):
            result = judge_module.evaluate_abstention('q', 'a', ['Synthetic source'])
        self.assertTrue(result['flagged'])
        judge.judge_abstention.assert_called_once()

    def test_invalid_model_judgment_is_rejected(self):
        judge = Mock()
        judge.judge_abstention.return_value = judge_module.AbstentionResult()
        with patch.object(judge_module, '_get_judge', return_value=judge), self.assertRaises(ValueError):
            judge_module.evaluate_abstention('q', 'a', ['Synthetic source'])

    def test_full_review_reuses_the_required_check_and_runs_five_more_calls(self):
        judge = judge_module.LLMJudgeV3.__new__(judge_module.LLMJudgeV3)
        judge.judge_abstention = Mock(return_value=judge_module.AbstentionResult('Should Have Abstained', 'Synthetic reason'))
        judge.judge_retrieval_adequacy = Mock(return_value=judge_module.RetrievalAdequacyResult(3, ''))
        judge.check_did_abstain = Mock(return_value='No')
        judge.judge_faithfulness = Mock(return_value=judge_module.FaithfulnessResult(3))
        judge.judge_correctness_rag_proxy = Mock(return_value=judge_module.CorrectnessResult(3))
        judge.judge_answer_quality = Mock(return_value=judge_module.AnswerQualityResult(3))
        with patch.object(judge_module, '_get_judge', return_value=judge):
            check = judge_module.evaluate_abstention('q', 'a', ['Synthetic source'])
            result = judge_module.evaluate_single_query('q', 'a', ['Synthetic source'], abstention_check=check)
        methods = [judge.judge_abstention, judge.judge_retrieval_adequacy, judge.check_did_abstain,
                   judge.judge_faithfulness, judge.judge_correctness_rag_proxy, judge.judge_answer_quality]
        self.assertEqual([method.call_count for method in methods], [1] * 6)
        self.assertTrue(result['abstention_triggered'])
        self.assertEqual(result['abstention_justification'], 'Synthetic reason')

    def test_interactive_factory_defaults_to_one_check_and_review_engine_to_full(self):
        normal = create_agent(use_mock=True)
        self.assertFalse(normal.enable_judge)
        review = AgenticRAGEngine(Config(), use_mock=True)
        self.assertTrue(review.enable_judge)


if __name__ == '__main__':
    unittest.main()
