"""Flexible source-anchored planning, cache invalidation and actual retry integration; no AWS."""
import copy
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from config import Config
from core.models import QueryAnalysis, QueryIntent, TaskStatus, SelfCritiqueResult
from core.task_planner import TaskPlanner
from core.versioned_planner import validate_payload

QUESTION = "How should pacemaker monitoring, dental assessment, fertility preservation, timing, and systemic therapy be coordinated?"
QUERY = "Patient vignette:\nSynthetic RIGHT-sided condition, prior 42 Gy.\n\nClinical question:\n" + QUESTION
QUOTES = ["pacemaker monitoring", "dental assessment", "fertility preservation", "timing", "systemic therapy"]


def payload(quotes=QUOTES, source="question"):
    return {"requirements": [{"id": f"R{i}", "source": source, "quote": q} for i, q in enumerate(quotes, 1)],
            "topics": [{"requirement_ids": [f"R{i}"]} for i in range(1, len(quotes) + 1)],
            "reasoning": "Each requested need gets a search; arbitrary topics are allowed."}


class VersionedPlannerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Config()
        self.config.features.versioned_planning = True
        self.config.features.plan_cache_path = self.temp.name
        self.llm = Mock()
        self.llm.generate_structured.return_value = payload()
        self.analysis = QueryAnalysis(QUERY, QueryIntent.GENERAL)
        self.planner = TaskPlanner(self.llm, self.config)
        self.capture = contextlib.redirect_stdout(io.StringIO())
        self.capture.__enter__()
        self.addCleanup(self.capture.__exit__, None, None, None)

    def test_unfamiliar_needs_and_more_than_three_topics_are_preserved(self):
        plan = self.planner.plan(self.analysis)
        self.assertEqual(len(plan.sub_tasks), 10)
        self.assertTrue(plan.analysis.is_complex)
        self.assertEqual([t.description for t in plan.sub_tasks[::2]], [f"[Guidelines] {q}" for q in QUOTES])
        self.assertTrue(all(QUERY in t.query for t in plan.sub_tasks))
        self.assertEqual(plan.planning["cache_status"], "created")

    def test_cache_hit_has_fresh_pending_tasks_and_no_passage_payload(self):
        first = self.planner.plan(self.analysis)
        first.sub_tasks[0].status = TaskStatus.COMPLETED
        first.sub_tasks[0].retrieved_chunks = [{"text": "Do not cache this passage"}]
        second = TaskPlanner(self.llm, self.config).plan(self.analysis)
        self.llm.generate_structured.assert_called_once()
        self.assertEqual(second.planning["cache_status"], "hit")
        self.assertNotEqual(first.plan_id, second.plan_id)
        self.assertTrue(all(t.status == TaskStatus.PENDING and not t.retrieved_chunks for t in second.sub_tasks))
        self.assertNotIn("Do not cache this passage", next(Path(self.temp.name).glob("*.json")).read_text())

    def test_changed_patient_model_prompt_or_retrieval_settings_get_new_keys(self):
        first = self.planner.plan(self.analysis).planning["cache_key"]
        changed = self.planner.plan(QueryAnalysis(QUERY.replace("RIGHT", "LEFT"), QueryIntent.GENERAL))
        self.assertNotEqual(first, changed.planning["cache_key"])
        self.config.llm.model_id = "synthetic-other-model"
        model = self.planner.plan(self.analysis).planning["cache_key"]
        self.assertNotEqual(first, model)
        with patch("core.versioned_planner.PROMPT", "Changed prompt"):
            prompt = self.planner.plan(self.analysis).planning["cache_key"]
        self.assertNotEqual(model, prompt)
        self.config.retriever.kb_complex_top_k += 1
        retrieval = self.planner.plan(self.analysis).planning["cache_key"]
        self.assertNotEqual(model, retrieval)

    def test_invented_source_quote_repairs_once_then_uses_explicit_uncached_fallback(self):
        self.llm.generate_structured.return_value = payload(["Invented diagnosis or recommendation"])
        plan = self.planner.plan(self.analysis)
        self.assertEqual(self.llm.generate_structured.call_count, 2)
        self.assertEqual(plan.planning["cache_status"], "fallback_not_cached")
        self.assertTrue(all(t.query == QUERY for t in plan.sub_tasks))
        self.assertFalse(list(Path(self.temp.name).glob("*.json")))
        self.assertEqual(len(plan.planning["failed_attempts"]), 2)

    def test_every_declared_requirement_needs_a_topic_and_budgets_do_not_truncate(self):
        missing = payload()
        missing["topics"].pop()
        with self.assertRaisesRegex(ValueError, "Every declared requirement"):
            validate_payload(missing, {"question": QUESTION}, 6)
        with self.assertRaisesRegex(ValueError, "Budget exceeded"):
            validate_payload(payload(), {"question": QUESTION}, 3)

    def test_model_cannot_remerge_separate_needs_when_budget_allows_individual_searches(self):
        merged = payload()
        merged["topics"] = [{"requirement_ids": [f"R{i}" for i in range(1, 6)]}]
        result = validate_payload(merged, {"question": QUESTION}, 6)
        self.assertEqual(len(result["topics"]), 5)
        grouped = validate_payload(merged, {"question": QUESTION}, 3)
        self.assertEqual(len(grouped["topics"]), 1)
        self.assertEqual(len(grouped["topics"][0]["requirement_ids"]), 5)

    def test_obviously_bundled_question_quote_requires_a_more_granular_extraction(self):
        with self.assertRaisesRegex(ValueError, "bundles a list"):
            validate_payload(payload([QUESTION]), {"question": QUESTION}, 6)

    def test_corrupt_cache_is_rejected_not_silently_used(self):
        self.planner.plan(self.analysis)
        p = next(Path(self.temp.name).glob("*.json"))
        record = json.loads(p.read_text())
        record["payload"]["reasoning"] = "tampered"
        p.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "corrupted"):
            self.planner.plan(self.analysis)

    def test_concurrent_misses_publish_one_plan(self):
        with ThreadPoolExecutor(max_workers=2) as pool:
            plans = list(pool.map(lambda _: self.planner.plan(self.analysis), range(2)))
        self.assertEqual(self.llm.generate_structured.call_count, 1)
        self.assertEqual({p.planning["cache_status"] for p in plans}, {"created", "hit"})
        self.assertEqual(plans[0].planning["payload_sha256"], plans[1].planning["payload_sha256"])

    def test_followup_uses_new_gap_keeps_patient_input_and_is_versioned_by_feedback(self):
        original = self.planner.plan(self.analysis)
        feedback = "Missing evidence about device interrogation after treatment."
        self.llm.generate_structured.return_value = payload(["device interrogation after treatment"], "feedback")
        followup = self.planner.replan(original, feedback)
        self.assertEqual(len(followup.sub_tasks), 2)
        self.assertTrue(all("device interrogation after treatment" in t.query and QUERY in t.query for t in followup.sub_tasks))
        self.assertEqual(followup.planning["parent_cache_key"], original.planning["cache_key"])
        reused = self.planner.replan(original, feedback)
        self.assertEqual(reused.planning["cache_status"], "hit")
        changed = self.planner.replan(original, feedback + " Please investigate.")
        self.assertNotEqual(followup.planning["cache_key"], changed.planning["cache_key"])
        self.assertEqual(self.llm.generate_structured.call_count, 3)

    def test_rewrite_only_and_invalid_followups_do_not_repeat_initial_searches(self):
        original = self.planner.plan(self.analysis)
        self.llm.generate_structured.return_value = {"requirements": [], "topics": [], "reasoning": "Rewrite only"}
        followup = self.planner.replan(original, "Improve clarity")
        self.assertEqual(followup.sub_tasks, [])
        self.llm.generate_structured.return_value = payload(["not in feedback"], "feedback")
        bad = self.planner.replan(original, "Another request")
        self.assertEqual(bad.sub_tasks, [])
        self.assertEqual(bad.planning["cache_status"], "fallback_not_cached")

    def test_repeated_existing_focus_is_skipped(self):
        original = self.planner.plan(self.analysis)
        self.llm.generate_structured.return_value = payload([QUOTES[0]])
        followup = self.planner.replan(original, "Review pacemaker monitoring")
        self.assertEqual(followup.sub_tasks, [])
        self.assertEqual(followup.planning["skipped_existing_topics"], [QUOTES[0]])

    def test_default_planner_keeps_legacy_behavior_without_plan_caching(self):
        self.config.features.versioned_planning = False
        self.llm.generate_structured.return_value = {
            "reasoning": "Legacy plan", "sub_tasks": [{"description": "Original search", "query": "Synthetic legacy query"}]}
        analysis = QueryAnalysis(QUERY, QueryIntent.GENERAL, is_complex=True)
        plan = TaskPlanner(self.llm, self.config).plan(analysis)
        self.assertEqual(len(plan.sub_tasks), 2)
        self.assertTrue(all(t.query == "Synthetic legacy query" for t in plan.sub_tasks))
        self.assertEqual(plan.planning, {})
        self.assertFalse(list(Path(self.temp.name).glob("*.json")))

    def test_engine_retry_reaches_new_search_and_writer_and_records_both_events(self):
        from core.agent import AgenticRAGEngine
        config = self.config
        config.features.citation_check = False
        engine = AgenticRAGEngine(config, use_mock=True, enable_judge=False)
        engine.query_analyzer.analyze = Mock(return_value=self.analysis)
        engine.query_analyzer.analyze_with_llm = Mock(side_effect=AssertionError("No uncached analysis call"))
        feedback = "Missing device interrogation after treatment"
        engine.task_planner = self.planner
        self.llm.generate_structured.side_effect = [payload(), payload(["device interrogation after treatment"], "feedback")]
        engine.self_critic.critique = Mock(return_value=SelfCritiqueResult(is_complete=False, is_safe=True))
        engine.self_critic.get_improvement_guidance = Mock(return_value=feedback)
        seen_queries = []
        def search(task, scope):
            seen_queries.append(task.query)
            task.status = TaskStatus.COMPLETED
            task.retrieved_chunks = [{"text": "Source " + task.query.split('\n')[0], "path": "Synthetic source",
                                     "faiss_id": len(seen_queries), "similarity_score": 0.8}]
        engine.retriever.execute_task = search
        response = engine.query(QUERY)
        self.assertEqual(len(seen_queries), 12)
        self.assertTrue(all("device interrogation after treatment" in q for q in seen_queries[-2:]))
        self.assertEqual(len(response.synthesis_requests), 2)
        self.assertIn(feedback, response.synthesis_requests[-1]["prompt"])
        self.assertEqual([e["phase"] for e in response.routing["planning_events"]], ["initial", "followup"])
        self.assertEqual(len(response.task_plan.sub_tasks), 12)
        self.assertTrue(any("pacemaker monitoring" in p["text"] for p in response.context_passages))


if __name__ == "__main__":
    unittest.main()
