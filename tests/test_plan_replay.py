"""Replaying a saved plan: the engine searches the saved sub-tasks without analyzing or planning again, without AWS."""

import contextlib
import io
import unittest

from config import Config
from core.models import QueryAnalysis, QueryIntent, RetrievalStrategy, SubTask, TaskPlan


def make_engine():
    from core.agent import AgenticRAGEngine
    with contextlib.redirect_stdout(io.StringIO()):
        return AgenticRAGEngine(config=Config(), use_mock=True, enable_judge=False)


def saved_plan(query):
    analysis = QueryAnalysis(original_query=query, intent=QueryIntent.TREATMENT_PROTOCOL,
                             disease_category="Synthetic site", is_complex=True)
    tasks = [SubTask(task_id=f"task_{i}", description=f"[Guidelines] topic {i}", query=f"saved query {i}",
                     retrieval_strategy=RetrievalStrategy.DENSE, top_k=6) for i in (1, 2)]
    return TaskPlan(plan_id="plan_saved", original_query=query, analysis=analysis, sub_tasks=tasks,
                    reasoning="Saved reasoning")


class PlanReplayTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.searched = []
        search = self.engine.retriever.execute_task
        self.engine.retriever.execute_task = lambda task, scope=None: (self.searched.append(task.query),
                                                                      search(task, scope))[1]

    def test_saved_plan_is_searched_without_analysis_or_planning(self):
        def refuse(*args, **kwargs):
            raise AssertionError("a replay analyzed or planned again")
        self.engine.query_analyzer.analyze = self.engine.query_analyzer.analyze_with_llm = refuse
        self.engine.task_planner.plan = refuse
        with contextlib.redirect_stdout(io.StringIO()):
            response = self.engine.query("Synthetic question", plan=saved_plan("Synthetic question"))

        # The saved sub-tasks are searched first and in order; a re-plan after self-critique may add more
        self.assertEqual(self.searched[:2], ["saved query 1", "saved query 2"])
        self.assertEqual(response.metadata["analysis"]["disease_category"], "Synthetic site")
        # The runner finds the first round's size in this line when the run is replayed in turn
        self.assertEqual(sum("PLAN: Created 2 sub-tasks" in line for line in response.reasoning_trace), 1)

    def test_plan_for_another_query_is_refused_before_searching(self):
        with self.assertRaises(ValueError), contextlib.redirect_stdout(io.StringIO()):
            self.engine.query("Synthetic question", plan=saved_plan("Another question"))
        self.assertEqual(self.searched, [])


if __name__ == "__main__":
    unittest.main()
