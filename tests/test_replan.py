"""The re-plan after self-critique, with a stand-in model instead of AWS."""

import unittest

from config import Config
from core.models import QueryAnalysis, QueryIntent, RetrievalStrategy, SubTask, TaskPlan
from core.task_planner import TaskPlanner


class ScriptedLLM:
    """Answers every structured request with the same re-plan."""

    def __init__(self, result):
        self.result = result

    def generate_structured(self, prompt, output_schema, system_prompt=None):
        return self.result


def first_round():
    analysis = QueryAnalysis(original_query="Synthetic question", intent=QueryIntent.TREATMENT_PROTOCOL,
                             is_complex=True)
    task = SubTask(task_id="task_1", description="[Guidelines] topic", query="first query",
                   retrieval_strategy=RetrievalStrategy.DENSE, top_k=6)
    return TaskPlan(plan_id="plan_1", original_query="Synthetic question", analysis=analysis, sub_tasks=[task],
                    reasoning="First round")


class ReplanTests(unittest.TestCase):
    def test_blank_queries_are_dropped(self):
        # As in the saved OLI-002 re-plan: a subtopic with no query and no description
        planner = TaskPlanner(ScriptedLLM({"reasoning": "r", "sub_tasks": [
            {"description": "spine constraints", "query": "spinal cord constraints 2 fractions"},
            {"description": "", "query": ""},
            {"description": "no query key"},
            {"description": "whitespace", "query": "  "},
            {"description": "null", "query": None},
        ]}), Config())
        plan = planner.replan(first_round(), "Address these missing aspects: dose")

        # One subtopic is left, searched both ways
        self.assertEqual([t.query for t in plan.sub_tasks], ["spinal cord constraints 2 fractions"] * 2)
        self.assertEqual([t.retrieval_strategy for t in plan.sub_tasks],
                         [RetrievalStrategy.DENSE, RetrievalStrategy.KNOWLEDGE_BASE])

    def test_all_blank_keeps_the_plan(self):
        plan = first_round()
        planner = TaskPlanner(ScriptedLLM({"reasoning": "r", "sub_tasks": [{"query": ""}]}), Config())
        self.assertIs(planner.replan(plan, "feedback"), plan)


if __name__ == "__main__":
    unittest.main()
