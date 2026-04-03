"""
Task Planner: Decomposes complex queries into executable sub-tasks.
Creates an execution plan for the agent to follow.
"""
import uuid
from typing import List, Optional
from datetime import datetime

from models import (
    QueryAnalysis, TaskPlan, SubTask, TaskStatus,
    RetrievalStrategy, QueryIntent
)
from llm_client import BaseLLMClient
from config import Config


class TaskPlanner:
    """Plans and decomposes queries into executable sub-tasks."""

    # Intent to retrieval strategy mapping
    STRATEGY_MAP = {
        QueryIntent.DOSE_FRACTIONATION: RetrievalStrategy.DENSE,
        QueryIntent.TREATMENT_PROTOCOL: RetrievalStrategy.DENSE,
        QueryIntent.SIDE_EFFECTS: RetrievalStrategy.DENSE,
        QueryIntent.STAGING: RetrievalStrategy.DENSE,  # Keyword-heavy
        QueryIntent.INDICATIONS: RetrievalStrategy.DENSE,
        QueryIntent.CONTRAINDICATIONS: RetrievalStrategy.DENSE,
        QueryIntent.COMPARISON: RetrievalStrategy.DENSE,
        QueryIntent.DEFINITION: RetrievalStrategy.DENSE,
        QueryIntent.GENERAL: RetrievalStrategy.DENSE,
    }

    def __init__(self, llm_client: BaseLLMClient, config: Config):
        self.llm = llm_client
        self.config = config
    
    def _create_task_id(self) -> str:
        """Generate a unique task ID."""
        return f"task_{uuid.uuid4().hex[:8]}"
    
    def _create_plan_id(self) -> str:
        """Generate a unique plan ID."""
        return f"plan_{uuid.uuid4().hex[:8]}"

    def _create_simple_plan(self, analysis: QueryAnalysis) -> TaskPlan:
        """
        Simple queries: always TWO subtasks — DENSE (FAISS) + KNOWLEDGE_BASE (KB).
        Same query sent to both sources. agent._collect_all_chunks() merges results,
        cross-encoder reranks everything together down to rerank_top_k chunks.
        """
        search_query = analysis.original_query
        if analysis.disease_category:
            search_query = f"{analysis.disease_category}: {search_query}"

        sub_tasks = [
            # Task 1: FAISS — RadOncReview physician-written guidelines
            SubTask(
                task_id=self._create_task_id(),
                description=f"Search RadOncReview guidelines: {analysis.original_query[:80]}",
                query=search_query,
                retrieval_strategy=RetrievalStrategy.DENSE,
                status=TaskStatus.PENDING
            ),
            # Task 2: KB — reading list clinical trial papers
            SubTask(
                task_id=self._create_task_id(),
                description=f"Search clinical trial papers: {analysis.original_query[:80]}",
                query=search_query,
                retrieval_strategy=RetrievalStrategy.KNOWLEDGE_BASE,
                status=TaskStatus.PENDING
            ),
        ]

        return TaskPlan(
            plan_id=self._create_plan_id(),
            original_query=analysis.original_query,
            analysis=analysis,
            sub_tasks=sub_tasks,
            reasoning=(
                "Dual-source retrieval: FAISS for RadOncReview guidelines + "
                "KB for clinical trial papers. Cross-encoder reranks all chunks together."
            )
        )

    def plan(self, analysis: QueryAnalysis) -> TaskPlan:
        """Create an execution plan based on query analysis."""
        # Complex/comparison queries use LLM to decompose into subtopics
        if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
            return self._plan_with_llm(analysis)

        # Simple queries: dual-source with same query
        return self._create_simple_plan(analysis)

    def _plan_with_llm(self, analysis: QueryAnalysis) -> TaskPlan:
        """
        Use LLM to decompose complex/comparison queries into subtopics.
        Each LLM subtopic becomes a DENSE + KNOWLEDGE_BASE task pair.

        Example — "Compare SBRT vs EBRT for prostate cancer":
          LLM produces 2 subtopics: "SBRT prostate", "EBRT prostate"
          Result: 4 tasks — DENSE(SBRT), KB(SBRT), DENSE(EBRT), KB(EBRT)
          agent._collect_multi_topic_chunks() groups by topic for synthesis
        """
        system_prompt = """You are a radiation oncology query planner.
Your job is to break down complex clinical questions into simpler sub-tasks
that can be answered by searching a knowledge base of treatment guidelines."""

        ### IMPORTANT FOR FUTURE CHANGES: dense strategy is used exclusively.
        ### Both FAISS and KB are searched for every subtopic automatically.
        planning_prompt = f"""Break down this radiation oncology query into sub-tasks for retrieval:

Original Query: "{analysis.original_query}"

Query Analysis:
- Intent: {analysis.intent.value}
- Disease Category: {analysis.disease_category or "Not specified"}
- Treatment Modality: {analysis.treatment_modality or "Not specified"}
- Key Entities: {", ".join(analysis.key_entities) if analysis.key_entities else "None"}

TASK GENERATION GUIDELINES:
1. Generate the MOST APPROPRIATE number of sub-tasks based on the query (typically 2-3)
2. Each sub-task should retrieve DISTINCT information needed to answer the query
3. Maximum 3 sub-tasks - ensure these 3 tasks comprehensively cover all information needed
4. Each sub-task should be focused and specific
5. Sub-tasks should be independent (can be executed in parallel)

For comparison queries (e.g., "Compare A vs B"):
- Create 2 tasks: one focused on A, one focused on B
- Each task should retrieve comprehensive information about that option

For complex queries with multiple aspects:
- Identify 2-3 key information needs
- Create one task per key information need
- Ensure tasks together cover the full query scope

RETRIEVAL STRATEGY:
- Always use "dense" strategy (both guidelines and trial papers are searched automatically)

Respond in JSON format:
{{
    "reasoning": "<explanation of your decomposition strategy and why this number of tasks>",
    "sub_tasks": [
        {{
            "description": "<what this task will find>",
            "query": "<search query for knowledge base>",
            "strategy": "dense"
        }}
    ]
}}

Remember: Generate the optimal number of tasks (2-3) that comprehensively cover information retrieval needs."""

        try:
            result = self.llm.generate_structured(
                planning_prompt,
                output_schema={"reasoning": "string", "sub_tasks": "array"},
                system_prompt=system_prompt
            )

            if "error" not in result and "sub_tasks" in result:
                sub_tasks = []
                tasks_to_process = result["sub_tasks"][:3]  # cap at 3 subtopics
                num_tasks = len(tasks_to_process)

                task_top_k = self.config.retriever.complex_top_k if num_tasks in [2, 3] else None

                for i, task in enumerate(tasks_to_process):
                    subtopic_query = task.get("query", analysis.original_query)

                    # ── Each LLM subtopic → DENSE task + KNOWLEDGE_BASE task ──
                    sub_tasks.append(SubTask(
                        task_id=self._create_task_id(),
                        description=f"[Guidelines] {task.get('description', f'Sub-task {i+1}')}",
                        query=subtopic_query,
                        retrieval_strategy=RetrievalStrategy.DENSE,
                        top_k=task_top_k,
                        status=TaskStatus.PENDING
                    ))
                    sub_tasks.append(SubTask(
                        task_id=self._create_task_id(),
                        description=f"[Trial papers] {task.get('description', f'Sub-task {i+1}')}",
                        query=subtopic_query,
                        retrieval_strategy=RetrievalStrategy.KNOWLEDGE_BASE,
                        top_k=task_top_k,
                        status=TaskStatus.PENDING
                    ))

                if sub_tasks:
                    return TaskPlan(
                        plan_id=self._create_plan_id(),
                        original_query=analysis.original_query,
                        analysis=analysis,
                        sub_tasks=sub_tasks,
                        reasoning=result.get("reasoning", "LLM-generated plan")
                    )

        except Exception as e:
            print(f"LLM planning failed, using simple plan: {e}")

        return self._create_simple_plan(analysis)

    def replan(self, plan: TaskPlan, feedback: str) -> TaskPlan:
        """Adjust the plan based on self-critique feedback."""
        system_prompt = """You are a radiation oncology query planner.
Adjust the search plan based on the feedback received."""

        replan_prompt = f"""The current plan needs adjustment.

Original Query: "{plan.original_query}"

Current Plan:
{self._format_plan(plan)}

Feedback: {feedback}

Create an updated plan that addresses the feedback.
Respond in the same JSON format as before."""

        try:
            result = self.llm.generate_structured(
                replan_prompt,
                output_schema={"reasoning": "string", "sub_tasks": "array"},
                system_prompt=system_prompt
            )

            if "error" not in result and "sub_tasks" in result:
                new_sub_tasks = []

                for task in result["sub_tasks"]:
                    subtopic_query = task.get("query", "")
                    # ── Same pattern: each replanned subtopic → DENSE + KB ──
                    new_sub_tasks.append(SubTask(
                        task_id=self._create_task_id(),
                        description=f"[Guidelines] {task.get('description', '')}",
                        query=subtopic_query,
                        retrieval_strategy=RetrievalStrategy.DENSE,
                        status=TaskStatus.PENDING
                    ))
                    new_sub_tasks.append(SubTask(
                        task_id=self._create_task_id(),
                        description=f"[Trial papers] {task.get('description', '')}",
                        query=subtopic_query,
                        retrieval_strategy=RetrievalStrategy.KNOWLEDGE_BASE,
                        status=TaskStatus.PENDING
                    ))

                if new_sub_tasks:
                    return TaskPlan(
                        plan_id=self._create_plan_id(),
                        original_query=plan.original_query,
                        analysis=plan.analysis,
                        sub_tasks=new_sub_tasks,
                        reasoning=f"Replanned: {result.get('reasoning', '')}"
                    )

        except Exception as e:
            print(f"Replanning failed: {e}")

        return plan

    def _format_plan(self, plan: TaskPlan) -> str:
        """Format plan for display/debugging."""
        lines = [f"Plan ID: {plan.plan_id}", f"Reasoning: {plan.reasoning}", "Sub-tasks:"]
        for task in plan.sub_tasks:
            lines.append(f"  - [{task.status.value}] {task.description}")
            lines.append(f"    Query: {task.query}")
            lines.append(f"    Strategy: {task.retrieval_strategy.value}")
        return "\n".join(lines)
