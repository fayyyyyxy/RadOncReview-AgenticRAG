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
from config import AgentConfig


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
        QueryIntent.COMPARISON: RetrievalStrategy.MULTI_HOP,
        QueryIntent.DEFINITION: RetrievalStrategy.DENSE,
        QueryIntent.GENERAL: RetrievalStrategy.DENSE,
    }
    
    def __init__(self, llm_client: BaseLLMClient, config: AgentConfig):
        self.llm = llm_client
        self.config = config
    
    def _create_task_id(self) -> str:
        """Generate a unique task ID."""
        return f"task_{uuid.uuid4().hex[:8]}"
    
    def _create_plan_id(self) -> str:
        """Generate a unique plan ID."""
        return f"plan_{uuid.uuid4().hex[:8]}"
    
    def _get_default_strategy(self, analysis: QueryAnalysis) -> RetrievalStrategy:
        """Get default retrieval strategy based on query analysis."""
        return self.STRATEGY_MAP.get(analysis.intent, RetrievalStrategy.HYBRID)
    
    def _create_simple_plan(self, analysis: QueryAnalysis) -> TaskPlan:
        """Create a single-task plan for simple queries."""
        strategy = self._get_default_strategy(analysis)
        
        # Build search query with context
        search_query = analysis.original_query
        if analysis.disease_category:
            search_query = f"{analysis.disease_category}: {search_query}"
        
        sub_task = SubTask(
            task_id=self._create_task_id(),
            description=f"Retrieve information for: {analysis.original_query}",
            query=search_query,
            retrieval_strategy=strategy,
            status=TaskStatus.PENDING
        )
        
        return TaskPlan(
            plan_id=self._create_plan_id(),
            original_query=analysis.original_query,
            analysis=analysis,
            sub_tasks=[sub_task],
            reasoning="Simple query - single retrieval task sufficient."
        )
    
    def _create_comparison_plan(self, analysis: QueryAnalysis) -> TaskPlan:
        """Create a multi-task plan for comparison queries."""
        # Extract comparison targets from query
        query_lower = analysis.original_query.lower()
        
        # Common comparison patterns
        comparison_pairs = []
        if ' vs ' in query_lower or ' versus ' in query_lower:
            parts = query_lower.replace(' versus ', ' vs ').split(' vs ')
            if len(parts) >= 2:
                comparison_pairs = [(parts[0].strip(), parts[1].strip())]
        
        sub_tasks = []
        
        if comparison_pairs:
            for item1, item2 in comparison_pairs:
                # Task for first item
                sub_tasks.append(SubTask(
                    task_id=self._create_task_id(),
                    description=f"Retrieve information about {item1}",
                    query=f"{analysis.disease_category or ''} {item1}".strip(),
                    retrieval_strategy=RetrievalStrategy.DENSE,
                    status=TaskStatus.PENDING
                ))
                
                # Task for second item
                sub_tasks.append(SubTask(
                    task_id=self._create_task_id(),
                    description=f"Retrieve information about {item2}",
                    query=f"{analysis.disease_category or ''} {item2}".strip(),
                    retrieval_strategy=RetrievalStrategy.DENSE,
                    status=TaskStatus.PENDING
                ))
        else:
            # Fallback: single comprehensive search
            sub_tasks.append(SubTask(
                task_id=self._create_task_id(),
                description="Retrieve comparison information",
                query=analysis.original_query,
                retrieval_strategy=RetrievalStrategy.HYBRID,
                status=TaskStatus.PENDING
            ))
        
        return TaskPlan(
            plan_id=self._create_plan_id(),
            original_query=analysis.original_query,
            analysis=analysis,
            sub_tasks=sub_tasks,
            reasoning="Comparison query - separate retrievals for each item being compared."
        )
    
    def plan(self, analysis: QueryAnalysis) -> TaskPlan:
        """Create an execution plan based on query analysis (rule-based)."""
        # Comparison queries need multiple retrievals
        if analysis.intent == QueryIntent.COMPARISON:
            return self._create_comparison_plan(analysis)
        
        # Complex queries may need decomposition
        if analysis.is_complex:
            return self._plan_with_llm(analysis)
        
        # Simple queries get a single-task plan
        return self._create_simple_plan(analysis)
    
    def _plan_with_llm(self, analysis: QueryAnalysis) -> TaskPlan:
        """Use LLM to create a plan for complex queries."""
        system_prompt = """You are a radiation oncology query planner. 
Your job is to break down complex clinical questions into simpler sub-tasks 
that can be answered by searching a knowledge base of treatment guidelines."""
        
        planning_prompt = f"""Break down this radiation oncology query into sub-tasks:

Original Query: "{analysis.original_query}"

Query Analysis:
- Intent: {analysis.intent.value}
- Disease Category: {analysis.disease_category or "Not specified"}
- Treatment Modality: {analysis.treatment_modality or "Not specified"}
- Key Entities: {", ".join(analysis.key_entities) if analysis.key_entities else "None"}

Create a plan with 1-4 sub-tasks. Each sub-task should:
1. Be a focused search query
2. Contribute to answering the original question
3. Be independent (can be executed in parallel)

Respond in JSON format:
{{
    "reasoning": "<explanation of your decomposition strategy>",
    "sub_tasks": [
        {{
            "description": "<what this task will find>",
            "query": "<search query for knowledge base>",
            "strategy": "<dense|sparse|hybrid>"
        }}
    ]
}}"""
        
        try:
            result = self.llm.generate_structured(
                planning_prompt,
                output_schema={
                    "reasoning": "string",
                    "sub_tasks": "array"
                },
                system_prompt=system_prompt
            )
            
            if "error" not in result and "sub_tasks" in result:
                sub_tasks = []
                for i, task in enumerate(result["sub_tasks"]):
                    strategy_map = {
                        "dense": RetrievalStrategy.DENSE,
                        "sparse": RetrievalStrategy.SPARSE,
                        "hybrid": RetrievalStrategy.HYBRID
                    }
                    
                    sub_tasks.append(SubTask(
                        task_id=self._create_task_id(),
                        description=task.get("description", f"Sub-task {i+1}"),
                        query=task.get("query", analysis.original_query),
                        retrieval_strategy=strategy_map.get(
                            task.get("strategy", "hybrid"), 
                            RetrievalStrategy.HYBRID
                        ),
                        status=TaskStatus.PENDING
                    ))
                
                return TaskPlan(
                    plan_id=self._create_plan_id(),
                    original_query=analysis.original_query,
                    analysis=analysis,
                    sub_tasks=sub_tasks if sub_tasks else [self._create_simple_plan(analysis).sub_tasks[0]],
                    reasoning=result.get("reasoning", "LLM-generated plan")
                )
        except Exception as e:
            print(f"LLM planning failed, using simple plan: {e}")
        
        # Fallback to simple plan
        return self._create_simple_plan(analysis)
    
    def replan(self, plan: TaskPlan, feedback: str) -> TaskPlan:
        """Adjust the plan based on execution feedback."""
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
                output_schema={
                    "reasoning": "string",
                    "sub_tasks": "array"
                },
                system_prompt=system_prompt
            )
            
            if "error" not in result and "sub_tasks" in result:
                new_sub_tasks = []
                for task in result["sub_tasks"]:
                    strategy_map = {
                        "dense": RetrievalStrategy.DENSE,
                        "sparse": RetrievalStrategy.SPARSE,
                        "hybrid": RetrievalStrategy.HYBRID
                    }
                    
                    new_sub_tasks.append(SubTask(
                        task_id=self._create_task_id(),
                        description=task.get("description", ""),
                        query=task.get("query", ""),
                        retrieval_strategy=strategy_map.get(
                            task.get("strategy", "hybrid"),
                            RetrievalStrategy.HYBRID
                        ),
                        status=TaskStatus.PENDING
                    ))
                
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
        """Format plan for display."""
        lines = [f"Plan ID: {plan.plan_id}", f"Reasoning: {plan.reasoning}", "Sub-tasks:"]
        for task in plan.sub_tasks:
            lines.append(f"  - [{task.status.value}] {task.description}")
            lines.append(f"    Query: {task.query}")
            lines.append(f"    Strategy: {task.retrieval_strategy.value}")
        return "\n".join(lines)
