"""
Agentic RAG Engine: Main agent that orchestrates all components.
"""
import time
import uuid
from dataclasses import replace
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from .models import (
    AgentResponse, TaskPlan, QueryAnalysis, Citation,
    SelfCritiqueResult, TaskStatus, QueryIntent, RetrievedChunk
)
from config import Config, default_config
from .llm_client import create_llm_client, BaseLLMClient
from .query_analyzer import QueryAnalyzer
from .task_planner import TaskPlanner
from .retriever import RetrieverManager, MockRetrieverManager
from .synthesizer import ResponseSynthesizer, SelfCritic
from .citation_check import CitationChecker, cited
from .guardrails import GuardrailManager, InputGuardrail
from .case_labels import CaseLabels
from .site_routing import EvidenceScope, scope_for, table_fingerprint


class AgenticRAGEngine:
    """
    Main agentic RAG engine for radiation oncology guideline retrieval.

    Orchestrates:
    1. Query analysis and understanding
    2. Task planning and decomposition
    3. Dynamic retrieval strategy selection
    4. Response synthesis with citations
    5. Self-critique and iteration
    6. Safety guardrails
    """

    def __init__(self, config: Config = None, use_mock: bool = False, enable_judge: bool = True):
        """
        Initialize the Agentic RAG Engine.

        Args:
            config: Configuration object. Uses default if not provided.
            enable_judge: Whether to run all quality assessments by default.
            use_mock: If True, use mock LLM and retriever for testing.
        """
        self.config = config or default_config
        self.use_mock = use_mock
        self.enable_judge = enable_judge

        # Initialize LLM client
        self.llm = create_llm_client(
            self.config.llm,
            self.config.aws,
            use_mock=use_mock
        )

        # Initialize components
        self.query_analyzer = QueryAnalyzer(self.llm, self.config)
        self.task_planner = TaskPlanner(self.llm, self.config)

        if use_mock:
            self.retriever = MockRetrieverManager()
        else:
            self.retriever = RetrieverManager(
                self.config.retriever,
                self.config.embedding,
                self.config.aws
            )

        self.synthesizer = ResponseSynthesizer(self.llm, self.config)
        self.self_critic = SelfCritic(self.llm, self.config.agent)
        self.citation_checker = CitationChecker(self.llm)
        self.guardrails = GuardrailManager(self.config.guardrail)

        # State
        self.conversation_history = []
        self.reasoning_trace = []

    def _log_reasoning(self, step: str, details: str):
        """Add a step to the reasoning trace."""
        entry = f"[{datetime.now().strftime('%H:%M:%S')}] {step}: {details}"
        self.reasoning_trace.append(entry)
        print(entry)  # Also print for debugging

    def _execute_retrieval_tasks(self, tasks, scope):
        """Search concurrently, but consume results in plan order before deterministic reranking."""
        tasks = list(tasks)
        if not tasks:
            return
        workers = min(len(tasks), max(1, self.config.retriever.max_workers))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(self.retriever.execute_task, task, scope) for task in tasks]
            for task, future in zip(tasks, futures):
                future.result()
                self._log_reasoning("RETRIEVE", f"Task '{task.description}': {task.result}")

    def _collect_all_chunks(self, plan: TaskPlan) -> List[Dict]:
        """Collect all retrieved chunks from completed tasks."""
        all_chunks = []
        seen_ids = set()

        for task in plan.sub_tasks:
            if task.status == TaskStatus.COMPLETED:
                for chunk in task.retrieved_chunks:
                    source_type = chunk.get("source_type", "review_notes")
                    if source_type == "clinical_trial":
                        chunk_id = f"{chunk.get('s3_location', '')}_{chunk.get('text', '')[:100]}"
                    else:
                        chunk_id = chunk.get("faiss_id")
                    if chunk_id not in seen_ids:
                        all_chunks.append(chunk)
                        seen_ids.add(chunk_id)

        # Sort by relevance/similarity score
        all_chunks.sort(
            key=lambda x: x.get("relevance_score", 0) or x.get("similarity_score", 0),
            reverse=True
        )

        # Convert to RetrievedChunk objects for reranking
        chunk_objects = []
        for chunk in all_chunks:
            chunk_objects.append(RetrievedChunk(
                faiss_id=chunk.get("faiss_id", -1),
                chunk_id=chunk.get("chunk_id", 0),
                text=chunk.get("text", ""),
                path=chunk.get("path", ""),
                category=chunk.get("category", ""),
                headings={},
                similarity_score=chunk.get("similarity_score", 0) or chunk.get("relevance_score", 0),
                source_type=chunk.get("source_type", "review_notes"),
                document_title=chunk.get("document_title", ""),
                s3_location=chunk.get("s3_location", "")
            ))

        # Get the original query for reranking
        query = plan.original_query

        # Rerank using cross-encoder
        reranked_chunks = self.retriever.rerank(
            query,
            chunk_objects,
            top_k=self.config.retriever.rerank_top_k
        )

        # Convert back to dict format
        final_chunks = []
        for chunk in reranked_chunks:
            final_chunks.append({
                "faiss_id": chunk.faiss_id,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "path": chunk.path,
                "category": chunk.category,
                "similarity_score": chunk.similarity_score,
                "relevance_score": chunk.relevance_score,
                "source_type": chunk.source_type,
                "document_title": chunk.document_title,
                "s3_location": chunk.s3_location
            })

        return final_chunks

    # !!!! NEW METHOD FOR COMPARISON/COMPLEX QUERIES !!!!
    def _collect_multi_topic_chunks(self, plan: TaskPlan) -> Dict[str, List[Dict]]:
        """
        Collect chunks for comparison/complex queries with balanced per-topic reranking.

        Groups FAISS and KB tasks that share the same subtopic query together,
        so each logical subtopic gets chunks from BOTH sources merged before reranking.
        Each topic → rerank all sources together → top complex_rerank_top_k=3 chunks.

        Example for "Compare SBRT vs EBRT":
          Tasks: [Guidelines]SBRT, [Trial papers]SBRT, [Guidelines]EBRT, [Trial papers]EBRT
          Groups: {"SBRT prostate": 16 chunks (FAISS6+KB10), "EBRT prostate": 16 chunks}
          Result: {"SBRT prostate": 3 chunks, "EBRT prostate": 3 chunks} → 6 to LLM
        """
        chunks_by_topic = {}

        for task in plan.sub_tasks:
            if task.status == TaskStatus.COMPLETED:
                topic_key = task.query
                if topic_key not in chunks_by_topic:
                    chunks_by_topic[topic_key] = {
                        "label": (task.description
                                  .replace("[Guidelines] ", "")
                                  .replace("[Trial papers] ", "")
                                  .replace("Search RadOncReview guidelines: ", "")
                                  .replace("Search clinical trial papers: ", "")),
                        "chunks": []
                    }
                chunks_by_topic[topic_key]["chunks"].extend(task.retrieved_chunks)

        # ── Deduplicate across all topics (by faiss_id) ──
        all_seen_ids = set()
        deduped_by_topic = {}

        for topic_key, data in chunks_by_topic.items():
            deduped = []
            for chunk in data["chunks"]:
                source_type = chunk.get("source_type", "review_notes")
                if source_type == "clinical_trial":
                    chunk_id = f"{chunk.get('s3_location', '')}_{chunk.get('text', '')[:100]}"
                else:
                    chunk_id = chunk.get("faiss_id")
                if chunk_id not in all_seen_ids:
                    deduped.append(chunk)
                    all_seen_ids.add(chunk_id)
            deduped_by_topic[topic_key] = {
                "label": data["label"],
                "chunks": deduped
            }

        # ── Rerank per topic: FAISS + KB chunks together → top 5 ──
        reranked_by_topic = {}

        for topic_key, data in deduped_by_topic.items():
            label = data["label"]
            chunks = data["chunks"]

            if not chunks:
                reranked_by_topic[topic_key] = []
                continue

            # Convert to RetrievedChunk objects for cross-encoder
            chunk_objects = []
            for chunk in chunks:
                chunk_objects.append(RetrievedChunk(
                    faiss_id=chunk.get("faiss_id", -1),
                    chunk_id=chunk.get("chunk_id", 0),
                    text=chunk.get("text", ""),
                    path=chunk.get("path", ""),
                    category=chunk.get("category", ""),
                    headings={},
                    similarity_score=chunk.get("similarity_score", 0) or chunk.get("relevance_score", 0),
                    source_type=chunk.get("source_type", "review_notes"),
                    document_title=chunk.get("document_title", ""),
                    s3_location=chunk.get("s3_location", "")
                ))

            # Rerank all chunks for this topic (FAISS + KB combined) → top k
            reranked = self.retriever.rerank(
                f"{plan.original_query} {label}",
                chunk_objects,
                top_k=self.config.retriever.complex_rerank_top_k
            )

            # Convert back to dict format preserving all fields
            final_chunks = []
            for chunk in reranked:
                final_chunks.append({
                    "faiss_id": chunk.faiss_id,
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "path": chunk.path,
                    "category": chunk.category,
                    "similarity_score": chunk.similarity_score,
                    "relevance_score": chunk.relevance_score,
                    "source_type": chunk.source_type,
                    "document_title": chunk.document_title,
                    "s3_location": chunk.s3_location
                })

            reranked_by_topic[topic_key] = final_chunks

        return reranked_by_topic

    def _enrich_citations_with_urls(self, citations: List[Citation]) -> List[Citation]:
        for citation in citations:
            if not citation.source_url:
                continue
            if citation.source_url.startswith("s3://"):
                try:
                    path = citation.source_url.replace("s3://", "")
                    bucket = path.split("/")[0]
                    key = "/".join(path.split("/")[1:])
                    citation.source_url = self.retriever.s3.generate_presigned_url(
                        "get_object",
                        Params={"Bucket": bucket, "Key": key},
                        ExpiresIn=604800  # 7 days expire
                    )
                except Exception as e:
                    print(f"Presigned URL generation failed: {e}")
        return citations


    def query(self, user_query: str, labels: Optional[CaseLabels] = None,
              *, prior_questions: Optional[List[str]] = None,
              conversation_context: str = "",
              evidence_scope: Optional[EvidenceScope] = None,
              progress_callback: Optional[Callable[[str], None]] = None,
              retain_history: bool = True, run_evaluation: Optional[bool] = None) -> AgentResponse:
        """
        Process a user query through the agentic RAG pipeline.

        Args:
            user_query: The user's clinical question.
            labels: Case labels selected by the user; used only when a feature switch asks for them.
            prior_questions: At most two earlier user questions, checked separately for input safety.
            conversation_context: Editable user context, capped at 4,000 characters and checked for safety.
            evidence_scope: Explicit per-request source filter; otherwise use the configured switch.
            progress_callback: Receives stage names without clinical text.
            retain_history: False for stateless web requests; browser supplies bounded context.
            run_evaluation: Request the full quality review. Evidence sufficiency remains on either way.

        Returns:
            AgentResponse with answer, citations, confidence, and reasoning trace.
        """
        start_time = time.time()
        self.reasoning_trace = []

        def progress(stage):
            if progress_callback is not None:
                progress_callback(stage)

        # ========== Step 1: Input Guardrails ==========
        self._log_reasoning("GUARDRAIL", "Checking input safety")
        progress("input")
        input_check = self.guardrails.check_input(user_query)
        if len(conversation_context) > 4000:
            raise ValueError("Case context must be at most 4,000 characters.")
        if conversation_context:
            context_check = InputGuardrail(replace(self.config.guardrail, max_input_length=4000)).check(conversation_context)
            if not context_check.passed:
                input_check = context_check

        # Only user-authored context is carried forward; generated answers never become evidence.
        # Bound both turns and length without silently truncating clinical qualifiers.
        prior_questions = list(prior_questions or [])
        if len(prior_questions) > 2:
            raise ValueError("At most two earlier questions may be used as context.")
        for previous in prior_questions:
            previous_check = self.guardrails.check_input(previous)
            if not previous_check.passed:
                input_check = previous_check
                break

        if not input_check.passed:
            self._log_reasoning("GUARDRAIL", f"Input blocked: {input_check.warnings}")
            return AgentResponse(
                query=user_query,
                answer="I cannot process this query due to safety concerns. " +
                       "; ".join(input_check.warnings),
                confidence_score=0.0,
                metadata={"input_blocked": True},
                reasoning_trace=self.reasoning_trace,
                latency_ms=(time.time() - start_time) * 1000
            )

        # Use filtered text if PHI was removed
        processed_query = input_check.filtered_text or user_query
        if prior_questions or conversation_context:
            previous = "\n\n".join(f"Earlier question {i}: {q}" for i, q in enumerate(prior_questions, 1))
            processed_query = (
                "Use the earlier user questions only to resolve references and case context. "
                "They are not clinical evidence. Answer only the current question; "
                "its corrections take precedence over earlier context.\n\n"
                "The edited case context supersedes conflicting earlier questions.\n\n"
                "Within case context, later corrections supersede earlier entries.\n\n"
                f"{previous}\n\nEdited case context:\n{conversation_context}\n\n"
                f"Current clinical question:\n{processed_query}"
            )

        # ========== Step 2: Query Analysis ==========
        self._log_reasoning("ANALYZE", "Analyzing query intent and entities")
        progress("analyze")

        # Use LLM-enhanced analysis for complex queries, rule-based for simple one
        analysis = self.query_analyzer.analyze(processed_query)

        if analysis.is_complex or analysis.confidence < 0.7:
            self._log_reasoning("ANALYZE", "Using LLM for deeper analysis")
            analysis = self.query_analyzer.analyze_with_llm(processed_query)

        self._log_reasoning("ANALYZE",
            f"Intent: {analysis.intent.value}, "
            f"Disease: {analysis.disease_category}, "
            f"Modality: {analysis.treatment_modality}, "
            f"Complex: {analysis.is_complex}"
        )

        # ========== Step 3: Task Planning ==========
        self._log_reasoning("PLAN", "Creating execution plan")
        progress("plan")
        plan = self.task_planner.plan(analysis)

        self._log_reasoning("PLAN",
            f"Created {len(plan.sub_tasks)} sub-tasks: {plan.reasoning}"
        )

        # ========== Step 4: Execute Retrieval Tasks ==========
        labels = labels or CaseLabels()
        scope = evidence_scope if evidence_scope is not None else (
            scope_for(labels.disease_site) if self.config.features.site_filter else EvidenceScope())
        if scope != EvidenceScope():
            self._log_reasoning("ROUTE", f"Evidence limited to {labels.disease_site}")
        self._log_reasoning("RETRIEVE", "Executing retrieval tasks")
        progress("retrieve")

        self._execute_retrieval_tasks(plan.sub_tasks, scope)

        # Collect chunks differently based on query type
        if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
            chunks_by_topic = self._collect_multi_topic_chunks(plan)
            total_chunks = sum(len(chunks) for chunks in chunks_by_topic.values())
            self._log_reasoning("RETRIEVE",
                f"Collected {total_chunks} chunks across {len(chunks_by_topic)} topics "
                f"(balanced with cross-encoder reranking)"
            )
        else:
            # Use standard collection with cross-encoder reranking
            all_chunks = self._collect_all_chunks(plan)
            self._log_reasoning("RETRIEVE",
                f"Collected {len(all_chunks)} chunks (cross-encoder reranked)"
            )

        # ========== Step 5: Response Synthesis ==========
        self._log_reasoning("SYNTHESIZE", "Generating response from retrieved information")
        progress("synthesize")

        if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
            answer, citations, confidence, context_passages = self.synthesizer.synthesize_multi_topic(
                processed_query, analysis, chunks_by_topic
            )
        else:
            answer, citations, confidence, context_passages = self.synthesizer.synthesize(
                processed_query, analysis, plan, all_chunks
            )
        self._log_reasoning("SYNTHESIZE",
            f"Generated answer with {len(citations)} citations, confidence: {confidence:.2f}"
        )

        # ========== Step 6: Self-Critique ==========
        critique = None
        if self.config.agent.enable_self_critique:
            self._log_reasoning("CRITIQUE", "Performing self-evaluation")
            progress("critique")
            critique = self.self_critic.critique(
                processed_query, answer, citations, analysis
            )

            self._log_reasoning("CRITIQUE",
                f"Complete: {critique.is_complete}, "
                f"Accurate: {critique.is_accurate}, "
                f"Safe: {critique.is_safe}, "
                f"Confidence: {critique.confidence_score:.2f}"
            )

            # Retry if critique suggests improvement needed
            if self.self_critic.should_retry(critique):
                self._log_reasoning("CRITIQUE", "Answer needs improvement, retrying...")
                progress("refine")

                # Get improvement guidance
                guidance = self.self_critic.get_improvement_guidance(critique)

                # Replan with feedback
                improved_plan = self.task_planner.replan(plan, guidance)

                # Re-execute new tasks
                self._execute_retrieval_tasks(
                    (task for task in improved_plan.sub_tasks if task.status == TaskStatus.PENDING), scope)

                # Re-synthesize — use same path as first attempt based on query type
                if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
                    chunks_by_topic = self._collect_multi_topic_chunks(improved_plan)
                    answer, citations, confidence, context_passages = self.synthesizer.synthesize_multi_topic(
                        processed_query, analysis, chunks_by_topic
                    )
                else:
                    all_chunks = self._collect_all_chunks(improved_plan)
                    answer, citations, confidence, context_passages = self.synthesizer.synthesize(
                        processed_query, analysis, improved_plan, all_chunks
                    )

                # Update confidence with critique score
                confidence = (confidence + critique.confidence_score) / 2
                plan = improved_plan

        # ========== Step 6b: Citation Check ==========
        # After any retry, so the final answer is the one checked
        citation_check = None
        if self.config.features.citation_check:
            self._log_reasoning("CHECK", "Checking each citation against its passage")
            progress("citations")
            answer_before = answer
            answer, citation_check = self.citation_checker.check(answer, context_passages)
            # 💡 Kept so that a change to the check can be replayed on the same answer instead of a new run
            citation_check["answer_before"] = answer_before
            # Only citations still in the answer stay in the source list
            remaining = cited(answer)
            citations = [c for c in citations if c.citation_id in remaining]
            self._log_reasoning("CHECK",
                f"Removed {citation_check['removed']} of {citation_check['citations']} citations, "
                f"{citation_check['unchecked']} unchecked"
                + (f", check failed: {citation_check['error']}" if citation_check["error"] else "")
            )

        # ========== Step 7: Output Guardrails ==========
        self._log_reasoning("GUARDRAIL", "Checking output safety")
        progress("safety")
        final_answer, output_check = self.guardrails.process_output(answer)

        if not output_check.passed:
            self._log_reasoning("GUARDRAIL", f"Output concerns: {output_check.warnings}")
            # Add warning but don't block
            final_answer = f"⚠️ Note: {'; '.join(output_check.warnings)}\n\n{final_answer}"

        citations = self._enrich_citations_with_urls(citations)

        # ========== Build Final Response ==========
        latency_ms = (time.time() - start_time) * 1000
        self._log_reasoning("ANSWER_READY", f"Answer checks completed: {latency_ms:.0f}ms; evaluation follows")

        if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
            total_chunks = sum(len(v) for v in chunks_by_topic.values())
        else:
            total_chunks = len(all_chunks)

        response = AgentResponse(
            query=user_query,
            answer=final_answer,
            citations=citations,
            confidence_score=confidence,
            reasoning_trace=self.reasoning_trace,
            self_critique=critique,
            task_plan=plan,
            context_passages=context_passages,
            routing={
                "labels": vars(labels),
                "site_filter": scope != EvidenceScope(),
                "review_note_categories": sorted(scope.review_note_categories)
                if scope.review_note_categories is not None else None,
                "trial_paper_folders": sorted(scope.trial_paper_folders)
                if scope.trial_paper_folders is not None else None,
                "site_table": table_fingerprint(),
            },
            metadata={
                "context": {
                    "earlier_questions": len(prior_questions),
                    "case_context_chars": len(conversation_context),
                },
                "analysis": {
                    "intent": analysis.intent.value,
                    "disease_category": analysis.disease_category,
                    "treatment_modality": analysis.treatment_modality,
                    "is_complex": analysis.is_complex
                },
                "retrieval": {
                    "total_chunks": total_chunks,
                    "tasks_completed": sum(
                        1 for t in plan.sub_tasks if t.status == TaskStatus.COMPLETED
                    ),
                    "tasks_total": len(plan.sub_tasks)
                }
            },
            latency_ms=latency_ms
        )

        if citation_check is not None:
            response.metadata["citation_check"] = citation_check

        # Evidence sufficiency stays on, even when the five additional assessments are off.
        retrieved_texts = [p.get("text", "") for p in context_passages]
        progress("evidence")
        evidence_check = {"status": "unavailable", "flagged": None, "judgment": None,
                          "reason": "The evidence sufficiency check could not finish."}
        try:
            if self.use_mock:
                evidence_check["reason"] = "Evidence sufficiency is not assessed in mock mode."
            else:
                from evaluation.llm_judge_v3 import evaluate_abstention
                evidence_check = evaluate_abstention(processed_query, final_answer, retrieved_texts)
        except Exception as exc:
            self._log_reasoning("EVIDENCE", f"Evidence sufficiency check unavailable: {type(exc).__name__}")
        response.metadata["evidence_check"] = evidence_check

        # ========== Step 8: Optional Full Quality Evaluation ==========
        full_evaluation = self.enable_judge if run_evaluation is None else run_evaluation
        response.metadata["quality_evaluation"] = {"requested": full_evaluation, "status": "skipped"}
        if full_evaluation:
            progress("evaluate")

            try:
                self._log_reasoning("JUDGE", "Evaluating with multi-agent judge")
                from evaluation.llm_judge_v3 import evaluate_single_query
                eval_result = evaluate_single_query(
                    question=processed_query,
                    answer=final_answer,
                    contexts=retrieved_texts,
                    abstention_check=evidence_check,
                )
                response.metadata["judge_evaluations"] = [eval_result]
                response.metadata["quality_evaluation"]["status"] = "completed"
                # A full review may recover a failed initial evidence check.
                judgment = eval_result.get("abstention_judgment")
                if judgment in {"Appropriately Answered", "Should Have Abstained", "Partial Abstention Preferred"}:
                    response.metadata["evidence_check"] = {
                        "status": "completed", "judgment": judgment,
                        "reason": eval_result.get("abstention_justification", ""),
                        "flagged": judgment != "Appropriately Answered",
                    }
                self._log_reasoning("JUDGE",
                    f"Retrieval Adequacy: {eval_result['retrieval_adequacy']}/5, "
                    f"Faithfulness: {eval_result['faithfulness']}/5, "
                    f"Correctness: {eval_result['correctness']}/5, "
                    f"Answer Quality: {eval_result['answer_quality']}/5, "
                    f"Composite: {eval_result['composite_score']:.2f}/5.00"
                    + (" ⚠️ ABSTAINED" if eval_result['abstention_triggered'] else "")
                )
            except Exception as e:
                err_msg = str(e).lower()
                if "token has expired" in err_msg or "sso" in err_msg or "refresh failed" in err_msg:
                    raise
                self._log_reasoning("JUDGE", f"Evaluation failed: {str(e)}")
                response.metadata["judge_evaluations"] = []
                response.metadata["quality_evaluation"]["status"] = "failed"

        # Include online evaluation when enabled; it finishes after answer generation.
        response.metadata["answer_ready_ms"] = latency_ms
        response.latency_ms = (time.time() - start_time) * 1000
        self._log_reasoning("COMPLETE", f"Total latency: {response.latency_ms:.0f}ms")

        if retain_history:
            self.conversation_history.append({
                "query": user_query, "response": response, "timestamp": datetime.now()
            })
        progress("complete")

        return response

    def get_reasoning_trace(self) -> List[str]:
        """Get the reasoning trace from the last query."""
        return self.reasoning_trace

    def reset_conversation(self):
        """Reset conversation state."""
        self.conversation_history = []
        self.guardrails.reset()


def create_agent(use_mock: bool = False, config: Config = None, enable_judge: bool = False) -> AgenticRAGEngine:
    """
    Factory function to create an Agentic RAG Engine.

    Args:
        use_mock: If True, create with mock components for testing.
        config: Optional custom configuration.

    Returns:
        Configured AgenticRAGEngine instance.
    """
    return AgenticRAGEngine(config=config, use_mock=use_mock, enable_judge=enable_judge)
