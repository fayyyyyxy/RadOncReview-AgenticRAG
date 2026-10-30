"""
Agentic RAG Engine: Main agent that orchestrates all components.
"""
import time
import uuid
from dataclasses import replace
from typing import List, Dict, Optional, Any
from datetime import datetime

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
from .guardrails import GuardrailManager
from .case_labels import CaseLabels
from .site_routing import EvidenceScope, scope_for, table_fingerprint
from .institutional_examples import CaseContext, LocalInstitutionalExamples
from .retrieval_identity import passage_identity


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

    def __init__(self, config: Config = None, use_mock: bool = False, enable_judge: bool = True,
                 example_provider=None):
        """
        Initialize the Agentic RAG Engine.

        Args:
            config: Configuration object. Uses default if not provided.
            enable_judge: If False, skip LLM judge evaluation (for batch eval).
            use_mock: If True, use mock LLM and retriever for testing.
        """
        self.config = config or default_config
        self.use_mock = use_mock
        self.enable_judge = enable_judge
        self.example_provider = example_provider
        if self.config.features.institutional_examples and self.example_provider is None:
            self.example_provider = LocalInstitutionalExamples(
                self.config.features.past_case_index_path,
                self.config.embedding.model_id, self.config.embedding.dimension,
                expected_source_sha256=self.config.features.past_case_source_sha256,
            )

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

    def _collect_all_chunks(self, plan: TaskPlan) -> List[Dict]:
        """Collect all retrieved chunks from completed tasks."""
        all_chunks = []
        seen_ids = set()

        for task in plan.sub_tasks:
            if task.status == TaskStatus.COMPLETED:
                for chunk in task.retrieved_chunks:
                    chunk_id = self._chunk_key(chunk)
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
        Each topic → rerank all its candidates → top complex_rerank_top_k=3 chunks that no earlier topic took.

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

        # ── Rerank per topic: FAISS + KB chunks together, then take the top ones no earlier topic took ──
        # ⚠️ Repeats across topics are removed after reranking, not before. Removing them first took every
        # candidate of an earlier topic, including the ones its reranking dropped, out of later topics' pools.
        reranked_by_topic = {}
        taken = set()  # passages already given to an earlier topic

        for topic_key, data in chunks_by_topic.items():
            label = data["label"]
            # Repeats inside one topic (the same passage found by two searches) go before reranking
            chunks, seen_ids = [], set()
            for chunk in data["chunks"]:
                chunk_id = self._chunk_key(chunk)
                if chunk_id not in seen_ids:
                    chunks.append(chunk)
                    seen_ids.add(chunk_id)

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

            # Rank every candidate for this topic (FAISS + KB combined); the cut to top k comes after the repeats
            reranked = self.retriever.rerank(
                f"{plan.original_query} {label}",
                chunk_objects,
                top_k=len(chunk_objects)
            )

            # Convert back to dict format preserving all fields, skipping passages an earlier topic took
            final_chunks = []
            for chunk in reranked:
                if len(final_chunks) == self.config.retriever.complex_rerank_top_k:
                    break
                candidate = {
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
                }
                # 💡 A passage an earlier topic took is skipped, so the next one in this topic's ranking fills the place
                if self._chunk_key(candidate) not in taken:
                    final_chunks.append(candidate)
                    taken.add(self._chunk_key(candidate))

            reranked_by_topic[topic_key] = final_chunks

        return reranked_by_topic

    @staticmethod
    def _chunk_key(chunk: Dict) -> Any:
        """Use the entire text to distinguish trial passages sharing the same opening."""
        return passage_identity(chunk)

    def _merge_chunks(self, first: List[Dict], added: List[Dict]) -> List[Dict]:
        """The first round's passages in their order, then the added ones they do not already include."""
        seen = {self._chunk_key(c) for c in first}
        return first + [c for c in added if self._chunk_key(c) not in seen]

    def _merge_topics(self, first: Dict[str, List[Dict]], added: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """The first round's topics, then each added topic's passages that no earlier topic includes."""
        merged = {topic: list(chunks) for topic, chunks in first.items()}
        seen = {self._chunk_key(c) for chunks in first.values() for c in chunks}
        for topic, chunks in added.items():
            new = [c for c in chunks if self._chunk_key(c) not in seen]
            seen.update(self._chunk_key(c) for c in new)
            if new:
                # 💡 setdefault: a re-planned search worded like a first-round one joins that topic
                merged.setdefault(topic, []).extend(new)
        return merged

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
              case_context: Optional[CaseContext] = None) -> AgentResponse:
        """
        Process a user query through the agentic RAG pipeline.

        Args:
            user_query: The user's clinical question.
            labels: Case labels selected by the user; used only when a feature switch asks for them.
            case_context: Exact vignette/question and study linkage, required when examples are enabled.

        Returns:
            AgentResponse with answer, citations, confidence, and reasoning trace.
        """
        start_time = time.time()
        self.reasoning_trace = []
        self.synthesizer.requests.clear()

        # ========== Step 1: Input Guardrails ==========
        self._log_reasoning("GUARDRAIL", "Checking input safety")
        input_check = self.guardrails.check_input(user_query)

        if not input_check.passed:
            self._log_reasoning("GUARDRAIL", f"Input blocked: {input_check.warnings}")
            return AgentResponse(
                query=user_query,
                answer="I cannot process this query due to safety concerns. " +
                       "; ".join(input_check.warnings),
                confidence_score=0.0,
                reasoning_trace=self.reasoning_trace,
                latency_ms=(time.time() - start_time) * 1000
            )

        # Use filtered text if PHI was removed
        processed_query = input_check.filtered_text or user_query

        # Linkage is validated before paid calls and is never included in the clinical query.
        if self.config.features.institutional_examples:
            if case_context is None or case_context.query() != processed_query:
                raise ValueError("Institutional examples require structured case context matching the query.")
            self.example_provider.validate_context(case_context)

        # ========== Step 2: Query Analysis ==========
        self._log_reasoning("ANALYZE", "Analyzing query intent and entities")

        # Use LLM-enhanced analysis for complex queries, rule-based for simple one
        analysis = self.query_analyzer.analyze(processed_query)

        if (analysis.is_complex or analysis.confidence < 0.7) and not self.config.features.versioned_planning:
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
        plan = self.task_planner.plan(analysis)
        analysis = plan.analysis
        planning_events = [dict(plan.planning)] if plan.planning else []

        self._log_reasoning("PLAN",
            f"Created {len(plan.sub_tasks)} sub-tasks: {plan.reasoning}"
        )

        # ========== Step 4: Execute Retrieval Tasks ==========
        labels = labels or CaseLabels()
        scope = scope_for(labels.disease_site) if self.config.features.site_filter else EvidenceScope()
        if scope != EvidenceScope():
            self._log_reasoning("ROUTE", f"Evidence limited to {labels.disease_site}")
        self._log_reasoning("RETRIEVE", "Executing retrieval tasks")

        for task in plan.sub_tasks:
            self._log_reasoning("RETRIEVE",
                f"Task '{task.description}' using {task.retrieval_strategy.value}"
            )
            self.retriever.execute_task(task, scope)
            self._log_reasoning("RETRIEVE", f"Result: {task.result}")

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
            total_chunks = len(all_chunks)
            self._log_reasoning("RETRIEVE",
                f"Collected {len(all_chunks)} chunks (cross-encoder reranked)"
            )

        example_audit = {"enabled": self.config.features.institutional_examples}
        selected_examples = None

        def writer_options(evidence_count):
            nonlocal selected_examples
            if not self.config.features.institutional_examples:
                return {}
            example_audit.update(self.example_provider.provenance())
            if not evidence_count:
                examples = []
                example_audit["status"] = "no_evidence"
            else:
                if selected_examples is None:
                    selected_examples = self.example_provider.select(
                        case_context, labels, self.retriever._get_embedding,
                    )
                    self._log_reasoning("EXAMPLES", f"Selected {len(selected_examples)} institutional examples")
                examples = selected_examples
                example_audit["status"] = "selected" if examples else "no_matching_examples"
            example_audit["examples"] = examples
            return {"institutional_examples": examples}

        # ========== Step 5: Response Synthesis ==========
        self._log_reasoning("SYNTHESIZE", "Generating response from retrieved information")
        options = writer_options(total_chunks)

        if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
            answer, citations, confidence, context_passages = self.synthesizer.synthesize_multi_topic(
                processed_query, analysis, chunks_by_topic, **options
            )
        else:
            answer, citations, confidence, context_passages = self.synthesizer.synthesize(
                processed_query, analysis, plan, all_chunks, **options
            )
        self._log_reasoning("SYNTHESIZE",
            f"Generated answer with {len(citations)} citations, confidence: {confidence:.2f}"
        )

        # ========== Step 6: Self-Critique ==========
        critique = None
        if self.config.agent.enable_self_critique:
            self._log_reasoning("CRITIQUE", "Performing self-evaluation")
            critique = self.self_critic.critique(processed_query, answer, analysis)

            self._log_reasoning("CRITIQUE",
                f"Complete: {critique.is_complete}, "
                f"Safe: {critique.is_safe}, "
                f"Confidence: {critique.confidence_score:.2f}"
            )

            # Retry if critique suggests improvement needed
            if self.self_critic.should_retry(critique):
                self._log_reasoning("CRITIQUE", "Answer needs improvement, retrying...")

                # Get improvement guidance
                guidance = self.self_critic.get_improvement_guidance(critique)

                # Replan with feedback
                improved_plan = self.task_planner.replan(plan, guidance)
                if improved_plan.planning:
                    planning_events.append(dict(improved_plan.planning))

                # Re-execute new tasks
                for task in improved_plan.sub_tasks:
                    if task.status == TaskStatus.PENDING:
                        # ⚠️ The retry must keep the scope, or re-planned tasks would reach other sites
                        self.retriever.execute_task(task, scope)

                # Re-synthesize — use same path as first attempt based on query type
                # ⚠️ The new passages are added to the first round's, not swapped in, so a rewrite asked to fill a
                # gap keeps what the first draft was built on. Reranking old and new together would not do: the
                # reranker scores against the question, not the gap, so the first round's passages would win again.
                # The writer also gets the feedback and its draft; before, only the search planner saw the feedback
                if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
                    chunks_by_topic = self._merge_topics(
                        chunks_by_topic, self._collect_multi_topic_chunks(improved_plan)
                    )
                    options = writer_options(sum(len(chunks) for chunks in chunks_by_topic.values()))
                    answer, citations, confidence, context_passages = self.synthesizer.synthesize_multi_topic(
                        processed_query, analysis, chunks_by_topic, feedback=guidance, draft=answer, **options
                    )
                else:
                    all_chunks = self._merge_chunks(all_chunks, self._collect_all_chunks(improved_plan))
                    options = writer_options(len(all_chunks))
                    answer, citations, confidence, context_passages = self.synthesizer.synthesize(
                        processed_query, analysis, improved_plan, all_chunks, feedback=guidance, draft=answer, **options
                    )

                # Update confidence with critique score
                confidence = (confidence + critique.confidence_score) / 2
                # The recorded plan lists every search the answer drew on; a failed replan returns the plan unchanged
                if improved_plan is not plan:
                    plan = replace(improved_plan, sub_tasks=plan.sub_tasks + improved_plan.sub_tasks)

        # ========== Step 6b: Citation Check ==========
        # After any retry, so the final answer is the one checked
        citation_check = None
        if self.config.features.citation_check:
            self._log_reasoning("CHECK", "Checking each citation against its passage")
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
        final_answer, output_check = self.guardrails.process_output(answer)

        if not output_check.passed:
            self._log_reasoning("GUARDRAIL", f"Output concerns: {output_check.warnings}")
            # Add warning but don't block
            final_answer = f"⚠️ Note: {'; '.join(output_check.warnings)}\n\n{final_answer}"

        citations = self._enrich_citations_with_urls(citations)

        # ========== Build Final Response ==========
        latency_ms = (time.time() - start_time) * 1000
        self._log_reasoning("COMPLETE", f"Total latency: {latency_ms:.0f}ms")

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
            institutional_examples=example_audit,
            synthesis_requests=list(self.synthesizer.requests),
            routing={
                "labels": vars(labels),
                "planning_mode": plan.planning.get("mode", "llm"),
                "planning_events": planning_events,
                "site_filter": self.config.features.site_filter,
                "review_note_categories": sorted(scope.review_note_categories)
                if scope.review_note_categories is not None else None,
                "trial_paper_folders": sorted(scope.trial_paper_folders)
                if scope.trial_paper_folders is not None else None,
                "site_table": table_fingerprint(),
            },
            metadata={
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

        # ========== Step 8: LLM Judge Evaluation ==========
        if self.enable_judge:
            if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
                retrieved_texts = [
                    chunk.get("text", "")
                    for topic_chunks in chunks_by_topic.values()
                    for chunk in topic_chunks
                ]
            else:
                retrieved_texts = [chunk.get("text", "") for chunk in all_chunks]

            try:
                self._log_reasoning("JUDGE", "Evaluating with multi-agent judge")
                from evaluation.llm_judge_v3 import evaluate_single_query
                eval_result = evaluate_single_query(
                    question=user_query,
                    answer=final_answer,
                    contexts=retrieved_texts,
                )
                response.metadata["judge_evaluations"] = [eval_result]
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

        # Store in conversation history
        self.conversation_history.append({
            "query": user_query,
            "response": response,
            "timestamp": datetime.now()
        })

        return response

    def get_reasoning_trace(self) -> List[str]:
        """Get the reasoning trace from the last query."""
        return self.reasoning_trace

    def reset_conversation(self):
        """Reset conversation state."""
        self.conversation_history = []
        self.guardrails.reset()


def create_agent(use_mock: bool = False, config: Config = None) -> AgenticRAGEngine:
    """
    Factory function to create an Agentic RAG Engine.

    Args:
        use_mock: If True, create with mock components for testing.
        config: Optional custom configuration.

    Returns:
        Configured AgenticRAGEngine instance.
    """
    return AgenticRAGEngine(config=config, use_mock=use_mock)
