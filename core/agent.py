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
    SelfCritiqueResult, TaskStatus, QueryIntent, RetrievedChunk, SubTask, RetrievalStrategy
)
from config import Config, default_config
from .llm_client import create_llm_client, BaseLLMClient
from .query_analyzer import QueryAnalyzer
from .task_planner import TaskPlanner
from .retriever import RetrieverManager, MockRetrieverManager
from .synthesizer import ResponseSynthesizer, SelfCritic
from .citation_check import CitationChecker, cited
from .critic_edits import GAP_LIMIT, GapFiller, apply_edits
from .guardrails import GuardrailManager
from .case_labels import NOT_SPECIFIED, CaseLabels
from .site_routing import EvidenceScope, scope_for, table_fingerprint
from .institutional_examples import CaseContext, LocalInstitutionalExamples


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
        # The critique may run on a model of its own, so the answer is checked by a model other than its writer
        critic_model = self.config.features.critic_model
        critic_llm = create_llm_client(replace(self.config.llm, model_id=critic_model), self.config.aws,
                                       use_mock=use_mock) if critic_model else self.llm
        self.self_critic = SelfCritic(critic_llm, self.config.agent, named_gap=self.config.features.critic_named_gap,
                                      edits=self.config.features.critic_edits)
        # The sentence for a searched gap is answer text, so the writer's model writes it and the critique checks it
        self.gap_filler = GapFiller(self.llm)
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
    def _collect_multi_topic_chunks(self, plan: TaskPlan, exclude: Optional[set] = None,
                                    site: Optional[str] = None) -> Dict[str, List[Dict]]:
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
        # Passages already given to an earlier topic, or to the answer before this search (exclude)
        taken = set(exclude or ())

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
            # ⚠️ Against the whole vignette, the reranker scored resemblance to the case story: on-site against
            # off-site AUC 0.54 over run3's sent passages, 0.75 with the site instead. A re-planned topic usually has
            # no label, so under the site it ranks on the site alone, as it was measured
            reranked = self.retriever.rerank(
                f"{site} {label}".strip() if site else f"{plan.original_query} {label}",
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
        """One identity per passage; trial-paper chunks have no FAISS id, so their file and opening text stand in."""
        if chunk.get("source_type", "review_notes") == "clinical_trial":
            return f"{chunk.get('s3_location', '')}_{chunk.get('text', '')[:100]}"
        return chunk.get("faiss_id")

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

    def _search_gaps(self, query: str, plan: TaskPlan, scope: EvidenceScope, gaps: List[Dict], answer: str,
                     first_chunks: List[Dict], passages: List[Dict], report: Dict) -> tuple:
        """Search once for each gap the critique verified and add what a new passage covers (critic edits).

        Returns the answer, the passages, the plan with the searches added, and whether any text was added."""
        # The limit is kept here, where the searches happen, whatever the critique returned
        gaps = gaps[:GAP_LIMIT]
        report.update({"gaps": gaps, "gap_passages": [], "fill": None})
        if not gaps:
            return answer, passages, plan, False
        retriever = self.config.retriever
        # 💡 Paired as the planner pairs them, one search of the review notes and one of the trial papers, and
        # labelled with the item, so each gap is reranked against the question plus that item
        tasks = [SubTask(task_id=f"gap_{i}_{source}", description=f"{prefix} {gap['item']}", query=gap["query"],
                         retrieval_strategy=strategy, top_k=top_k, status=TaskStatus.PENDING)
                 for i, gap in enumerate(gaps, start=1)
                 for source, prefix, strategy, top_k in (
                     ("notes", "[Guidelines]", RetrievalStrategy.DENSE, retriever.faiss_complex_top_k),
                     ("trials", "[Trial papers]", RetrievalStrategy.KNOWLEDGE_BASE, retriever.kb_complex_top_k))]
        for task in tasks:
            self.retriever.execute_task(task, scope)
        # ⚠️ The answer's own passages are left out before the cut, so each gap keeps up to three new ones; the
        # critique already found that they do not cover it
        found = self._collect_multi_topic_chunks(replace(plan, sub_tasks=tasks),
                                                 exclude={self._chunk_key(c) for c in first_chunks})
        new_chunks = [c for chunks in found.values() for c in chunks]
        plan = replace(plan, sub_tasks=plan.sub_tasks + tasks)
        self._log_reasoning("GAPS", f"Searched {len(gaps)} gaps, found {len(new_chunks)} new passages")
        if not new_chunks:
            return answer, passages, plan, False
        start = max((p["citation_id"] for p in passages), default=0) + 1
        context, added = self.synthesizer.extend_passages(new_chunks, start)
        report["gap_passages"] = [p["citation_id"] for p in added]
        answer, report["fill"] = apply_edits(answer, self.gap_filler.fill(query, answer, gaps, context))
        self._log_reasoning("GAPS", f"Filled {len(report['fill']['applied'])} of {len(gaps)} gaps")
        return answer, passages + added, plan, bool(report["fill"]["applied"])

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
              case_context: Optional[CaseContext] = None, plan: Optional[TaskPlan] = None) -> AgentResponse:
        """
        Process a user query through the agentic RAG pipeline.

        Args:
            user_query: The user's clinical question.
            labels: Case labels selected by the user; used only when a feature switch asks for them.
            case_context: Exact vignette/question and study linkage, required when examples are enabled.
            plan: A saved plan to search with instead of analyzing and planning again (evaluation runs only).

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

        # A plan made for another query would search for the wrong patient; checked before any paid call
        if plan is not None and plan.original_query != processed_query:
            raise ValueError("A replayed plan must have been made for this query.")

        # ========== Step 2: Query Analysis ==========
        if plan is not None:
            # 💡 The plan carries the analysis it was made from, so a replay also takes the same single- or
            # multi-topic path and gives the writer and self-critique the same intent and disease
            analysis = plan.analysis
            self._log_reasoning("ANALYZE", "Using the analysis of a replayed plan")
        else:
            self._log_reasoning("ANALYZE", "Analyzing query intent and entities")

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
        if plan is None:
            self._log_reasoning("PLAN", "Creating execution plan")
            plan = self.task_planner.plan(analysis)
        else:
            self._log_reasoning("PLAN", "Replaying a saved plan")

        # ⚠️ The runner reads the first round's size and reasoning from this line when it replays a run, so keep its
        # wording; the plan's own reasoning is overwritten by a re-plan
        self._log_reasoning("PLAN",
            f"Created {len(plan.sub_tasks)} sub-tasks: {plan.reasoning}"
        )

        # ========== Step 4: Execute Retrieval Tasks ==========
        labels = labels or CaseLabels()
        scope = scope_for(labels.disease_site) if self.config.features.site_filter else EvidenceScope()
        rerank_site = (labels.disease_site if self.config.features.rerank_site_query
                       and labels.disease_site != NOT_SPECIFIED else None)
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
            chunks_by_topic = self._collect_multi_topic_chunks(plan, site=rerank_site)
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
        edit_report = None
        edits_on = self.config.features.critic_edits
        if self.config.agent.enable_self_critique:
            self._log_reasoning("CRITIQUE", "Performing self-evaluation")
            # Kept with the critique, so why an answer was rewritten can be read against the text that was judged
            first_draft = answer
            # Only the edit critique reads the passages; the default one never did
            extra = {"passages": context_passages} if edits_on else {}
            critique = self.self_critic.critique(processed_query, answer, analysis, **extra)

            self._log_reasoning("CRITIQUE",
                f"Complete: {critique.is_complete}, "
                f"Safe: {critique.is_safe}, "
                f"Confidence: {critique.confidence_score:.2f}"
            )

            if edits_on:
                edit_report = {"edits": len(critique.edits), "applied": [], "failed": [],
                               "rewritten": critique.main_recommendation_wrong}
            if edits_on and critique.edits and not critique.main_recommendation_wrong:
                answer, edit_report = apply_edits(answer, critique.edits)
                edit_report["rewritten"] = False
                # 💡 An edit may cite a passage the draft did not, so the source list follows the edited answer
                citations = [c for c in self.synthesizer.citations if c.citation_id in cited(answer)]
                confidence = (confidence + critique.confidence_score) / 2
                self._log_reasoning("CRITIQUE",
                    f"Applied {len(edit_report['applied'])} of {edit_report['edits']} edits")

            # Retry if critique suggests improvement needed
            if self.self_critic.should_retry(critique):
                self._log_reasoning("CRITIQUE", "Answer needs improvement, retrying...")

                # Get improvement guidance
                guidance = self.self_critic.get_improvement_guidance(critique)

                improved_plan = None
                # ⚠️ Under critic edits the rewrite of a wrong recommendation does not re-plan either: it works
                # from the first round's passages, and only the gaps are searched, after the retry
                if self.config.features.replan and not edits_on:
                    # Replan with feedback
                    improved_plan = self.task_planner.replan(plan, guidance)

                    # Re-execute new tasks
                    for task in improved_plan.sub_tasks:
                        if task.status == TaskStatus.PENDING:
                            # ⚠️ The retry must keep the scope, or re-planned tasks would reach other sites
                            self.retriever.execute_task(task, scope)
                else:
                    # ⚠️ Re-plan passages carried 6 of 11 judged table and attribution errors and added 8-20 searches
                    # per case; without them the rewrite works from the first round's passages, the feedback and
                    # its own draft
                    self._log_reasoning("CRITIQUE", "Rewriting from the first round's passages"
                                        + (" (main recommendation wrong)" if edits_on else " (re-plan off)"))

                # Re-synthesize — use same path as first attempt based on query type
                # ⚠️ The new passages are added to the first round's, not swapped in, so a rewrite asked to fill a
                # gap keeps what the first draft was built on. Reranking old and new together would not do: the
                # reranker scores against the question, not the gap, so the first round's passages would win again.
                # The writer also gets the feedback and its draft; before, only the search planner saw the feedback
                if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
                    if improved_plan is not None:
                        chunks_by_topic = self._merge_topics(
                            chunks_by_topic, self._collect_multi_topic_chunks(improved_plan, site=rerank_site)
                        )
                    options = writer_options(sum(len(chunks) for chunks in chunks_by_topic.values()))
                    answer, citations, confidence, context_passages = self.synthesizer.synthesize_multi_topic(
                        processed_query, analysis, chunks_by_topic, feedback=guidance, draft=answer, **options
                    )
                else:
                    if improved_plan is not None:
                        all_chunks = self._merge_chunks(all_chunks, self._collect_all_chunks(improved_plan))
                    options = writer_options(len(all_chunks))
                    answer, citations, confidence, context_passages = self.synthesizer.synthesize(
                        processed_query, analysis, improved_plan or plan, all_chunks, feedback=guidance, draft=answer,
                        **options
                    )

                # Update confidence with critique score
                confidence = (confidence + critique.confidence_score) / 2
                # The recorded plan lists every search the answer drew on; a failed replan returns the plan unchanged
                if improved_plan is not None and improved_plan is not plan:
                    plan = replace(improved_plan, sub_tasks=plan.sub_tasks + improved_plan.sub_tasks)

            if edits_on:
                multi = analysis.intent == QueryIntent.COMPARISON or analysis.is_complex
                first_chunks = [c for chunks in chunks_by_topic.values() for c in chunks] if multi else all_chunks
                answer, context_passages, plan, added = self._search_gaps(
                    processed_query, plan, scope, critique.gaps, answer, first_chunks, context_passages, edit_report)
                edit_report["second"] = None
                # ⚠️ New text, from a gap search or a rewrite, is checked once more; that pass only edits, so the
                # loop ends there
                if added or critique.main_recommendation_wrong:
                    second = self.self_critic.critique(processed_query, answer, analysis, passages=context_passages)
                    answer, edit_report["second"] = apply_edits(answer, second.edits)
                    self._log_reasoning("CRITIQUE", f"Second pass applied {len(edit_report['second']['applied'])} "
                                                    f"of {edit_report['second']['edits']} edits")
                # 💡 The source list follows the final answer, which may cite a passage the draft did not
                citations = [c for c in self.synthesizer.citations if c.citation_id in cited(answer)]

        # ========== Step 6b: Citation Check ==========
        # After any retry, so the final answer is the one checked
        citation_check = None
        if self.config.features.citation_check:
            self._log_reasoning("CHECK", "Checking each citation against its passage")
            answer_before = answer
            checked, citation_check = self.citation_checker.check(answer, context_passages)
            # 💡 Kept so that a change to the check can be replayed on the same answer instead of a new run
            citation_check["answer_before"] = answer_before
            # ⚠️ In a sample of 50 removals, 16 were wrong and 29 were sentences the passage only partly supports
            # (docs/rag_improvement_plan_v2.md, 0.3); flag-only leaves the answer as written and lists them instead
            citation_check["flag_only"] = self.config.features.citation_flag_only
            if not citation_check["flag_only"]:
                answer = checked
                # Only citations still in the answer stay in the source list
                remaining = cited(answer)
                citations = [c for c in citations if c.citation_id in remaining]
            self._log_reasoning("CHECK",
                f"{'Flagged' if citation_check['flag_only'] else 'Removed'} "
                f"{citation_check['removed']} of {citation_check['citations']} citations, "
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
        if critique is not None:
            response.metadata["first_draft"] = first_draft
        if edit_report is not None:
            response.metadata["critic_edits"] = edit_report

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
