"""
Agentic RAG Engine: Main agent that orchestrates all components.
"""
import time
import uuid
from typing import List, Dict, Optional, Any
from datetime import datetime

from models import (
    AgentResponse, TaskPlan, QueryAnalysis, Citation,
    SelfCritiqueResult, TaskStatus, QueryIntent
)
from config import Config, default_config
from llm_client import create_llm_client, BaseLLMClient
from query_analyzer import QueryAnalyzer
from task_planner import TaskPlanner
from retriever import RetrieverManager, MockRetrieverManager
from synthesizer import ResponseSynthesizer, SelfCritic
from guardrails import GuardrailManager
from models import RetrievedChunk
from llm_judge import create_llm_judge, LLMJudge


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
    
    def __init__(self, config: Config = None, use_mock: bool = False):
        """
        Initialize the Agentic RAG Engine.
        
        Args:
            config: Configuration object. Uses default if not provided.
            use_mock: If True, use mock LLM and retriever for testing.
        """
        self.config = config or default_config
        self.use_mock = use_mock
        
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
        self.guardrails = GuardrailManager(self.config.guardrail)
        
        # Initialize LLM Judge (requires AWS)
        self.judge = create_llm_judge(self.config.aws)
        
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
        Each topic → rerank all sources together → top complex_rerank_top_k=5 chunks.

        Example for "Compare SBRT vs EBRT":
          Tasks: [Guidelines]SBRT, [Trial papers]SBRT, [Guidelines]EBRT, [Trial papers]EBRT
          Groups: {"SBRT prostate": 20 chunks, "EBRT prostate": 20 chunks}
          Result: {"SBRT prostate": 5 chunks, "EBRT prostate": 5 chunks} → 10 to LLM
        """
        from models import RetrievedChunk
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
                reranked_by_topic[label] = []
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

            reranked_by_topic[label] = final_chunks

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


    
    def query(self, user_query: str) -> AgentResponse:
        """
        Process a user query through the agentic RAG pipeline.
        
        Args:
            user_query: The user's clinical question.
            
        Returns:
            AgentResponse with answer, citations, confidence, and reasoning trace.
        """
        start_time = time.time()
        self.reasoning_trace = []
        
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
        
        # ========== Step 2: Query Analysis ==========
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
        self._log_reasoning("PLAN", "Creating execution plan")
        plan = self.task_planner.plan(analysis)
        
        self._log_reasoning("PLAN", 
            f"Created {len(plan.sub_tasks)} sub-tasks: {plan.reasoning}"
        )
        
        # ========== Step 4: Execute Retrieval Tasks ==========
        self._log_reasoning("RETRIEVE", "Executing retrieval tasks")
        
        for task in plan.sub_tasks:
            self._log_reasoning("RETRIEVE", 
                f"Task '{task.description}' using {task.retrieval_strategy.value}"
            )
            self.retriever.execute_task(task)
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
            self._log_reasoning("RETRIEVE", 
                f"Collected {len(all_chunks)} chunks (cross-encoder reranked)" 
            )
        
        # ========== Step 5: Response Synthesis ==========
        self._log_reasoning("SYNTHESIZE", "Generating response from retrieved information")
        
        if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
            answer, citations, confidence = self.synthesizer.synthesize_multi_topic(
                processed_query, analysis, chunks_by_topic 
            )
        else:
            answer, citations, confidence = self.synthesizer.synthesize(
                processed_query, analysis, plan, all_chunks
            )
        ### Changed
        self._log_reasoning("SYNTHESIZE", 
            f"Generated answer with {len(citations)} citations, confidence: {confidence:.2f}"
        )
        
        # ========== Step 6: Self-Critique ==========
        critique = None
        if self.config.agent.enable_self_critique:
            self._log_reasoning("CRITIQUE", "Performing self-evaluation")
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
                
                # Get improvement guidance
                guidance = self.self_critic.get_improvement_guidance(critique)
                
                # Replan with feedback
                improved_plan = self.task_planner.replan(plan, guidance)
                
                # Re-execute new tasks
                for task in improved_plan.sub_tasks:
                    if task.status == TaskStatus.PENDING:
                        self.retriever.execute_task(task)
                
                # Re-synthesize — use same path as first attempt based on query type
                if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
                    chunks_by_topic = self._collect_multi_topic_chunks(improved_plan)
                    answer, citations, confidence = self.synthesizer.synthesize_multi_topic(
                        processed_query, analysis, chunks_by_topic
                    )
                else:
                    all_chunks = self._collect_all_chunks(improved_plan)
                    answer, citations, confidence = self.synthesizer.synthesize(
                        processed_query, analysis, improved_plan, all_chunks
                    )
                
                # Update confidence with critique score
                confidence = (confidence + critique.confidence_score) / 2
                plan = improved_plan
        
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
        

        #
        if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
            total_chunks = sum(len(v) for v in chunks_by_topic.values())
        else:
            total_chunks = len(all_chunks)
        #

        response = AgentResponse(
            query=user_query,
            answer=final_answer,
            citations=citations,
            confidence_score=confidence,
            reasoning_trace=self.reasoning_trace,
            self_critique=critique,
            task_plan=plan,
            metadata={
                "analysis": {
                    "intent": analysis.intent.value,
                    "disease_category": analysis.disease_category,
                    "treatment_modality": analysis.treatment_modality,
                    "is_complex": analysis.is_complex
                },
                "retrieval": {
                    "total_chunks": total_chunks, # changed 
                    "tasks_completed": sum(
                        1 for t in plan.sub_tasks if t.status == TaskStatus.COMPLETED
                    ),
                    "tasks_total": len(plan.sub_tasks)
                }
            },
            latency_ms=latency_ms
        )
        
        # ========== Step 8: LLM Judge Evaluation ==========
        if self.judge is not None:
            self._log_reasoning("JUDGE", "Evaluating answer quality with Llama 3 70B judge")
            
            # Extract retrieved context texts for evaluation
            if analysis.intent == QueryIntent.COMPARISON or analysis.is_complex:
                retrieved_texts = []
                for topic_chunks in chunks_by_topic.values():
                    for chunk in topic_chunks[:3]:
                        retrieved_texts.append(chunk.get("text", ""))
            else:
                retrieved_texts = [
                    chunk.get("text", "")
                    for chunk in all_chunks[:5]
                ]
            
            try:
                judge_evaluations = self.judge.evaluate(
                    query=user_query,
                    answer=final_answer,
                    retrieved_contexts=retrieved_texts
                )

                judge_evaluation = judge_evaluations[0]  # Llama 3 70B only
                response.metadata["judge_evaluations"] = [
                    {
                        "model_name": judge_evaluation.model_name,
                        "accuracy": judge_evaluation.accuracy.score,
                        "accuracy_reasoning": judge_evaluation.accuracy.reasoning,
                        "completeness": judge_evaluation.completeness.score,
                        "completeness_reasoning": judge_evaluation.completeness.reasoning,
                        "retrieval_adherence": judge_evaluation.retrieval_adherence.score,
                        "retrieval_adherence_reasoning": judge_evaluation.retrieval_adherence.reasoning,
                        "overall_score": judge_evaluation.overall_score,
                        "overall_reasoning": judge_evaluation.overall_reasoning,
                        "evaluation_time_ms": judge_evaluation.evaluation_time_ms
                    }
                ]
                
                self._log_reasoning("JUDGE",
                    f"Evaluation complete — Accuracy: {judge_evaluation.accuracy.score}/5, "
                    f"Completeness: {judge_evaluation.completeness.score}/5, "
                    f"Retrieval Adherence: {judge_evaluation.retrieval_adherence.score}/5 "
                    f"(overall {judge_evaluation.overall_score:.2f}/5.00)"
                )
            except Exception as e:
                err_msg = str(e).lower()
                is_token_err = "token has expired" in err_msg or "sso" in err_msg or "refresh failed" in err_msg
                if is_token_err:
                    # Re-raise so UI can catch it and show the correct message
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