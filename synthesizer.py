"""
Response Synthesizer: Generates answers from retrieved chunks with citations.
"""
from typing import List, Dict, Optional
import re

from models import (
    RetrievedChunk, Citation, AgentResponse, TaskPlan,
    SelfCritiqueResult, QueryAnalysis
)
from llm_client import BaseLLMClient
from config import AgentConfig


class ResponseSynthesizer:
    """Synthesizes final answers from retrieved information."""
    
    SYSTEM_PROMPT = """You are a radiation oncology clinical assistant. 
Your role is to provide accurate, guideline-based answers to clinical questions.

Guidelines for your responses:
1. Be concise but comprehensive
2. Always cite your sources using [1], [2], etc.
3. Use standard medical terminology
4. If information is incomplete or uncertain, clearly state this
5. Never make up information - only use what's provided in the context
6. Highlight any safety-critical information
7. Structure complex answers with clear sections"""

    def __init__(self, llm_client: BaseLLMClient, config: AgentConfig):
        self.llm = llm_client
        self.config = config
    
    def _format_context(self, chunks: List[Dict]) -> tuple[str, List[Citation]]:
        """Format retrieved chunks into context string and citation list."""
        context_parts = []
        citations = []
        
        for i, chunk in enumerate(chunks):
            citation_id = i + 1
            text = chunk.get("text", "")
            path = chunk.get("path", "Unknown source")
            score = chunk.get("similarity_score", 0) or chunk.get("relevance_score", 0)
            
            # Add to context
            context_parts.append(f"[{citation_id}] Source: {path}\n{text}\n")
            
            # Create citation object
            citations.append(Citation(
                citation_id=citation_id,
                source_path=path,
                chunk_text=text[:500],  # Truncate for storage
                relevance_score=float(score) if score else 0.0,
                faiss_id=chunk.get("faiss_id", -1)
            ))
        
        context = "\n---\n".join(context_parts)
        return context, citations
    
    def _extract_citations_used(self, answer: str) -> List[int]:
        """Extract citation numbers used in the answer."""
        pattern = r'\[(\d+)\]'
        matches = re.findall(pattern, answer)
        return list(set(int(m) for m in matches))
    
    def synthesize(
        self,
        query: str,
        analysis: QueryAnalysis,
        plan: TaskPlan,
        all_chunks: List[Dict]
    ) -> tuple[str, List[Citation], float]:
        """Synthesize an answer from retrieved chunks."""
        
        if not all_chunks:
            return (
                "I could not find relevant information in the knowledge base to answer "
                "your question. Please try rephrasing your query or ask about a different topic.",
                [],
                0.0
            )
        
        # Format context and prepare citations
        context, all_citations = self._format_context(all_chunks)
        
        # Build synthesis prompt
        synthesis_prompt = f"""Based on the following retrieved information, answer the clinical question.

Question: {query}

Query Type: {analysis.intent.value}
{f"Disease Category: {analysis.disease_category}" if analysis.disease_category else ""}
{f"Treatment Modality: {analysis.treatment_modality}" if analysis.treatment_modality else ""}

Retrieved Information:
{context}

Instructions:
1. Synthesize the information to directly answer the question
2. Cite sources using [1], [2], etc. for each claim
3. If the retrieved information is insufficient, clearly state what's missing
4. Highlight any important safety considerations
5. Be concise but thorough

Answer:"""
        
        # Generate response
        answer = self.llm.generate(synthesis_prompt, self.SYSTEM_PROMPT)
        
        # Identify which citations were actually used
        citations_used = self._extract_citations_used(answer)
        used_citations = [c for c in all_citations if c.citation_id in citations_used]
        
        # Calculate confidence based on citation coverage and scores
        if used_citations:
            avg_score = sum(c.relevance_score for c in used_citations) / len(used_citations)
            coverage = len(used_citations) / max(len(all_citations), 1)
            confidence = (avg_score * 0.7) + (coverage * 0.3)
        else:
            confidence = 0.3  # Low confidence if no citations
        
        return answer, used_citations, confidence
    
    def synthesize_comparison(
        self,
        query: str,
        analysis: QueryAnalysis,
        chunks_by_topic: Dict[str, List[Dict]]
    ) -> tuple[str, List[Citation], float]:
        """Synthesize a comparison answer from multiple retrieval results."""
        
        # Format context for each topic
        context_parts = []
        all_citations = []
        citation_counter = 1
        
        for topic, chunks in chunks_by_topic.items():
            topic_context, topic_citations = self._format_context(chunks)
            
            # Renumber citations
            for citation in topic_citations:
                citation.citation_id = citation_counter
                citation_counter += 1
            
            context_parts.append(f"### {topic.upper()} ###\n{topic_context}")
            all_citations.extend(topic_citations)
        
        full_context = "\n\n".join(context_parts)
        
        # Build comparison synthesis prompt
        synthesis_prompt = f"""Compare and contrast the following based on the retrieved information.

Question: {query}

Retrieved Information:
{full_context}

Instructions:
1. Provide a structured comparison addressing the question
2. Use a table format if comparing multiple specific attributes
3. Cite sources using [1][2][3] format (separate brackets for each), never [1, 2, 3]
4. Highlight key differences and similarities
5. Note any guideline recommendations for choosing between options
6. Include any safety considerations

Comparison:"""
        
        answer = self.llm.generate(synthesis_prompt, self.SYSTEM_PROMPT)
        
        # Calculate confidence
        citations_used = self._extract_citations_used(answer)
        used_citations = [c for c in all_citations if c.citation_id in citations_used]
        
        if used_citations:
            confidence = sum(c.relevance_score for c in used_citations) / len(used_citations)
        else:
            confidence = 0.3
        
        return answer, used_citations, confidence


class SelfCritic:
    """Evaluates generated answers for completeness, accuracy, and safety."""
    
    CRITIQUE_PROMPT = """You are a radiation oncology expert reviewer. 
Evaluate the following answer for completeness, accuracy, and safety."""
    
    def __init__(self, llm_client: BaseLLMClient, config: AgentConfig):
        self.llm = llm_client
        self.config = config
    
    def critique(
        self,
        query: str,
        answer: str,
        citations: List[Citation],
        analysis: QueryAnalysis
    ) -> SelfCritiqueResult:
        """Perform self-critique on the generated answer."""
        
        # Format citations for review
        citation_text = "\n".join([
            f"[{c.citation_id}] {c.source_path}: {c.chunk_text[:200]}..."
            for c in citations
        ])
        
        critique_prompt = f"""Evaluate this radiation oncology answer:

Question: {query}
Query Type: {analysis.intent.value}

Answer:
{answer}

Sources used:
{citation_text}

Evaluate on these criteria and respond in JSON format:
{{
    "is_complete": <true if answer fully addresses the question>,
    "is_accurate": <true if claims are supported by citations>,
    "is_safe": <true if no potentially harmful recommendations>,
    "missing_aspects": ["<list of important aspects not addressed>"],
    "concerns": ["<list of accuracy or safety concerns>"],
    "confidence_score": <0.0-1.0 overall confidence>,
    "suggestions": ["<suggestions for improvement>"]
}}"""
        
        try:
            result = self.llm.generate_structured(
                critique_prompt,
                output_schema={
                    "is_complete": "boolean",
                    "is_accurate": "boolean", 
                    "is_safe": "boolean",
                    "missing_aspects": "array",
                    "concerns": "array",
                    "confidence_score": "number",
                    "suggestions": "array"
                },
                system_prompt=self.CRITIQUE_PROMPT
            )
            
            if "error" not in result:
                return SelfCritiqueResult(
                    is_complete=result.get("is_complete", True),
                    is_accurate=result.get("is_accurate", True),
                    is_safe=result.get("is_safe", True),
                    missing_aspects=result.get("missing_aspects", []),
                    concerns=result.get("concerns", []),
                    confidence_score=result.get("confidence_score", 0.5),
                    suggestions=result.get("suggestions", [])
                )
        except Exception as e:
            print(f"Self-critique failed: {e}")
        
        # Default result if critique fails
        return SelfCritiqueResult(
            is_complete=True,
            is_accurate=True,
            is_safe=True,
            confidence_score=0.5
        )
    
    def should_retry(self, critique: SelfCritiqueResult) -> bool:
        """Determine if the answer should be regenerated."""
        # Retry if not safe or confidence is very low
        if not critique.is_safe:
            return True
        if critique.confidence_score < 0.4:
            return True
        if not critique.is_complete and len(critique.missing_aspects) > 2:
            return True
        return False
    
    def get_improvement_guidance(self, critique: SelfCritiqueResult) -> str:
        """Generate guidance for improving the answer."""
        guidance_parts = []
        
        if not critique.is_complete:
            guidance_parts.append(
                f"Address these missing aspects: {', '.join(critique.missing_aspects)}"
            )
        
        if critique.concerns:
            guidance_parts.append(
                f"Address these concerns: {', '.join(critique.concerns)}"
            )
        
        if critique.suggestions:
            guidance_parts.append(
                f"Consider: {', '.join(critique.suggestions)}"
            )
        
        return " | ".join(guidance_parts) if guidance_parts else "Improve clarity and completeness."
