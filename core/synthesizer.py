"""
Response Synthesizer: Generates answers from retrieved chunks with citations.
"""
from typing import List, Dict, Optional

from .models import (
    RetrievedChunk, Citation, AgentResponse, TaskPlan,
    SelfCritiqueResult, QueryAnalysis
)
from .llm_client import BaseLLMClient
from .citation_check import GROUP, cited, split_references
from .query_analyzer import as_bool
from config import AgentConfig, Config

# The parts of a complete answer, in the order of the workbook's Completeness anchor ("covers indication, targets,
# dose/fractionation, technique, key OAR constraints, systemic therapy and follow-up where relevant").
# 💡 One list for the writer and the critique, so the headings written and the parts checked cannot drift apart
PLAN_PARTS = ("Indication", "Target volumes", "Dose and fractionation", "Technique", "Organ-at-risk constraints",
              "Systemic therapy", "Follow-up")


class ResponseSynthesizer:
    """Synthesizes final answers from retrieved information."""

    # ⚠️ The answer covers every part of a plan, which may include parts the question does not ask about, so
    # guidelines 1 and 2 ask for brevity within each part, not for leaving tangents out
    SYSTEM_PROMPT = """You are a radiation oncology clinical assistant.
Your role is to provide accurate, guideline-based answers to clinical questions.

Guidelines for your responses:
1. **Answer the specific question asked under the part of the answer it belongs to**
2. Be concise within each part - only include what this patient needs
3. Always cite your sources using [1], [2], etc.
4. Use standard medical terminology
5. If information is incomplete or uncertain, clearly state this
6. Never make up information - only use what's provided in the context
7. Highlight any safety-critical information
8. For specific clinical scenarios or rare conditions, focus on the exact scenario mentioned"""

    def __init__(self, llm_client: BaseLLMClient, config: Config):
        self.llm = llm_client
        self.config = config
        self.requests = []

    def _generate(self, prompt: str) -> str:
        self.requests.append({"prompt": prompt, "system_prompt": self.SYSTEM_PROMPT})
        return self.llm.generate(prompt, self.SYSTEM_PROMPT)

    def _format_context(
        self, chunks: List[Dict], start: int = 1
    ) -> tuple[str, List[Citation], List[Dict]]:
        """Format retrieved chunks into context string, citation list and full context passages."""
        context_parts = []
        citations = []
        passages = []

        for i, chunk in enumerate(chunks):
            # 💡 The only place numbers are assigned: the [n] label the LLM reads and the
            # Citation object always share one value, so they cannot drift apart.
            citation_id = start + i
            text = chunk.get("text", "")
            score = chunk.get("similarity_score", 0) or chunk.get("relevance_score", 0)

            source_type = chunk.get("source_type", "review_notes")
            doc_title = chunk.get("document_title", "")
            path = (
                doc_title if source_type == "clinical_trial" and doc_title
                else chunk.get("path", "Unknown source")
            )

            # Add to context
            context_parts.append(f"[{citation_id}] Source: {path}\n{text}\n")

            # Create citation object
            citations.append(Citation(
                citation_id=citation_id,
                source_path=path,
                chunk_text=text[:500],  # Truncate for storage
                relevance_score=float(score) if score else 0.0,
                faiss_id=chunk.get("faiss_id", -1),
                source_url=chunk.get("s3_location", ""),
                document_title=chunk.get("document_title", ""),
                source_type=source_type
            ))
            # Full text, unlike chunk_text, so evaluators see exactly what the LLM saw
            # Category lets a check confirm each passage came from the selected site's evidence
            passages.append({"citation_id": citation_id, "source_path": path, "text": text,
                             "source_type": chunk.get("source_type", "review_notes"),
                             "category": chunk.get("category", "")})

        context = "\n---\n".join(context_parts)
        return context, citations, passages

    def _extract_citations_used(self, answer: str) -> List[int]:
        """Extract citation numbers used in the answer, including [4, 5] and [1-3]."""
        return list(cited(answer))

    def _template_request(self) -> str:
        """The answer template: the seven parts of a plan as headings, in order (docs/rag_improvement_plan_v2.md, E1)."""
        # ⚠️ A part no passage covers is supplied as usual practice, marked and uncited: the judge scores a part
        # named as missing as missing, but a correct statement absent from the passages is no hallucination to it.
        # Uncited, it cannot make a citation fail. A detail the case leaves out is assumed out loud, not asserted:
        # the judge flags "45 Gy in 25 fractions" for a case that gives 45 Gy, and declining cost GU-034 its dose.
        # The cap per heading keeps seven parts from growing the answer the way rewrites did (+71% median)
        return f"""
Write the answer under these headings, in this order: {"; ".join(PLAN_PARTS)}.
- Answer what the question asks under the heading it belongs to.
- Under each heading give the recommendation for this patient, citing the passages that support it.
- If no passage covers a heading, give the usual practice without a citation and mark it \
"(not in the provided evidence)".
- If the answer depends on a case detail that is not given, state the assumption, answer under it, and say what to \
confirm.
- If a heading does not apply to this patient or this question, write "Not applicable" and the reason in a few words.
- Write at most three sentences or bullet points under each heading.
"""

    def _revision_request(self, feedback: Optional[str], draft: Optional[str]) -> str:
        """What a rewrite after self-critique is asked to change; empty for a first draft, whose prompt stays as it was."""
        if not feedback:
            return ""
        # ⚠️ The draft's citations were never checked (the citation check removes most of them) and their numbers
        # belonged to the earlier passage list, so they are dropped and the rewrite cites the passages anew
        body, _ = split_references(draft or "")
        body = GROUP.sub("", body).strip()
        # ⚠️ Feedback such as "add OAR constraints" invites invented numbers, so a gap the passages cannot fill is
        # named instead
        return f"""
A reviewer read an earlier draft of this answer and asked for changes.
Reviewer feedback: {feedback}

Earlier draft, without its citations:
{body}

Write a new answer that keeps what the earlier draft got right and addresses the feedback, citing only the passages
above by their numbers. If the feedback asks for something the passages do not contain, say that it is missing
instead of supplying it.
"""

    def synthesize(
        self,
        query: str,
        analysis: QueryAnalysis,
        plan: TaskPlan,
        all_chunks: List[Dict],
        feedback: Optional[str] = None,
        draft: Optional[str] = None,
    ) -> tuple[str, List[Citation], float, List[Dict]]:
        """Synthesize an answer from retrieved chunks; feedback and draft are given on a rewrite after self-critique."""

        if not all_chunks:
            return (
                "I could not find relevant information in the knowledge base to answer "
                "your question. Please try rephrasing your query or ask about a different topic.",
                [],
                0.0,
                []
            )

        # Format context and prepare citations
        context, all_citations, passages = self._format_context(all_chunks)
        revision = self._revision_request(feedback, draft)
        template = self._template_request()

        # Build synthesis prompt
        # ⚠️ Item 3 once asked for 3-5 sentences, which cannot cover seven parts
        synthesis_prompt = f"""Based on the following retrieved information, answer the clinical question.

Question: {query}

Query Type: {analysis.intent.value}
{f"Disease Category: {analysis.disease_category}" if analysis.disease_category else ""}
{f"Treatment Modality: {analysis.treatment_modality}" if analysis.treatment_modality else ""}

Retrieved Information:
{context}

Instructions:
1. **Answer the EXACT question asked - focus on the specific scenario or condition mentioned**
2. Prioritize the most directly relevant information from the sources
3. Be concise within each heading
4. Cite sources using [1], [2], etc. (use separate brackets for each citation)
5. Only include general background information if the specific question is fully addressed first
6. If key information is missing from the sources, state what's missing
7. Highlight any important safety considerations
{template}{revision}
Answer:"""

        # Generate response
        answer = self._generate(synthesis_prompt)

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

        return answer, used_citations, confidence, passages

    def synthesize_multi_topic(
        self,
        query: str,
        analysis: QueryAnalysis,
        chunks_by_topic: Dict[str, List[Dict]],
        feedback: Optional[str] = None,
        draft: Optional[str] = None,
    ) -> tuple[str, List[Citation], float, List[Dict]]:
        """Synthesize an answer from multiple retrieval results; feedback and draft as in synthesize."""

        # Format context for each topic
        context_parts = []
        all_citations = []
        all_passages = []
        citation_counter = 1

        # A topic whose search found nothing gets no heading: an empty group tells the writer nothing
        topics = [(topic, chunks) for topic, chunks in chunks_by_topic.items() if chunks]
        for group, (topic, chunks) in enumerate(topics, start=1):
            # ⚠️ Number while formatting. Renumbering Citation objects afterwards left every
            # topic's text labelled [1]..[k], so a cited [2] could point at the wrong topic.
            topic_context, topic_citations, topic_passages = self._format_context(
                chunks, start=citation_counter
            )
            citation_counter += len(topic_citations)

            # ⚠️ The heading numbers the group and never shows the topic. The topic is the planner's search text,
            # which can hold a guessed dose or patient fact ("45 Gy 25 fractions", "clinically N0"), and the writer
            # read a heading as evidence. Evaluators still get the topic on each passage
            context_parts.append(f"### Evidence group {group} ###\n{topic_context}")
            all_citations.extend(topic_citations)
            all_passages.extend({**p, "topic": topic, "group": group} for p in topic_passages)

        full_context = "\n\n".join(context_parts)
        revision = self._revision_request(feedback, draft)
        template = self._template_request()

        synthesis_prompt = f"""
        Answer the following clinical question using the grouped information below.

        Question: {query}

        Retrieved Information:
        {full_context}

        Instructions:
        1. Directly answer the question using ALL relevant subtopics
        2. Integrate information across subtopics where needed
        3. Be concise and clinically focused
        4. Use bullet points or tables if it improves clarity
        5. Cite sources using [1], [2], [3] format (separate brackets for each citation), never [1, 2, 3]
        6. Preserve distinctions between subtopics when relevant
        7. Highlight key differences or relationships if the question involves comparison
        8. Note any guideline recommendations or safety considerations
{template}{revision}
        Answer:
        """

        answer = self._generate(synthesis_prompt)

        # Calculate confidence
        citations_used = self._extract_citations_used(answer)
        used_citations = [c for c in all_citations if c.citation_id in citations_used]

        if used_citations:
            confidence = sum(c.relevance_score for c in used_citations) / len(used_citations)
        else:
            confidence = 0.3

        return answer, used_citations, confidence, all_passages


class SelfCritic:
    """Evaluates generated answers for completeness and safety.

    Whether citations support the answer is left to core/citation_check.py, which reads each cited passage in
    full; this step used to judge it from the first 200 characters of each passage, about a fifth of one.
    Completeness is a checklist of what the question asks for, each item judged on its own: asked only whether the
    answer was complete, the model called 36 of 40 answers complete, 21 of which the judge scored 2 of 5 or lower.
    """

    CRITIQUE_PROMPT = """You are a radiation oncology expert reviewer.
Evaluate the following answer for completeness and safety."""

    # ⚠️ Checking only what the question asks passed answers the judge marked down: 0 of 80 checklists held follow-up
    # and 14 held OAR constraints, the two parts the judge most often found missing. The checklist adds the parts of
    # a plan too, and "not applicable" keeps a part that does not apply from sending an answer back for nothing
    ITEMS_RULE = ("List each thing the question explicitly asks for, in the question's own words. Then add each "
                  f"of these parts of a complete answer that the list does not already cover: "
                  f"{'; '.join(PLAN_PARTS)}. Leave out requests to cite sources; citations are checked separately.")
    # ⚠️ Counting "the provided evidence does not cover it" as addressed marked every item addressed, and
    # completeness fell (docs/rag_improvement_plan_v2.md, Part 1)
    ADDRESSED_RULE = ("Mark an item addressed only if the answer gives a specific recommendation for this patient. "
                      "Saying that the sources do not cover it, giving general background, or answering for another "
                      "condition or setting does not count. A part the answer marks not applicable counts as "
                      "addressed if it truly does not apply to this patient; if it does apply, it is not addressed.")

    def __init__(self, llm_client: BaseLLMClient, config: Config):
        self.llm = llm_client
        self.config = config

    def critique(self, query: str, answer: str, analysis: QueryAnalysis) -> SelfCritiqueResult:
        """Perform self-critique on the generated answer."""
        # ⚠️ "Cite the supporting evidence" stays off the checklist: a rewrite told to name trials would invent them,
        # and naming a trial, study or year is where most judged hallucinations were.
        # ⚠️ No example items either: given "for example ... follow-up", the model listed follow-up as missing in 23
        # of 40 answers although no question asked for it. The parts of a plan are listed on purpose, with "not
        # applicable" as the way out
        critique_prompt = f"""Evaluate this radiation oncology answer:

Question: {query}
Query Type: {analysis.intent.value}

Answer:
{answer}

1. {self.ITEMS_RULE}
2. {self.ADDRESSED_RULE}
3. Note any recommendation that could harm this patient if followed.

Respond in JSON format:
{{
    "checklist": [{{"item": "<what the question asks for>", "addressed": <true or false>, \
"gap": "<what is missing, if not addressed>"}}],
    "is_safe": <true if no potentially harmful recommendations>,
    "concerns": ["<list of safety concerns>"],
    "confidence_score": <0.0-1.0 overall confidence>,
    "suggestions": ["<suggestions for improvement>"]
}}"""

        try:
            result = self.llm.generate_structured(
                critique_prompt,
                output_schema={
                    "checklist": "array",
                    "is_safe": "boolean",
                    "concerns": "array",
                    "confidence_score": "number",
                    "suggestions": "array"
                },
                system_prompt=self.CRITIQUE_PROMPT
            )

            checklist = [c for c in result.get("checklist") or [] if isinstance(c, dict) and str(c.get("item", "")).strip()]
            # Without a usable checklist the critique failed, and the default below applies
            if "error" not in result and checklist:
                # ⚠️ as_bool, not bool(): "addressed": "false" is truthy text. An unreadable value counts as addressed
                missing = [f"{c['item']}: {c.get('gap') or 'not addressed'}"
                           for c in checklist if not as_bool(c.get("addressed"), default=True)]
                return SelfCritiqueResult(
                    is_complete=not missing,  # from the items, not the model's overall impression
                    is_safe=as_bool(result.get("is_safe"), default=True),
                    missing_aspects=missing,
                    concerns=result.get("concerns", []),
                    confidence_score=result.get("confidence_score", 0.5),
                    suggestions=result.get("suggestions", []),
                    checklist=checklist
                )
        except Exception as e:
            print(f"Self-critique failed: {e}")

        # Default result if critique fails
        return SelfCritiqueResult(
            is_complete=True,
            is_safe=True,
            confidence_score=0.5
        )

    def should_retry(self, critique: SelfCritiqueResult) -> bool:
        """Determine if the answer should be regenerated."""
        # Retry if not safe, confidence is very low, or anything the question asks for is missing
        if not critique.is_safe:
            return True
        if critique.confidence_score < 0.4:
            return True
        # One missing item is enough: requiring more than two retried 3 of 40 answers
        return not critique.is_complete

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
