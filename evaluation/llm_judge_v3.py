"""
LLM Judge v3 — Multi-Agent Evaluation Pipeline
================================================
5 specialized judge agents:
  1. Retrieval Adequacy  — are retrieved chunks sufficient?
  2. Abstention Check    — should the system have answered at all?
  3. Faithfulness        — is the answer grounded in retrieved docs?
  4. Correctness:
       Batch eval: BOTH versions scored
         correctness_a — compare answer vs expected_answer
         correctness_b — RAG proxy (conclusion aligned with docs?)
       Chatbot: only correctness_b (no expected_answer available)
  5. Answer Quality      — clarity, completeness, clinical usefulness

Batch eval (run_evaluation):
  Runs all judges, saves both correctness scores to CSV
  No composite score (calibration not done yet)

Chatbot (evaluate_single_query):
  Runs Mode B only (no expected_answer)
  Loads calibration_map.json + weights.json from calibration/ folder
  Applies Isotonic calibration: lookup table {raw_score: calibrated_value}
  Computes composite score

Calibration:
  Run calibration/calibration.py first to generate:
    calibration/calibration_map.json  — per-criteria lookup table
    calibration/weights.json          — normalized Spearman weights
  llm_judge_v3 will raise FileNotFoundError if these are missing.
"""

import csv
import json
import os
import re
import sys
import time
import argparse
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import boto3
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from config import default_config, create_boto3_session

EXCEL_FILE       = "KP_test_questions_REVISED.xlsx"
OUTPUT_DIR       = "eval_results"
SKIP_FIRST_SHEET = True
JUDGE_MODEL_ID   = "us.amazon.nova-pro-v1:0"
DELAY_BETWEEN_Q  = 2.0
CALIBRATION_DIR  = os.path.join(_PROJECT_ROOT, "calibration")

ABSTENTION_PHRASES = [
    "i don't have", "i cannot find", "not enough information",
    "insufficient evidence", "unable to answer", "no relevant information",
    "cannot provide", "not available in", "no information available",
]

# Batch eval CSV columns
CSV_COLUMNS = [
    "disease_site", "topic", "question",
    "expected_answer", "source_document",
    "rag_answer",
    "retrieval_adequacy_score",
    "should_abstain", "did_abstain",
    "faithfulness_score",
    "correctness_a_score",          # with expected_answer
    "correctness_b_score",          # RAG proxy
    "answer_quality_score",
]


# Calibration loading
def _load_calibration(cal_dir: str = CALIBRATION_DIR) -> tuple:
    """
    Load calibration_map.json and weights.json from calibration/.
    Raises FileNotFoundError if either file is missing — run calibration/calibration.py first.
    """
    for fname in ["calibration_map.json", "weights.json"]:
        path = os.path.join(cal_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing {path} — run calibration/calibration.py first"
            )
    with open(os.path.join(cal_dir, "calibration_map.json")) as f:
        calibration_map = json.load(f)
    with open(os.path.join(cal_dir, "weights.json")) as f:
        weights = json.load(f)
    print(f"  Calibration loaded from {cal_dir}/")
    return calibration_map, weights


# Lazy-loaded — not executed at import time.
# Loaded on first call to _calibrate() or _get_weights().
# This allows run_evaluation() (batch eval) to run without calibration files present.
_CALIBRATION_MAP: Optional[dict] = None
_WEIGHTS_B: Optional[dict] = None


def _get_calibration() -> tuple:
    global _CALIBRATION_MAP, _WEIGHTS_B
    if _CALIBRATION_MAP is None:
        _CALIBRATION_MAP, _WEIGHTS_B = _load_calibration()
    return _CALIBRATION_MAP, _WEIGHTS_B


def _calibrate(criteria: str, raw_score: float) -> float:
    """Apply Isotonic calibration: lookup table {raw_score: calibrated_value}."""
    calibration_map, _ = _get_calibration()
    lookup = calibration_map.get(criteria)
    if lookup is None:
        return float(raw_score)   # no calibration for this criteria (e.g. faithfulness, retrieval)
    key = str(int(raw_score))
    return float(lookup.get(key, raw_score))


def _get_weights() -> dict:
    _, weights = _get_calibration()
    return weights


# Module-level singleton — initialized once on import, reused for every query
_judge_singleton: Optional["LLMJudgeV3"] = None

def _get_judge() -> "LLMJudgeV3":
    global _judge_singleton
    if _judge_singleton is None:
        _judge_singleton = LLMJudgeV3()
        print("  LLMJudgeV3 singleton initialized.")
    return _judge_singleton


# Dataclasses
@dataclass
class RetrievalAdequacyResult:
    score: int = 0
    reasoning: str = ""

@dataclass
class AbstentionResult:
    judgment: str = ""
    justification: str = ""

@dataclass
class FaithfulnessResult:
    score: int = 0
    unsupported_claims: List[str] = field(default_factory=list)
    reasoning: str = ""

@dataclass
class CorrectnessResult:
    score: int = 0
    reasoning: str = ""

@dataclass
class AnswerQualityResult:
    score: int = 0
    reasoning: str = ""

@dataclass
class JudgeV3Result:
    """Used for chatbot (Mode B only)."""
    retrieval_adequacy: RetrievalAdequacyResult = field(default_factory=RetrievalAdequacyResult)
    abstention: AbstentionResult               = field(default_factory=AbstentionResult)
    faithfulness: FaithfulnessResult           = field(default_factory=FaithfulnessResult)
    correctness_b: CorrectnessResult           = field(default_factory=CorrectnessResult)
    answer_quality: AnswerQualityResult        = field(default_factory=AnswerQualityResult)
    should_abstain: str = "No"
    did_abstain: str = "No"
    composite_score: float = 0.0
    abstention_triggered: bool = False

@dataclass
class BatchJudgeV3Result:
    """Used for batch eval (both correctness versions)."""
    retrieval_adequacy: RetrievalAdequacyResult = field(default_factory=RetrievalAdequacyResult)
    abstention: AbstentionResult               = field(default_factory=AbstentionResult)
    faithfulness: FaithfulnessResult           = field(default_factory=FaithfulnessResult)
    correctness_a: CorrectnessResult           = field(default_factory=CorrectnessResult)
    correctness_b: CorrectnessResult           = field(default_factory=CorrectnessResult)
    answer_quality: AnswerQualityResult        = field(default_factory=AnswerQualityResult)
    should_abstain: str = "No"
    did_abstain: str = "No"
    abstention_triggered: bool = False


# LLM Judge V3
class LLMJudgeV3:

    def __init__(self):
        self.session = create_boto3_session()
        self.client  = self.session.client(
            "bedrock-runtime", region_name=default_config.aws.region
        )

    # Nova Pro API call
    def _invoke(self, prompt: str, max_tokens: int = 1024) -> str:
        body = json.dumps({
            "messages": [
                {"role": "user", "content": [{"text": prompt}]}
            ],
            "system": [
                {"text": (
                    "You are an expert evaluator for a radiation oncology RAG system. "
                    "Respond with valid JSON only. No markdown, no explanation outside JSON."
                )}
            ],
            "inferenceConfig": {
                "maxTokens": max_tokens,
                "temperature": 0.1,
                "topP": 0.9,
            }
        })
        resp     = self.client.invoke_model(modelId=JUDGE_MODEL_ID, body=body)
        body_out = json.loads(resp["body"].read())
        gen      = body_out["output"]["message"]["content"][0]["text"].strip()
        if gen.startswith("```"):
            gen = re.sub(r"```(?:json)?", "", gen).strip().rstrip("`").strip()
        return gen

    def _parse(self, raw: str) -> dict:
        raw = re.sub(r"```(?:json)?", "", raw).strip()
        if not raw.startswith("{"):
            raw = raw[raw.find("{"):] if "{" in raw else raw
        if not raw.endswith("}"):
            raw = raw[:raw.rfind("}")+1] if "}" in raw else raw
        raw = re.sub(r",(\s*[}\]])", r"\1", raw)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _fmt_ctx(self, contexts: List[str]) -> str:
        parts = [f"[Context {i}]\n{ctx}" for i, ctx in enumerate(contexts, 1)]
        return "\n\n".join(parts) or "No retrieved documents."

    # Judge 1: Retrieval Adequacy
    def judge_retrieval_adequacy(
        self, question: str, contexts: List[str]
    ) -> RetrievalAdequacyResult:
        prompt = f"""You are evaluating whether retrieved documents contain sufficient information to answer a radiation oncology clinical question.

Question: {question}

Retrieved Documents:
{self._fmt_ctx(contexts)}

Score retrieval adequacy (1-5):
- 5: Documents contain specific, directly relevant information that fully addresses the question
- 4: Documents mostly relevant, minor gaps
- 3: Documents partially relevant, missing some important details
- 2: Documents marginally related, major gaps
- 1: Documents do not contain information to answer this question

Return ONLY valid JSON:
{{
  "score": <1-5>,
  "reasoning": "<one sentence>"
}}"""
        try:
            d = self._parse(self._invoke(prompt))
            return RetrievalAdequacyResult(
                score=int(d.get("score", 0)),
                reasoning=d.get("reasoning", ""),
            )
        except Exception as e:
            print(f"    [Retrieval Adequacy] failed: {e}")
            return RetrievalAdequacyResult()

    # Judge 2: Abstention
    def judge_abstention(
        self, question: str, contexts: List[str], answer: str
    ) -> AbstentionResult:
        prompt = f"""You are a safety evaluator for a radiation oncology RAG system.
Determine whether the system should have answered or abstained given the retrieved evidence.

Question: {question}

Retrieved Documents:
{self._fmt_ctx(contexts)}

Generated Answer: {answer}

Classify:
- "Appropriately Answered": retrieved evidence adequately supports the answer
- "Should Have Abstained": insufficient evidence; system should have said it cannot find relevant information
- "Partial Abstention Preferred": answer partially supported but overstated confidence

Return ONLY valid JSON:
{{
  "judgment": "<classification>",
  "justification": "<2 sentences>"
}}"""
        try:
            d = self._parse(self._invoke(prompt))
            return AbstentionResult(
                judgment=d.get("judgment", ""),
                justification=d.get("justification", ""),
            )
        except Exception as e:
            print(f"    [Abstention] failed: {e}")
            return AbstentionResult()

    # Judge 3: Faithfulness
    def judge_faithfulness(
        self, question: str, contexts: List[str], answer: str
    ) -> FaithfulnessResult:
        prompt = f"""You are evaluating whether a radiation oncology answer is faithfully grounded in the retrieved documents.
Do NOT use your external medical knowledge. Only assess whether claims in the answer are supported by the documents.

Question: {question}

Retrieved Documents:
{self._fmt_ctx(contexts)}

Generated Answer: {answer}

Score faithfulness (1-5):
- 5: Every factual claim is explicitly supported by the documents
- 4: Mostly supported, minor extrapolation acceptable
- 3: Partially supported, some claims lack document support
- 2: Significant unsupported or contradicted claims
- 1: Answer largely disconnected from retrieved documents

Return ONLY valid JSON:
{{
  "score": <1-5>,
  "unsupported_claims": ["<claim1>", "<claim2>"],
  "reasoning": "<one sentence>"
}}"""
        try:
            d = self._parse(self._invoke(prompt))
            return FaithfulnessResult(
                score=int(d.get("score", 0)),
                unsupported_claims=d.get("unsupported_claims", []),
                reasoning=d.get("reasoning", ""),
            )
        except Exception as e:
            print(f"    [Faithfulness] failed: {e}")
            return FaithfulnessResult()

    # Judge 4A: Correctness with expected_answer
    def judge_correctness_reference(
        self, question: str, answer: str, expected_answer: str
    ) -> CorrectnessResult:
        prompt = f"""You are evaluating the factual correctness of a radiation oncology answer.
Use the reference answer as ground truth. It is not a strict checklist — focus on whether key clinical facts are aligned.

Question: {question}

Reference Answer: {expected_answer}

Generated Answer: {answer}

Score correctness (1-5):
- 5: Fully aligned — key clinical facts consistent, no contradictions
- 4: Mostly aligned — minor differences in detail or emphasis
- 3: Partially aligned — some key facts present, others missing or contradicted
- 2: Poorly aligned — significant factual differences
- 1: Contradicts reference or provides completely wrong information

Return ONLY valid JSON:
{{
  "score": <1-5>,
  "reasoning": "<one sentence>"
}}"""
        try:
            d = self._parse(self._invoke(prompt))
            return CorrectnessResult(
                score=int(d.get("score", 0)),
                reasoning=d.get("reasoning", ""),
            )
        except Exception as e:
            print(f"    [Correctness-A] failed: {e}")
            return CorrectnessResult()

    # Judge 4B: RAG-based correctness proxy
    def judge_correctness_rag_proxy(
        self, question: str, answer: str, contexts: List[str]
    ) -> CorrectnessResult:
        prompt = f"""You are evaluating whether a radiation oncology answer's conclusion aligns with what the retrieved documents say.

This is NOT the same as faithfulness (which checks if the answer stays within document scope).
This checks whether the answer's core conclusion points in the same direction as the documents.

Question: {question}

Retrieved Documents:
{self._fmt_ctx(contexts)}

Generated Answer: {answer}

Example:
  Document says: "Treatment A is non-inferior to Treatment B"
  Answer says "A is similar to B" → conclusion aligned (score 5)
  Answer says "A is inferior to B" → conclusion contradicts documents (score 1)

Score conclusion alignment (1-5):
- 5: Answer's core conclusion fully consistent with what documents say
- 4: Mostly consistent, minor differences in interpretation
- 3: Partially consistent, some conclusions differ from documents
- 2: Answer's conclusion significantly differs from documents
- 1: Answer's conclusion contradicts what documents say

Return ONLY valid JSON:
{{
  "score": <1-5>,
  "reasoning": "<one sentence explaining alignment or contradiction>"
}}"""
        try:
            d = self._parse(self._invoke(prompt))
            return CorrectnessResult(
                score=int(d.get("score", 0)),
                reasoning=d.get("reasoning", ""),
            )
        except Exception as e:
            print(f"    [Correctness-B] failed: {e}")
            return CorrectnessResult()

    # Judge 5: Answer Quality
    def judge_answer_quality(
        self, question: str, answer: str
    ) -> AnswerQualityResult:
        prompt = f"""You are evaluating the quality of a radiation oncology answer for a clinical audience.
Do NOT assess factual correctness — only assess quality, structure, and completeness.

Question: {question}

Generated Answer: {answer}

Score answer quality (1-5):
- 5: Comprehensive, well-structured, clinically useful, addresses all aspects
- 4: Good coverage, minor gaps
- 3: Addresses main point but lacks important details
- 2: Incomplete or poorly structured
- 1: Does not meaningfully address the question

Return ONLY valid JSON:
{{
  "score": <1-5>,
  "reasoning": "<one sentence>"
}}"""
        try:
            d = self._parse(self._invoke(prompt))
            return AnswerQualityResult(
                score=int(d.get("score", 0)),
                reasoning=d.get("reasoning", ""),
            )
        except Exception as e:
            print(f"    [Answer Quality] failed: {e}")
            return AnswerQualityResult()

    # Abstention check
    def check_did_abstain(self, answer: str) -> str:
        prompt = f"""You are evaluating whether a radiation oncology RAG system actually abstained from answering.

Generated Answer: {answer}

Did the system abstain? An abstention means the system explicitly said it could not find
sufficient information to answer — e.g. saying the information was not in the sources,
or that it cannot provide a reliable answer based on available evidence.

Note: If the system provided a speculative or general answer while noting uncertainty,
that is NOT a true abstention.

Classify:
- "Yes": the system clearly stated it could not answer from available information
- "No": the system provided an answer (even if hedged or uncertain)

Return ONLY valid JSON:
{{
  "did_abstain": "<Yes or No>",
  "reasoning": "<one sentence>"
}}"""
        try:
            d = self._parse(self._invoke(prompt, max_tokens=256))
            return d.get("did_abstain", "No")
        except Exception:
            return "Yes" if any(p in answer.lower() for p in ABSTENTION_PHRASES) else "No"

    # Composite (chatbot only, Mode B)
    def compute_composite(
        self,
        retrieval_adequacy: RetrievalAdequacyResult,
        abstention: AbstentionResult,
        faithfulness: FaithfulnessResult,
        correctness_b: CorrectnessResult,
        answer_quality: AnswerQualityResult,
    ) -> tuple:
        if abstention.judgment == "Should Have Abstained":
            return 0.0, True

        raw_scores = {
            "retrieval_adequacy": retrieval_adequacy.score,
            "faithfulness":       faithfulness.score,
            "correctness":        correctness_b.score,
            "answer_quality":     answer_quality.score,
        }

        # Apply Isotonic calibration: lookup table per criteria
        scores = {k: _calibrate(k, v) for k, v in raw_scores.items()}

        weights   = _get_weights()
        composite = sum(weights[k] * scores[k] for k in weights)
        return round(composite, 2), False

    # Batch eval: run all judges including both correctness
    def evaluate_batch(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        expected_answer: str,
    ) -> BatchJudgeV3Result:
        result = BatchJudgeV3Result()

        result.retrieval_adequacy = self.judge_retrieval_adequacy(question, contexts)
        result.abstention         = self.judge_abstention(question, contexts, answer)
        result.should_abstain     = (
            "Yes" if result.abstention.judgment == "Should Have Abstained" else "No"
        )
        result.did_abstain = self.check_did_abstain(answer)

        result.faithfulness   = self.judge_faithfulness(question, contexts, answer)
        result.correctness_a  = self.judge_correctness_reference(
            question, answer, expected_answer
        )
        result.correctness_b  = self.judge_correctness_rag_proxy(
            question, answer, contexts
        )
        result.answer_quality = self.judge_answer_quality(question, answer)
        result.abstention_triggered = (
            result.abstention.judgment == "Should Have Abstained"
        )

        return result

    # Chatbot: Mode B only + composite
    def evaluate_chatbot(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> JudgeV3Result:
        result = JudgeV3Result()

        result.retrieval_adequacy = self.judge_retrieval_adequacy(question, contexts)
        result.abstention         = self.judge_abstention(question, contexts, answer)
        result.should_abstain     = (
            "Yes" if result.abstention.judgment == "Should Have Abstained" else "No"
        )
        result.did_abstain = self.check_did_abstain(answer)

        result.faithfulness  = self.judge_faithfulness(question, contexts, answer)
        result.correctness_b = self.judge_correctness_rag_proxy(
            question, answer, contexts
        )
        result.answer_quality = self.judge_answer_quality(question, answer)

        result.composite_score, result.abstention_triggered = self.compute_composite(
            result.retrieval_adequacy, result.abstention,
            result.faithfulness, result.correctness_b, result.answer_quality,
        )
        return result


# Helpers
def load_questions_from_sheet(ws) -> List[dict]:
    headers   = [cell.value for cell in ws[1]]
    questions = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not any(row):
            continue
        q = dict(zip(headers, row))
        questions.append({
            "disease_site":    q.get("Disease Site", "") or "",
            "topic":           q.get("Topic", "") or "",
            "question":        q.get("Question", "") or "",
            "expected_answer": q.get("Expected Answer", "") or "",
            "source_document": q.get("Source Document", "") or "",
        })
    return [q for q in questions if q["question"].strip()]


def load_completed_questions(csv_path: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, "r", encoding="utf-8") as f:
        return {row["question"].strip() for row in csv.DictReader(f)}


def extract_contexts(response) -> List[str]:
    contexts = []
    if response.task_plan:
        for task in response.task_plan.sub_tasks:
            for chunk in task.retrieved_chunks:
                text = chunk.get("text", "")
                if text:
                    contexts.append(text)
    if not contexts:
        for c in response.citations:
            if c.chunk_text:
                contexts.append(c.chunk_text)
    return contexts


# Terminal display
def print_batch_result(
    i: int, total: int, question: str, answer: str, result: BatchJudgeV3Result
):
    print(f"\n  ── RAG Answer ──────────────────────────────────────────")
    print(f"  {answer}")
    print(f"\n  ── Judge Scores ────────────────────────────────────────")
    print(f"  Retrieval Adequacy ({result.retrieval_adequacy.score}/5)")
    print(f"  Should Abstain: {result.should_abstain} | Did Abstain: {result.did_abstain}")
    unsupported = (f" | Unsupported: {result.faithfulness.unsupported_claims}"
                   if result.faithfulness.unsupported_claims else "")
    print(f"  Faithfulness       ({result.faithfulness.score}/5){unsupported}")
    print(f"  Correctness-A      ({result.correctness_a.score}/5): {result.correctness_a.reasoning}")
    print(f"  Correctness-B      ({result.correctness_b.score}/5): {result.correctness_b.reasoning}")
    print(f"  Answer Quality     ({result.answer_quality.score}/5): {result.answer_quality.reasoning}")
    if result.abstention_triggered:
        print(f"  ⚠️  ABSTENTION TRIGGERED — composite = 0")
    print(f"  ───────────────────────────────────────────────────────\n")


# Batch evaluation loop
def run_evaluation(
    sheet_names: Optional[List[str]] = None,
    output_dir: str = OUTPUT_DIR,
):
    from core.agent import AgenticRAGEngine
    import openpyxl

    os.makedirs(output_dir, exist_ok=True)

    wb         = openpyxl.load_workbook(EXCEL_FILE)
    all_sheets = wb.sheetnames[1:] if SKIP_FIRST_SHEET else wb.sheetnames
    if sheet_names:
        all_sheets = [s for s in all_sheets if s in sheet_names]

    print(f"\nSheets: {all_sheets}\nOutput: {output_dir}/\n")

    print("Initializing RAG agent...")
    agent = AgenticRAGEngine(config=default_config, enable_judge=False)
    judge = LLMJudgeV3()
    print("Agent ready.\n")

    total_done = total_skip = total_error = 0

    for sheet_name in all_sheets:
        ws        = wb[sheet_name]
        questions = load_questions_from_sheet(ws)
        csv_path  = os.path.join(output_dir, f"{sheet_name}.csv")
        completed = load_completed_questions(csv_path)

        print(f"{'='*70}")
        print(f"Sheet: {sheet_name} ({len(questions)} questions, {len(completed)} done)")
        print(f"{'='*70}")

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()

            for i, q in enumerate(questions, 1):
                question = q["question"].strip()

                if question in completed:
                    print(f"  [{i:2d}/{len(questions)}] SKIP: {question[:70]}")
                    total_skip += 1
                    continue

                try:
                    print(f"\n  [{i:2d}/{len(questions)}] {question}")
                    response = agent.query(question)
                    answer   = response.answer
                    contexts = extract_contexts(response)
                except Exception as e:
                    print(f"  [{i:2d}/{len(questions)}] RAG FAILED: {e}")
                    total_error += 1
                    continue

                try:
                    result = judge.evaluate_batch(
                        question        = question,
                        answer          = answer,
                        contexts        = contexts,
                        expected_answer = q["expected_answer"],
                    )
                except Exception as e:
                    print(f"  [{i:2d}/{len(questions)}] JUDGE FAILED: {e}")
                    total_error += 1
                    continue

                print_batch_result(i, len(questions), question, answer, result)

                writer.writerow({
                    "disease_site":             q["disease_site"],
                    "topic":                    q["topic"],
                    "question":                 question,
                    "expected_answer":          q["expected_answer"],
                    "source_document":          q["source_document"],
                    "rag_answer":               answer,
                    "retrieval_adequacy_score": result.retrieval_adequacy.score,
                    "should_abstain":           result.should_abstain,
                    "did_abstain":              result.did_abstain,
                    "faithfulness_score":       result.faithfulness.score,
                    "correctness_a_score":      result.correctness_a.score,
                    "correctness_b_score":      result.correctness_b.score,
                    "answer_quality_score":     result.answer_quality.score,
                })
                f.flush()
                total_done += 1
                time.sleep(DELAY_BETWEEN_Q)

        print(f"  → Saved to {csv_path}\n")

    print(f"\n{'='*60}")
    print(f"Done: {total_done} | Skipped: {total_skip} | Errors: {total_error}")
    print(f"CSVs in: {os.path.abspath(output_dir)}/")
    print(f"{'='*60}\n")


# evaluate_single_query: called from core/agent.py
def evaluate_single_query(
    question: str,
    answer: str,
    contexts: List[str],
) -> Dict[str, Any]:
    """
    Chatbot evaluation — Mode B only, with Isotonic calibration.
    Returns dict for response.metadata["judge_evaluations"].
    Requires calibration/calibration_map.json and calibration/weights.json to exist.
    """
    judge  = _get_judge()
    result = judge.evaluate_chatbot(
        question=question, answer=answer, contexts=contexts
    )

    return {
        "retrieval_adequacy":           result.retrieval_adequacy.score,
        "retrieval_adequacy_reasoning": result.retrieval_adequacy.reasoning,
        "should_abstain":               result.should_abstain,
        "did_abstain":                  result.did_abstain,
        "faithfulness":                 result.faithfulness.score,
        "faithfulness_reasoning":       result.faithfulness.reasoning,
        "correctness":                  result.correctness_b.score,
        "correctness_reasoning":        result.correctness_b.reasoning,
        "answer_quality":               result.answer_quality.score,
        "answer_quality_reasoning":     result.answer_quality.reasoning,
        "composite_score":              result.composite_score,
        "abstention_triggered":         result.abstention_triggered,
        "overall_score":                result.composite_score,
    }


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Judge v3 — Batch Evaluation")
    parser.add_argument("--sheet",      nargs="+", default=None,
                        help="Sheet(s) to evaluate, e.g. --sheet Breast CNS")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()
    run_evaluation(sheet_names=args.sheet, output_dir=args.output_dir)
