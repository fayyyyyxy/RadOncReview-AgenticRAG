"""
LLM-as-a-Judge Evaluator for RAG System.
Evaluates generated answers on multiple criteria using AWS Bedrock models.
"""
import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
import boto3

from config import AWSConfig, create_boto3_session


@dataclass
class JudgeScore:
    """Score for a single evaluation criterion."""
    criterion: str
    score: int  # 1-5 scale
    reasoning: str
    max_score: int = 5


@dataclass
class JudgeEvaluation:
    """Complete evaluation from a single judge model."""
    model_name: str
    accuracy: JudgeScore
    completeness: JudgeScore
    retrieval_adherence: JudgeScore
    overall_score: float  # Average of all scores
    overall_reasoning: str
    evaluation_time_ms: float = 0.0


class LLMJudge:
    """
    LLM-as-a-Judge evaluator that scores RAG answers using AWS Bedrock models.
    
    Evaluates on three criteria:
    1. Accuracy: How factually correct is the answer?
    2. Completeness: Does the answer fully address the query?
    3. Retrieval Adherence: Does the answer stick to retrieved contexts?
    """
    
    def __init__(self, aws_config: AWSConfig):
        """Initialize the LLM Judge with Llama 3 models."""
        self.aws_config = aws_config
        self.session = create_boto3_session(aws_config)
        self.client = self.session.client(
            "bedrock-runtime",
            region_name=aws_config.region
        )
        
        # Judge model - Llama 3 70B
        self.judge_models = [
            {
                "model_id": "meta.llama3-70b-instruct-v1:0",
                "name": "Llama 3 70B Instruct",
                "type": "llama"
            }
        ]
    
    def _create_evaluation_prompt(
        self,
        query: str,
        answer: str,
        retrieved_contexts: Optional[List[str]] = None
    ) -> str:
        """Create the evaluation prompt for the judge model."""
        context_section = ""
        if retrieved_contexts:
            context_section = "\n\n## Retrieved Contexts:\n"
            for i, ctx in enumerate(retrieved_contexts[:5], 1):
                context_section += f"\n### Context {i}:\n{ctx}\n"
        
        prompt = f"""You are an expert evaluator for a Radiation Oncology RAG system. Evaluate this answer on three criteria using a 1-5 scale.

## Original Query:
{query}

## Generated Answer:
{answer}
{context_section}

## Evaluation Criteria:

### 1. Accuracy (1-5)
How factually correct is the answer?
- 5: Completely accurate, medically sound
- 4: Mostly accurate with minor imprecisions
- 3: Partially accurate but has notable issues
- 2: Contains significant inaccuracies
- 1: Fundamentally incorrect

### 2. Completeness (1-5)
Does the answer fully address the query?
- 5: Comprehensively addresses all aspects
- 4: Addresses most aspects, minor gaps
- 3: Addresses main points but misses details
- 2: Incomplete, missing major aspects
- 1: Does not adequately address query

### 3. Retrieval Adherence (1-5)
Does the answer stick to retrieved contexts?
- 5: Fully grounded, no hallucination
- 4: Mostly grounded, minor extrapolation
- 3: Partially grounded with unsupported claims
- 2: Significant unsupported information
- 1: Largely disconnected from contexts

## Response Format:

Provide ONLY valid JSON. Keep reasoning concise (max 100 words each). Use this exact format:

{{
    "accuracy": {{"score": <1-5>, "reasoning": "<brief explanation>"}},
    "completeness": {{"score": <1-5>, "reasoning": "<brief explanation>"}},
    "retrieval_adherence": {{"score": <1-5>, "reasoning": "<brief explanation>"}},
    "overall_reasoning": "<brief summary>"
}}"""
        
        return prompt
    
    def _invoke_judge_model(
        self,
        model_config: Dict,
        prompt: str,
        temperature: float = 0.1
    ) -> str:
        """Invoke a Llama 3 judge model on Bedrock."""
        model_id = model_config["model_id"]
        
        # Llama 3 format
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert evaluator. Respond with valid JSON only. Keep explanations brief.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{{"""
        
        request_body = {
            "prompt": full_prompt,
            "max_gen_len": 1024,  # Reduced to encourage brevity
            "temperature": temperature,
            "top_p": 0.9
        }
        
        response = self.client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response["body"].read())
        generation = response_body["generation"]
        
        # Llama often starts with the opening brace, add it back
        if not generation.strip().startswith('{'):
            generation = '{' + generation
            
        return generation
    
    def _clean_json_string(self, text: str) -> str:
        """Clean and fix common JSON issues in LLM responses."""
        # Remove markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        
        # Ensure it starts and ends with braces
        text = text.strip()
        if not text.startswith('{'):
            # Try to find the first {
            start_idx = text.find('{')
            if start_idx != -1:
                text = text[start_idx:]
        
        if not text.endswith('}'):
            # Try to find the last }
            end_idx = text.rfind('}')
            if end_idx != -1:
                text = text[:end_idx + 1]
        
        # Fix common JSON issues
        # Remove trailing commas before closing braces
        text = re.sub(r',(\s*[}\]])', r'\1', text)
        
        # Fix unescaped quotes in strings (rough heuristic)
        # This is tricky, so we'll just try to parse and let it fail gracefully
        
        return text
    
    def _parse_evaluation_response(
        self,
        response_text: str,
        model_name: str
    ) -> JudgeEvaluation:
        """Parse the judge model's evaluation response."""
        try:
            # Clean the JSON
            json_text = self._clean_json_string(response_text)
            
            # Try to parse
            eval_data = json.loads(json_text)
            
            # Create JudgeScore objects
            accuracy = JudgeScore(
                criterion="Accuracy",
                score=eval_data["accuracy"]["score"],
                reasoning=eval_data["accuracy"]["reasoning"]
            )
            
            completeness = JudgeScore(
                criterion="Completeness",
                score=eval_data["completeness"]["score"],
                reasoning=eval_data["completeness"]["reasoning"]
            )
            
            retrieval_adherence = JudgeScore(
                criterion="Retrieval Adherence",
                score=eval_data["retrieval_adherence"]["score"],
                reasoning=eval_data["retrieval_adherence"]["reasoning"]
            )
            
            # Calculate overall score
            overall_score = (
                accuracy.score + completeness.score + retrieval_adherence.score
            ) / 3.0
            
            return JudgeEvaluation(
                model_name=model_name,
                accuracy=accuracy,
                completeness=completeness,
                retrieval_adherence=retrieval_adherence,
                overall_score=overall_score,
                overall_reasoning=eval_data.get("overall_reasoning", "")
            )
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # If parsing fails, try to extract scores using regex as fallback
            print(f"Warning: Failed to parse JSON from {model_name}, trying regex extraction...")
            print(f"Error: {e}")
            print(f"Raw response (first 500 chars): {response_text[:500]}")
            
            try:
                # Try to extract scores using regex
                accuracy_match = re.search(r'"accuracy":\s*{\s*"score":\s*(\d+)', response_text)
                completeness_match = re.search(r'"completeness":\s*{\s*"score":\s*(\d+)', response_text)
                adherence_match = re.search(r'"retrieval_adherence":\s*{\s*"score":\s*(\d+)', response_text)
                
                if accuracy_match and completeness_match and adherence_match:
                    acc_score = int(accuracy_match.group(1))
                    comp_score = int(completeness_match.group(1))
                    adh_score = int(adherence_match.group(1))
                    
                    return JudgeEvaluation(
                        model_name=model_name,
                        accuracy=JudgeScore("Accuracy", acc_score, "Successfully extracted from malformed JSON"),
                        completeness=JudgeScore("Completeness", comp_score, "Successfully extracted from malformed JSON"),
                        retrieval_adherence=JudgeScore("Retrieval Adherence", adh_score, "Successfully extracted from malformed JSON"),
                        overall_score=(acc_score + comp_score + adh_score) / 3.0,
                        overall_reasoning="Scores extracted via regex due to JSON parse error"
                    )
            except Exception as regex_e:
                print(f"Regex extraction also failed: {regex_e}")
            
            # If all else fails, return default
            return JudgeEvaluation(
                model_name=model_name,
                accuracy=JudgeScore("Accuracy", 0, f"Parse error: {str(e)}"),
                completeness=JudgeScore("Completeness", 0, f"Parse error: {str(e)}"),
                retrieval_adherence=JudgeScore("Retrieval Adherence", 0, f"Parse error: {str(e)}"),
                overall_score=0.0,
                overall_reasoning=f"Failed to parse evaluation: {str(e)}"
            )
    
    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved_contexts: Optional[List[str]] = None
    ) -> List[JudgeEvaluation]:
        """Evaluate the answer using all configured judge models."""
        evaluations = []
        prompt = self._create_evaluation_prompt(query, answer, retrieved_contexts)
        
        for model_config in self.judge_models:
            try:
                import time
                start_time = time.time()
                
                response_text = self._invoke_judge_model(model_config, prompt)
                evaluation = self._parse_evaluation_response(
                    response_text,
                    model_config["name"]
                )
                
                evaluation.evaluation_time_ms = (time.time() - start_time) * 1000
                evaluations.append(evaluation)
                
            except Exception as e:
                print(f"Error evaluating with {model_config['name']}: {e}")
                evaluations.append(JudgeEvaluation(
                    model_name=model_config["name"],
                    accuracy=JudgeScore("Accuracy", 0, f"Evaluation failed: {str(e)}"),
                    completeness=JudgeScore("Completeness", 0, f"Evaluation failed: {str(e)}"),
                    retrieval_adherence=JudgeScore("Retrieval Adherence", 0, f"Evaluation failed: {str(e)}"),
                    overall_score=0.0,
                    overall_reasoning=f"Evaluation failed: {str(e)}"
                ))
        
        return evaluations
    
    def print_evaluation_report(
        self,
        evaluations: List[JudgeEvaluation],
        query: str,
        answer: str
    ):
        """Print a formatted evaluation report."""
        print("\n" + "=" * 80)
        print("                    LLM JUDGE EVALUATION REPORT")
        print("=" * 80)
        
        print(f"\n📝 Query: {query}")
        print(f"\n💬 Generated Answer:\n{answer[:300]}{'...' if len(answer) > 300 else ''}")
        
        print("\n" + "-" * 80)
        print("EVALUATION RESULTS")
        print("-" * 80)
        
        for eval in evaluations:
            print(f"\n🤖 Judge: {eval.model_name}")
            print(f"⏱️  Evaluation Time: {eval.evaluation_time_ms:.0f}ms")
            print(f"📊 Overall Score: {eval.overall_score:.2f}/5.00")
            print()
            
            for criterion_name, score_obj in [
                ("Accuracy", eval.accuracy),
                ("Completeness", eval.completeness),
                ("Retrieval Adherence", eval.retrieval_adherence)
            ]:
                score_bar = "█" * score_obj.score + "░" * (5 - score_obj.score)
                print(f"  • {score_obj.criterion}: {score_obj.score}/5 [{score_bar}]")
                print(f"    → {score_obj.reasoning}")
                print()
            
            print(f"  💭 Overall Assessment:")
            print(f"     {eval.overall_reasoning}")
            print()
        
        print("\n" + "=" * 80 + "\n")


def create_llm_judge(aws_config: AWSConfig) -> LLMJudge:
    """Factory function to create an LLM Judge."""
    return LLMJudge(aws_config)