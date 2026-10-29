"""
LLM Client wrapper for AWS Bedrock.
Handles all interactions with the language model.
"""
import boto3
import json
import re
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from config import LLMConfig, AWSConfig, create_boto3_session


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a response with tool use capability."""
        pass


class BedrockLLMClient(BaseLLMClient):
    """AWS Bedrock LLM client implementation."""

    def __init__(self, llm_config: LLMConfig, aws_config: AWSConfig):
        self.config = llm_config
        self.aws_config = aws_config
        self.session = create_boto3_session(aws_config)
        self.client = self.session.client(
            "bedrock-runtime",
            region_name=aws_config.region
        )
        # Sent through Converse until a model rejects them
        self._sampling = {"temperature": llm_config.temperature, "topP": llm_config.top_p}

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using LLM on Bedrock."""
        model_id = self.config.model_id
        if not ("anthropic" in model_id or "titan" in model_id or "meta" in model_id or "llama" in model_id.lower()):
            # ⚠️ invoke_model needs each vendor's own request format; Converse takes one format for every model
            return self._converse(prompt, system_prompt)

        # Detect the model type and use the corresponding API format
        if "anthropic" in self.config.model_id:
            # Claude
            messages = [{"role": "user", "content": prompt}]
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "messages": messages
            }
            if system_prompt:
                request_body["system"] = system_prompt

        elif "titan" in self.config.model_id:
            # Amazon Titan
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            request_body = {
                "inputText": full_prompt,
                "textGenerationConfig": {
                    "maxTokenCount": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "topP": self.config.top_p
                }
            }

        elif "meta" in self.config.model_id or "llama" in self.config.model_id.lower():
            # Llama 3 format (different from Llama 2)
            if system_prompt:
                full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            else:
                full_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
            request_body = {
                "prompt": full_prompt,
                "max_gen_len": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p
            }

        response = self.client.invoke_model(
            modelId=self.config.model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())

        # Parse the response based on the model type
        if "anthropic" in self.config.model_id:
            return response_body["content"][0]["text"]
        elif "titan" in self.config.model_id:
            return response_body["results"][0]["outputText"]
        else:
            return response_body["generation"]

    def _converse(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Any other Bedrock model (OpenAI, DeepSeek, Qwen, Mistral...) through Converse, which applies its chat format."""
        request = {
            "modelId": self.config.model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": self.config.max_tokens, **self._sampling},
        }
        if system_prompt:
            request["system"] = [{"text": system_prompt}]
        try:
            response = self.client.converse(**request)
        except self.client.exceptions.ValidationException as exc:
            # Some models reject temperature or top_p (the judge's Opus 5.5 does); drop both once and remember it
            if not self._sampling or not re.search(r"temperature|top_?p", str(exc), re.IGNORECASE):
                raise
            self._sampling = {}
            request["inferenceConfig"] = {"maxTokens": self.config.max_tokens}
            response = self.client.converse(**request)
        if response.get("stopReason") == "max_tokens":
            # Reasoning models spend part of maxTokens before writing, so a cut-off reply is logged, not hidden
            print(f"Converse reply cut off at maxTokens={self.config.max_tokens} ({self.config.model_id})")
        # 💡 Reasoning blocks carry no "text" key, so only the written reply is joined
        return "".join(block.get("text", "") for block in response["output"]["message"]["content"])

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a response with tool use capability."""
        messages = [{"role": "user", "content": prompt}]

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": messages,
            "tools": tools
        }

        if system_prompt:
            request_body["system"] = system_prompt

        response = self.client.invoke_model(
            modelId=self.config.model_id,
            body=json.dumps(request_body)
        )

        response_body = json.loads(response["body"].read())
        return response_body

    def generate_structured(
        self,
        prompt: str,
        output_schema: Dict,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a structured JSON response."""
        # Add JSON formatting instructions
        json_prompt = f"""{prompt}

Respond ONLY with a valid JSON object matching this schema:
{json.dumps(output_schema, indent=2)}

Do not include any text before or after the JSON."""

        response_text = self.generate(json_prompt, system_prompt)

        # Parse JSON from response
        try:
            # Handle potential markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json.loads(response_text.strip())
        except json.JSONDecodeError as e:
            # Return a default error structure
            return {"error": f"Failed to parse JSON: {str(e)}", "raw_response": response_text}


class MockLLMClient(BaseLLMClient):
    """Mock LLM client for testing without AWS."""

    def __init__(self):
        self.call_count = 0

    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self.call_count += 1
        return f"Mock response #{self.call_count} for prompt: {prompt[:100]}..."

    def generate_with_tools(
        self,
        prompt: str,
        tools: List[Dict],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        self.call_count += 1
        return {
            "content": [{"type": "text", "text": f"Mock tool response #{self.call_count}"}],
            "stop_reason": "end_turn"
        }

    def generate_structured(
        self,
        prompt: str,
        output_schema: Dict,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate mock structured response based on schema."""
        self.call_count += 1

        # Return mock data based on what the schema expects
        if "checklist" in str(output_schema):
            # Self-critique response
            return {
                "checklist": [{"item": "dose and fractionation", "addressed": True, "gap": ""}],
                "is_safe": True,
                "concerns": [],
                "confidence_score": 0.85,
                "suggestions": []
            }
        elif "intent" in str(output_schema):
            # Query analysis response
            return {
                "intent": "general",
                "disease_category": None,
                "treatment_modality": None,
                "key_entities": [],
                "temporal_context": None,
                "is_complex": False,
                "reasoning": "Mock analysis"
            }
        elif "sub_tasks" in str(output_schema):
            # Task planning response
            return {
                "reasoning": "Mock planning - breaking down into sub-tasks",
                "sub_tasks": [
                    {
                        "description": "Search for relevant guidelines",
                        "query": prompt[:50],
                        "strategy": "hybrid"
                    }
                ]
            }
        else:
            return {"mock": True, "call_count": self.call_count}


def create_llm_client(
    llm_config: LLMConfig,
    aws_config: AWSConfig,
    use_mock: bool = False
) -> BaseLLMClient:
    """Factory function to create LLM client."""
    if use_mock:
        return MockLLMClient()
    return BedrockLLMClient(llm_config, aws_config)
