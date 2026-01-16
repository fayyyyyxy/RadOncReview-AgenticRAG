"""
LLM Client wrapper for AWS Bedrock.
Handles all interactions with the language model.
"""
import boto3
import json
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from config import LLMConfig, AWSConfig


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
        self.session = boto3.Session(profile_name=aws_config.profile_name)
        self.client = self.session.client(
            "bedrock-runtime", 
            region_name=aws_config.region
        )
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using Claude on Bedrock."""
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
        
        response = self.client.invoke_model(
            modelId=self.config.model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response["body"].read())
        return response_body["content"][0]["text"]
    
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


def create_llm_client(
    llm_config: LLMConfig, 
    aws_config: AWSConfig,
    use_mock: bool = False
) -> BaseLLMClient:
    """Factory function to create LLM client."""
    if use_mock:
        return MockLLMClient()
    return BedrockLLMClient(llm_config, aws_config)
