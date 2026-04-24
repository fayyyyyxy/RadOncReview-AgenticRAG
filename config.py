"""
Configuration settings for the Agentic RAG Engine.
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import boto3


@dataclass
class AWSConfig:
    """AWS service configuration."""
    # Local dev: defaults to "YOUR_AWS_PROFILE" profile.
    # EC2 deploy: set AWS_PROFILE_NAME="" to fall back to the instance IAM role.
    profile_name: Optional[str] = os.getenv("AWS_PROFILE_NAME", "YOUR_AWS_PROFILE") or None
    region: str = os.getenv("AWS_REGION", "YOUR_AWS_REGION")
    s3_bucket: str = "YOUR_S3_BUCKET"
    dynamodb_table: str = "YOUR_DYNAMODB_TABLE"  
    faiss_index_key: str = "radonc_faiss_index.bin"

    knowledge_base_id: str = "YOUR_KB_ID"
    knowledge_base_enabled: bool = True
    knowledge_base_datasource_id: str = "YOUR_KB_DATASOURCE_ID"


@dataclass
class LLMConfig:
    """LLM configuration for AWS Bedrock."""
    model_id: str = "meta.llama3-70b-instruct-v1:0"
    max_tokens: int = 2048  # Llama 3 max_gen_len
    temperature: float = 0.1  # Low temperature for factual responses
    top_p: float = 0.9


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_id: str = "amazon.titan-embed-text-v2:0"
    dimension: int = 1024  # Titan v2 embedding dimension


@dataclass
class RetrieverConfig:
    """Retrieval configuration."""
    top_k: int = 15                  # fallback for retriever.py (not used directly in planning)
    similarity_threshold: float = 0.3

    # Simple query
    faiss_top_k: int = 6             # FAISS retrieve for simple queries
    kb_top_k: int = 10              # KB retrieve for simple queries
    rerank_top_k: int = 5           # final chunks sent to LLM after rerank

    # Complex query
    faiss_complex_top_k: int = 6    # FAISS retrieve per subtopic task
    kb_complex_top_k: int = 10      # KB retrieve per subtopic task
    complex_rerank_top_k: int = 3   # final chunks per topic after rerank


@dataclass
class AgentConfig:
    """Agent behavior configuration."""
    max_iterations: int = 5  # Maximum reasoning iterations
    enable_self_critique: bool = True
    enable_source_citation: bool = True
    confidence_threshold: float = 0.7  # Minimum confidence to provide answer
    
    # Radiation oncology specific settings
    disease_categories: List[str] = field(default_factory=lambda: [
        "BrainMets", "Oligo", "Immuno", "Breast", "CNS", "Pediatrics", "Gastrointestinal", 
        "Genitourinary", "Gynecology", "Head & Neck", "Skin", "Hematologic", "Sarcoma", "Thorax"
    ])
    
    # Treatment modalities for query classification
    treatment_modalities: List[str] = field(default_factory=lambda: [
        "EBRT", "IMRT", "VMAT", "SRS", "SBRT", "Brachytherapy", 
        "Proton", "Electron", "TBI", "TSEI"
    ])


@dataclass
class GuardrailConfig:
    """Safety and compliance guardrails."""
    enable_phi_filter: bool = True
    enable_version_check: bool = True
    max_input_length: int = 2000
    blocked_terms: List[str] = field(default_factory=lambda: [
        # PHI patterns to filter
        "SSN", "social security", "date of birth", "DOB",
        "medical record number", "MRN", "patient name"
    ])


@dataclass
class Config:
    """Main configuration container."""
    aws: AWSConfig = field(default_factory=AWSConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)


# Default configuration instance
default_config = Config()


def create_boto3_session(aws_config: Optional[AWSConfig] = None) -> boto3.Session:
    """Create a boto3 Session that works both locally (named profile) and on EC2 (instance role).

    If profile_name is set, uses it (local dev). If None/empty, returns a default
    Session so boto3's credential chain picks up the EC2 instance role.
    """
    cfg = aws_config or default_config.aws
    if cfg.profile_name:
        return boto3.Session(profile_name=cfg.profile_name)
    return boto3.Session()