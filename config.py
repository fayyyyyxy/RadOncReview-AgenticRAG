"""
Configuration settings for the Agentic RAG Engine.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AWSConfig:
    """AWS service configuration."""
    profile_name: str = "YOUR_AWS_PROFILE"
    region: str = "YOUR_AWS_REGION"
    s3_bucket: str = "YOUR_S3_BUCKET"
    dynamodb_table: str = "YOUR_DYNAMODB_TABLE"
    faiss_index_key: str = "radonc_faiss_index.bin"


@dataclass
class LLMConfig:
    """LLM configuration for AWS Bedrock."""
    # Available models (based on check_available_models.py results):
    # - meta.llama3-70b-instruct-v1:0  ✅ Available (recommended, more powerful)
    # - meta.llama3-8b-instruct-v1:0   ✅ Available (faster, cheaper)
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
    top_k: int = 15  # Number of chunks to retrieve initially for simple queries
    similarity_threshold: float = 0.3  # Minimum similarity score (lowered for better recall)
    rerank_top_k: int = 7  # Final number after reranking for simple queries
    complex_top_k: int = 10  # Number to retrieve per subtask
    complex_rerank_top_k: int = 5  # Final number per topic after reranking
    use_hybrid_search: bool = True  # Combine dense + sparse retrieval


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
