"""
Agentic RAG Engine for Radiation Oncology Guidelines

A multi-step retrieval agent that:
- Parses clinical queries and formulates sub-tasks
- Chooses the best retriever (dense vs. sparse, vector vs. keyword)
- Synthesizes guideline passages with citations
- Self-critiques for completeness and safety
"""

from .agent import AgenticRAGEngine, create_agent
from .config import Config, default_config
from .models import AgentResponse, QueryAnalysis, Citation

__version__ = "1.0.0"
__all__ = [
    "AgenticRAGEngine",
    "create_agent", 
    "Config",
    "default_config",
    "AgentResponse",
    "QueryAnalysis",
    "Citation"
]
