"""
Data models for the Agentic RAG Engine.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum
from datetime import datetime


class QueryIntent(Enum):
    """Classification of query intent."""
    TREATMENT_PROTOCOL = "treatment_protocol"  # Treatment guidelines/protocols
    DOSE_FRACTIONATION = "dose_fractionation"  # Dose and fractionation questions
    SIDE_EFFECTS = "side_effects"  # Toxicity and side effects
    STAGING = "staging"  # Cancer staging information
    INDICATIONS = "indications"  # Treatment indications
    CONTRAINDICATIONS = "contraindications"  # Contraindications
    COMPARISON = "comparison"  # Compare treatments/approaches
    DEFINITION = "definition"  # Define terms/concepts
    GENERAL = "general"  # General information


class RetrievalStrategy(Enum):
    """Retrieval strategy types."""
    DENSE = "dense"  # Vector similarity search
    SPARSE = "sparse"  # BM25/keyword search
    HYBRID = "hybrid"  # Combined dense + sparse
    MULTI_HOP = "multi_hop"  # Multi-step retrieval for complex queries
    KNOWLEDGE_BASE = "knowledge_base" # Bedrock KB (reading list papers)


class TaskStatus(Enum):
    """Status of a sub-task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class QueryAnalysis:
    """Result of query analysis."""
    original_query: str
    intent: QueryIntent
    disease_category: Optional[str] = None
    treatment_modality: Optional[str] = None
    key_entities: List[str] = field(default_factory=list)
    temporal_context: Optional[str] = None  # e.g., "postoperative", "adjuvant"
    is_complex: bool = False  # Requires multi-step reasoning
    confidence: float = 0.0
    needs_trial_evidence: bool = False  # query asks about trial results/evidence


@dataclass
class SubTask:
    """A sub-task in the agent's plan."""
    task_id: str
    description: str
    query: str
    retrieval_strategy: RetrievalStrategy
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    retrieved_chunks: List[Dict] = field(default_factory=list)
    relevance_scores: List[float] = field(default_factory=list)
    top_k: Optional[int] = None  # Limit retrieval count (for comparison queries)


@dataclass
class TaskPlan:
    """Agent's task execution plan."""
    plan_id: str
    original_query: str
    analysis: QueryAnalysis
    sub_tasks: List[SubTask] = field(default_factory=list)
    reasoning: str = ""  # Explanation of the plan
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk."""
    faiss_id: int
    chunk_id: int
    text: str
    path: str  # Hierarchical path (Category > Heading1 > Heading2...)
    category: str
    headings: Dict[str, str] = field(default_factory=dict)
    similarity_score: float = 0.0
    relevance_score: float = 0.0  # After reranking

    source_type: str = "review_notes"  # "review_notes" (FAISS) | "clinical_trial" (KB)
    document_title: str = ""           # e.g. "CROSS Trial, NEJM 2012"
    s3_location: str = ""              # full S3 URI of source file


@dataclass
class Citation:
    """Source citation."""
    citation_id: int
    source_path: str
    chunk_text: str
    relevance_score: float
    faiss_id: int
    source_url: Optional[str] = None        # source url
    document_title: Optional[str] = None    #
    source_type: str = "review_notes"       #


@dataclass
class SelfCritiqueResult:
    """Result of self-critique evaluation."""
    is_complete: bool
    is_accurate: bool
    is_safe: bool
    missing_aspects: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    suggestions: List[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    """Final response from the agent."""
    query: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence_score: float = 0.0
    reasoning_trace: List[str] = field(default_factory=list)
    self_critique: Optional[SelfCritiqueResult] = None
    task_plan: Optional[TaskPlan] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    turn_id: int
    user_query: str
    agent_response: AgentResponse
    feedback: Optional[str] = None  # "helpful" or "needs_revision"
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Conversation:
    """A conversation session."""
    conversation_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
