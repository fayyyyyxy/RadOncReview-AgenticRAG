"""
REST API for the Agentic RAG Engine.
Provides endpoints for querying and managing the system.
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import hashlib
import hmac
import os
from pathlib import Path

from agent import AgenticRAGEngine, create_agent
from config import Config, default_config


# ============ Pydantic Models ============

class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="Clinical question")
    include_reasoning: bool = Field(default=False, description="Include reasoning trace")
    include_citations: bool = Field(default=True, description="Include source citations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the standard dose fractionation for early-stage breast cancer?",
                "include_reasoning": True,
                "include_citations": True
            }
        }


class CitationResponse(BaseModel):
    """Citation in the response."""
    citation_id: int
    source_path: str
    relevance_score: float
    excerpt: Optional[str] = None
    source_url: Optional[str] = None
    document_title: Optional[str] = None
    source_type: str = "review_notes"


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    query: str
    answer: str
    citations: List[CitationResponse] = []
    reasoning_trace: Optional[List[str]] = None
    metadata: Dict[str, Any] = {}
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    timestamp: str


class FeedbackRequest(BaseModel):
    """Request model for feedback endpoint."""
    query_id: str
    feedback: str = Field(..., pattern="^(helpful|needs_revision)$")
    comment: Optional[str] = None


# ============ API Application ============

app = FastAPI(
    title="RadOnc RAG API",
    description="Agentic RAG system for radiation oncology treatment guidelines",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global agent instance (lazy initialization)
_agent: Optional[AgenticRAGEngine] = None


def get_agent() -> AgenticRAGEngine:
    """Get or create the agent instance."""
    global _agent
    if _agent is None:
        # Check for mock mode (for testing without AWS)
        use_mock = os.getenv("USE_MOCK_AGENT", "false").lower() == "true"
        _agent = create_agent(use_mock=use_mock)
    return _agent


# Simple API key authentication (for demo - use proper auth in production)
API_KEY = os.getenv("API_KEY", "YOUR_API_KEY")


async def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key from header."""
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key required")
    
    # Constant-time comparison to prevent timing attacks
    if not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    
    return x_api_key


# ============ Endpoints ============

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health status."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_guidelines(
    request: QueryRequest,
    api_key: str = Depends(verify_api_key),
    agent: AgenticRAGEngine = Depends(get_agent)
):
    """
    Query the radiation oncology guidelines.
    
    Processes the clinical question through the agentic RAG pipeline:
    1. Analyzes query intent and entities
    2. Plans retrieval strategy
    3. Retrieves relevant guideline chunks
    4. Synthesizes answer with citations
    5. Self-critiques for quality assurance
    """
    try:
        # Process query through agent
        response = agent.query(request.query)
        
        # Build citation responses
        citations = []
        if request.include_citations:
            for citation in response.citations:
                citations.append(CitationResponse(
                    citation_id=citation.citation_id,
                    source_path=citation.source_path,
                    relevance_score=citation.relevance_score,
                    excerpt=citation.chunk_text[:200] if citation.chunk_text else None,
                    source_url=citation.source_url or None,
                    document_title=citation.document_title or None,
                    source_type=citation.source_type,
                ))
        
        # Build response
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            citations=citations,
            reasoning_trace=response.reasoning_trace if request.include_reasoning else None,
            metadata=response.metadata,
            latency_ms=response.latency_ms,
            timestamp=response.created_at.isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    request: FeedbackRequest,
    api_key: str = Depends(verify_api_key)
):
    """
    Submit feedback on a query response.
    
    Used for reinforcement learning and quality improvement.
    """
    # In production, store feedback in database
    # For now, just acknowledge receipt
    return {
        "status": "received",
        "query_id": request.query_id,
        "feedback": request.feedback,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/categories", tags=["Reference"])
async def list_categories(api_key: str = Depends(verify_api_key)):
    """List available disease categories in the knowledge base."""
    return {
        "categories": [
            {"id": "breast", "name": "Breast Cancer", "status": "available"},
            {"id": "thorax", "name": "Thoracic Cancers", "status": "available"},
            {"id": "hematologic", "name": "Hematologic Malignancies", "status": "available"},
            {"id": "cns", "name": "CNS Tumors", "status": "coming_soon"},
            {"id": "gi", "name": "GI Cancers", "status": "coming_soon"},
            {"id": "gu", "name": "GU Cancers", "status": "coming_soon"},
            {"id": "gyn", "name": "Gynecologic Cancers", "status": "coming_soon"},
            {"id": "head_neck", "name": "Head & Neck Cancers", "status": "coming_soon"},
        ]
    }


@app.get("/modalities", tags=["Reference"])
async def list_modalities(api_key: str = Depends(verify_api_key)):
    """List treatment modalities covered in the knowledge base."""
    return {
        "modalities": [
            {"id": "ebrt", "name": "External Beam RT (EBRT)", "description": "Standard external beam radiation"},
            {"id": "imrt", "name": "IMRT", "description": "Intensity-modulated radiation therapy"},
            {"id": "vmat", "name": "VMAT", "description": "Volumetric modulated arc therapy"},
            {"id": "sbrt", "name": "SBRT/SABR", "description": "Stereotactic body radiation therapy"},
            {"id": "srs", "name": "SRS", "description": "Stereotactic radiosurgery"},
            {"id": "brachy", "name": "Brachytherapy", "description": "Internal radiation therapy"},
            {"id": "proton", "name": "Proton Therapy", "description": "Particle beam therapy"},
        ]
    }


# ============ Static Frontend ============

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", tags=["Frontend"])
async def serve_frontend():
    """Serve the chat UI."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(
        index_path,
        media_type="text/html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ============ Run Server ============

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
