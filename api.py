"""
REST API for the Agentic RAG Engine.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
from datetime import datetime
import os
from pathlib import Path

from core.agent import AgenticRAGEngine, create_agent
from config import Config, default_config


# ============ Pydantic Models ============

class QueryRequest(BaseModel):
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
    citation_id: int
    source_path: str
    relevance_score: float
    excerpt: Optional[str] = None
    source_url: Optional[str] = None
    document_title: Optional[str] = None
    source_type: str = "review_notes"


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationResponse] = []
    reasoning_trace: Optional[List[str]] = None
    metadata: Dict[str, Any] = {}
    latency_ms: float
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ============ API Application ============

app = FastAPI(
    title="RadOnc RAG API",
    description="Agentic RAG system for radiation oncology treatment guidelines",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_agent: Optional[AgenticRAGEngine] = None


def get_agent() -> AgenticRAGEngine:
    global _agent
    if _agent is None:
        use_mock = os.getenv("USE_MOCK_AGENT", "false").lower() == "true"
        _agent = create_agent(use_mock=use_mock)
    return _agent


# ============ Endpoints ============

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_guidelines(
    request: QueryRequest,
    agent: AgenticRAGEngine = Depends(get_agent)
):
    try:
        response = agent.query(request.query)

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


# ============ Static Frontend ============

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", tags=["Frontend"])
async def serve_frontend():
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
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
