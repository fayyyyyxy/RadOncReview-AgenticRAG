"""
REST API for the Agentic RAG Engine.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Annotated
import uvicorn
from datetime import datetime
import os
import asyncio
import json
import time
from threading import Lock
from pathlib import Path

from core.agent import AgenticRAGEngine, create_agent
from core.case_labels import NOT_SPECIFIED, CaseLabels, DISEASE_SITES
from core.site_routing import scope_for
from config import validate_config

validate_config()


# ============ Pydantic Models ============

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="Clinical question")
    include_reasoning: bool = Field(default=False, description="Include reasoning trace")
    include_citations: bool = Field(default=True, description="Include source citations")
    run_evaluation: bool = Field(default=False, description="Run the five extra quality assessments. Evidence sufficiency always runs.")
    # Disease site is used for retrieval when filter_by_site is enabled.
    disease_site: str = Field(default=NOT_SPECIFIED, description="Disease site of the case")
    setting_group: str = Field(default=NOT_SPECIFIED, description="Treatment setting group")
    prior_rt: str = Field(default=NOT_SPECIFIED, description="Prior radiotherapy")
    encounter_type: str = Field(default=NOT_SPECIFIED, description="Encounter type")
    filter_by_site: bool = Field(default=False, description="Restrict retrieval to the selected disease site")
    prior_questions: List[Annotated[str, Field(min_length=1, max_length=2000)]] = Field(
        default_factory=list, max_length=2, description="Initial and most recent earlier user questions")
    conversation_context: str = Field(default="", max_length=4000,
        description="User-visible, editable case context; not clinical evidence")

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
    evidence_passages: List[Dict[str, Any]] = Field(default_factory=list)
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
_agent_init_lock = Lock()
# Keep the existing stateful engine single-query-at-a-time. Only searches within
# one query run concurrently; worker offloading keeps health/progress responsive.
_query_gate = asyncio.Lock()
_stream_tasks = set()


def get_agent() -> AgenticRAGEngine:
    global _agent
    with _agent_init_lock:
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


def request_labels(request):
    try:
        return CaseLabels(request.disease_site, request.setting_group, request.prior_rt, request.encounter_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


def _run_query(request, labels, progress):
    progress("initialize")
    initialize_started = time.perf_counter()
    agent = get_agent()
    initialize_ms = (time.perf_counter() - initialize_started) * 1000
    disease_site = labels.disease_site if request.filter_by_site else NOT_SPECIFIED
    response = agent.query(
        request.query, labels, prior_questions=request.prior_questions,
        conversation_context=request.conversation_context,
        evidence_scope=scope_for(disease_site),
        progress_callback=progress, retain_history=False,
        run_evaluation=request.run_evaluation,
    )

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

    payload = QueryResponse(
        query=response.query,
        answer=response.answer,
        citations=citations,
        evidence_passages=response.context_passages if request.include_citations else [],
        reasoning_trace=response.reasoning_trace if request.include_reasoning else None,
        metadata=response.metadata,
        latency_ms=response.latency_ms,
        timestamp=response.created_at.isoformat()
    )
    payload.metadata["search_scope"] = {"disease_site": disease_site}
    payload.metadata["initialize_ms"] = initialize_ms
    return payload


async def _execute_query(request, labels, progress=lambda stage: None):
    started = time.perf_counter()
    progress("queued")
    async with _query_gate:
        queue_ms = (time.perf_counter() - started) * 1000
        # Shield the worker so cancellation cannot release the gate while it still
        # mutates the engine. A disconnected client does not cancel Bedrock calls.
        worker = asyncio.create_task(run_in_threadpool(_run_query, request, labels, progress))
        try:
            payload = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await worker
            raise
    payload.metadata["engine_latency_ms"] = payload.latency_ms
    payload.metadata["queue_ms"] = queue_ms
    payload.latency_ms = (time.perf_counter() - started) * 1000
    return payload


@app.get("/options", tags=["Query"])
async def query_options():
    return {"disease_sites": DISEASE_SITES}


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query_guidelines(request: QueryRequest):
    labels = request_labels(request)
    try:
        return await _execute_query(request, labels)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.post("/query/stream", tags=["Query"])
async def query_with_progress(request: QueryRequest):
    """Stream stage updates, then one complete answer after all enabled checks/evaluations."""
    labels = request_labels(request)

    async def events():
        loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=32)
        listening = True

        def push_progress(stage):
            # Reserve one slot for the final answer or error.
            if listening and queue.qsize() < queue.maxsize - 1:
                queue.put_nowait({"type": "progress", "stage": stage})

        def progress(stage):
            loop.call_soon_threadsafe(push_progress, stage)

        async def produce():
            try:
                payload = await _execute_query(request, labels, progress)
                if listening:
                    queue.put_nowait({"type": "result", "data": payload.model_dump(mode="json")})
            except Exception:
                if listening:
                    queue.put_nowait({"type": "error", "message": "The request could not finish. Please retry your question."})

        producer = asyncio.create_task(produce())
        _stream_tasks.add(producer)
        producer.add_done_callback(_stream_tasks.discard)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                except asyncio.TimeoutError:
                    yield json.dumps({"type": "heartbeat"}) + "\n"
                    continue
                yield json.dumps(event) + "\n"
                if event["type"] in ("result", "error"):
                    break
        finally:
            listening = False

    return StreamingResponse(events(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


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
