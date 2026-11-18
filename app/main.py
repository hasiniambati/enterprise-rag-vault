"""
FastAPI Server for Enterprise Multi-Agent RAG Knowledge Vault
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import os

from .rag import ingest_documents, similarity_search
from .agent import AgentRouter, ValidatorAgent

app = FastAPI(
    title="Enterprise Multi-Agent RAG Knowledge Vault",
    description="Agentic RAG system with automated routing, retrieval, and validation guardrails.",
    version="1.0.0",
)

# Enable CORS for frontend dashboard communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngestRequest(BaseModel):
    documents: List[str] = Field(..., example=["Employees are entitled to 24 annual paid leaves."])
    sources: List[str] = Field(..., example=["HR_Policy_2026.txt"])

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is our annual paid leave policy?")

class QueryResponse(BaseModel):
    query: str
    route_taken: str
    retrieved_chunks: List[dict]
    validated_answer: str
    confidence: str


@app.get("/", include_in_schema=False)
def serve_frontend():
    """Serves the interactive web dashboard frontend."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Welcome to Enterprise Multi-Agent RAG Vault API. Visit /docs for Swagger UI."}


@app.post("/ingest", status_code=status.HTTP_201_CREATED)
def api_ingest(payload: IngestRequest):
    """Ingests enterprise documentation into the vector vault."""
    try:
        metadatas = [{"source": src} for src in payload.sources]
        count = ingest_documents(payload.documents, metadatas)
        return {"status": "success", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def api_query(payload: QueryRequest):
    """
    Agentic Query Workflow:
    1. Router Agent classifies intent.
    2. Retrieval Agent pulls verified knowledge.
    3. Validator Agent checks for alignment and confidence.
    """
    route = AgentRouter.route_query(payload.query)
    retrieved_data = similarity_search(payload.query, k=2)
    validation_result = ValidatorAgent.validate_response(retrieved_data, payload.query)

    return QueryResponse(
        query=payload.query,
        route_taken=route,
        retrieved_chunks=retrieved_data,
        validated_answer=validation_result["validated_answer"],
        confidence=validation_result["confidence"]
    )


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Agentic RAG Vault API"}