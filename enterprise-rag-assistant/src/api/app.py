"""FastAPI backend with streaming and advanced features"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
from pathlib import Path
import shutil
from datetime import datetime
import uuid
import json

from src.core.rag_engine import RAGEngine

app = FastAPI(title="Enterprise RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_engine = None
session_docs = {}

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    session_id: Optional[str] = None
    use_hybrid: bool = True

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    session_id: str
    processing_time_ms: float
    hybrid_used: bool

class DocumentUploadResponse(BaseModel):
    session_id: str
    documents_processed: int
    chunks_created: int
    filename: str

@app.on_event("startup")
async def startup_event():
    global rag_engine
    import os
    use_openai = os.getenv("USE_OPENAI", "false").lower() == "true"
    rag_engine = RAGEngine(use_openai=use_openai)

@app.post("/upload", response_model=DocumentUploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = None,
    background_tasks: BackgroundTasks = None
):
    """Upload documents for a session"""
    global rag_engine, session_docs
    
    if not session_id:
        session_id = str(uuid.uuid4())
    
    upload_dir = Path(f"./uploads/{session_id}")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for file in files:
        if not file.filename.endswith(('.pdf', '.txt', '.docx')):
            raise HTTPException(400, f"Unsupported file type: {file.filename}")
        
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(str(file_path))
    
    def process_docs():
        num_chunks = rag_engine.load_documents(saved_paths)
        rag_engine.create_vector_store(f"./vector_store/{session_id}")
        session_docs[session_id] = saved_paths
    
    if background_tasks:
        background_tasks.add_task(process_docs)
    else:
        process_docs()
    
    return DocumentUploadResponse(
        session_id=session_id,
        documents_processed=len(saved_paths),
        chunks_created=len(rag_engine.documents) if rag_engine.documents else 0,
        filename=", ".join([f.filename for f in files])
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the documents"""
    import time
    start_time = time.time()
    
    if not rag_engine or not rag_engine.qa_chain:
        raise HTTPException(400, "No documents loaded. Please upload documents first.")
    
    result = rag_engine.query(request.question, use_hybrid=request.use_hybrid)
    
    processing_time = (time.time() - start_time) * 1000
    
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        session_id=request.session_id or "default",
        processing_time_ms=round(processing_time, 2),
        hybrid_used=result.get("hybrid_search_used", False)
    )

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "vector_store_ready": rag_engine and rag_engine.vector_store is not None,
        "documents_loaded": len(rag_engine.documents) if rag_engine else 0
    }