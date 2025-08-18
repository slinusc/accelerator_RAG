from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import structlog
from datetime import datetime

from src.core.rag_orchestrator import AgenticRAGOrchestrator, RAGResponse
from config.settings import settings

# Configure logging
logger = structlog.get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SwissFEL Agentic RAG API",
    description="Intelligent RAG system for SwissFEL commissioning data",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG orchestrator
rag_orchestrator: Optional[AgenticRAGOrchestrator] = None


# Pydantic models
class QueryRequest(BaseModel):
    query: str
    max_sources: Optional[int] = 10
    include_elog: Optional[bool] = True
    include_knowledge_graph: Optional[bool] = True


class QueryResponse(BaseModel):
    answer: str
    confidence_score: float
    query_type: str
    processing_time: float
    sources: List[Dict[str, str]]
    follow_up_questions: List[str]
    timestamp: str
    total_sources: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, str]


class StatsResponse(BaseModel):
    elog_stats: Dict[str, int]
    knowledge_graph_status: str
    llm_status: str


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG orchestrator on startup."""
    global rag_orchestrator
    try:
        rag_orchestrator = AgenticRAGOrchestrator()
        logger.info("RAG orchestrator initialized")
    except Exception as e:
        logger.error("Failed to initialize RAG orchestrator", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global rag_orchestrator
    if rag_orchestrator:
        try:
            await rag_orchestrator.close()
            logger.info("RAG orchestrator closed")
        except Exception as e:
            logger.error("Error closing RAG orchestrator", error=str(e))


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with basic info."""
    return {
        "name": "SwissFEL Agentic RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    components = {}
    
    try:
        # Check ELOG connection
        if rag_orchestrator and rag_orchestrator.elog_client:
            try:
                stats = rag_orchestrator.elog_client.get_entry_statistics()
                components["elog"] = "healthy" if stats else "unhealthy"
            except Exception:
                components["elog"] = "unhealthy"
        else:
            components["elog"] = "not_initialized"
        
        # Check knowledge graph
        if rag_orchestrator and rag_orchestrator.knowledge_client:
            try:
                # Simple test query
                results = rag_orchestrator.knowledge_client.search_similar_content("test", limit=1)
                components["knowledge_graph"] = "healthy"
            except Exception:
                components["knowledge_graph"] = "unhealthy"
        else:
            components["knowledge_graph"] = "not_initialized"
        
        # Check LLM
        if rag_orchestrator and rag_orchestrator.llm_generator:
            try:
                health = await rag_orchestrator.llm_generator.ollama.check_health()
                components["llm"] = "healthy" if health else "unhealthy"
            except Exception:
                components["llm"] = "unhealthy"
        else:
            components["llm"] = "not_initialized"
        
        # Overall status
        status = "healthy" if all(c in ["healthy"] for c in components.values()) else "degraded"
        
        return HealthResponse(
            status=status,
            timestamp=datetime.now().isoformat(),
            components=components
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            components={"error": str(e)}
        )


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get system statistics."""
    if not rag_orchestrator:
        raise HTTPException(status_code=503, detail="RAG orchestrator not initialized")
    
    try:
        # ELOG stats
        elog_stats = rag_orchestrator.elog_client.get_entry_statistics()
        
        # Knowledge graph status
        kg_status = "healthy"
        try:
            rag_orchestrator.knowledge_client.search_similar_content("test", limit=1)
        except Exception:
            kg_status = "unhealthy"
        
        # LLM status
        llm_status = "healthy"
        try:
            health = await rag_orchestrator.llm_generator.ollama.check_health()
            if not health:
                llm_status = "unhealthy"
        except Exception:
            llm_status = "unhealthy"
        
        return StatsResponse(
            elog_stats=elog_stats,
            knowledge_graph_status=kg_status,
            llm_status=llm_status
        )
        
    except Exception as e:
        logger.error("Failed to get stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a RAG query."""
    if not rag_orchestrator:
        raise HTTPException(status_code=503, detail="RAG orchestrator not initialized")
    
    try:
        # Process the query
        response: RAGResponse = await rag_orchestrator.process_query(request.query)
        
        return QueryResponse(
            answer=response.answer,
            confidence_score=response.context.confidence_score,
            query_type=response.context.query_type.value,
            processing_time=response.context.processing_time,
            sources=response.sources,
            follow_up_questions=response.follow_up_questions,
            timestamp=response.timestamp.isoformat(),
            total_sources=response.context.total_sources
        )
        
    except Exception as e:
        logger.error("Query processing failed", query=request.query, error=str(e))
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/elog/recent")
async def get_recent_elog_entries(days: int = 7, limit: int = 10):
    """Get recent ELOG entries."""
    if not rag_orchestrator:
        raise HTTPException(status_code=503, detail="RAG orchestrator not initialized")
    
    try:
        entries = rag_orchestrator.elog_client.get_recent_entries(days=days)
        
        # Convert to serializable format
        serialized_entries = []
        for entry in entries[:limit]:
            serialized_entries.append({
                "id": entry.id,
                "date": entry.date.isoformat(),
                "author": entry.author,
                "subject": entry.subject,
                "text": entry.text[:500] + "..." if len(entry.text) > 500 else entry.text,
                "url": entry.url,
                "attachments": entry.attachments
            })
        
        return {
            "entries": serialized_entries,
            "total": len(entries),
            "days": days
        }
        
    except Exception as e:
        logger.error("Failed to get recent ELOG entries", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/knowledge/search")
async def search_knowledge_graph(q: str, limit: int = 5, threshold: float = 0.7):
    """Search the knowledge graph."""
    if not rag_orchestrator:
        raise HTTPException(status_code=503, detail="RAG orchestrator not initialized")
    
    try:
        results = rag_orchestrator.knowledge_client.search_similar_content(
            query=q,
            limit=limit,
            score_threshold=threshold
        )
        
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error("Knowledge graph search failed", query=q, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/available")
async def get_available_models():
    """Get available Ollama models."""
    if not rag_orchestrator:
        raise HTTPException(status_code=503, detail="RAG orchestrator not initialized")
    
    try:
        models = await rag_orchestrator.llm_generator.ollama.list_models()
        return {
            "models": models,
            "current_model": settings.ollama_model
        }
        
    except Exception as e:
        logger.error("Failed to get available models", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
        log_level=settings.log_level.lower()
    )