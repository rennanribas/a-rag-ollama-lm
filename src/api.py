"""FastAPI service for the AI RAG Agent."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from .agent import RAGAgent, QueryContext, AgentResponse
from .config import settings
from .crawler import WebCrawler
from .indexer import IncrementalIndexer


# Request/Response models
class CrawlRequest(BaseModel):
    urls: List[str]
    max_depth: int = 2
    force_refresh: bool = False


class QueryRequest(BaseModel):
    query: str
    domain: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict = {}


class CrawlStatus(BaseModel):
    status: str
    message: str
    documents_processed: int = 0
    total_documents: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str
    indexer_stats: Dict
    conversation_stats: Dict


# Global instances
indexer: Optional[IncrementalIndexer] = None
agent: Optional[RAGAgent] = None
crawl_status: Dict[str, CrawlStatus] = {}


# FastAPI app
app = FastAPI(
    title="AI RAG Agent",
    description="Domain-specific RAG agent with incremental crawling and OpenAI integration",
    version="1.0.0"
)

# CORS middleware for IDE integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    global indexer, agent
    
    try:
        logger.info("Initializing AI RAG Agent...")
        
        # Initialize indexer
        indexer = IncrementalIndexer()
        logger.info("Indexer initialized")
        
        # Initialize agent
        agent = RAGAgent(indexer)
        logger.info("Agent initialized")
        
        logger.info("AI RAG Agent startup complete")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not indexer or not agent:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        indexer_stats=indexer.get_stats(),
        conversation_stats=agent.get_conversation_stats()
    )


@app.post("/crawl")
async def start_crawl(request: CrawlRequest, background_tasks: BackgroundTasks):
    """Start crawling specified URLs."""
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")
    
    crawl_id = f"crawl_{len(crawl_status)}"
    
    # Initialize crawl status
    crawl_status[crawl_id] = CrawlStatus(
        status="starting",
        message="Crawl job queued"
    )
    
    # Start crawl in background
    background_tasks.add_task(
        _perform_crawl,
        crawl_id,
        request.urls,
        request.max_depth,
        request.force_refresh
    )
    
    return {
        "crawl_id": crawl_id,
        "status": "started",
        "message": "Crawl job started in background"
    }


@app.get("/crawl/{crawl_id}/status", response_model=CrawlStatus)
async def get_crawl_status(crawl_id: str):
    """Get the status of a crawl job."""
    if crawl_id not in crawl_status:
        raise HTTPException(status_code=404, detail="Crawl job not found")
    
    return crawl_status[crawl_id]


@app.post("/query", response_model=AgentResponse)
async def query_agent(request: QueryRequest):
    """Query the RAG agent."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        query_context = QueryContext(
            query=request.query,
            domain=request.domain,
            user_id=request.user_id,
            session_id=request.session_id,
            metadata=request.metadata
        )
        
        response = await agent.process_query(query_context)
        return response
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")
    
    try:
        stats = indexer.get_stats()
        return {
            "total_documents": stats.get("total_documents", 0),
            "collection_name": stats.get("collection_name"),
            "storage_directory": stats.get("storage_directory")
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/documents")
async def clear_documents():
    """Clear all indexed documents."""
    if not indexer:
        raise HTTPException(status_code=503, detail="Indexer not initialized")
    
    try:
        indexer.clear_index()
        return {"message": "All documents cleared successfully"}
    except Exception as e:
        logger.error(f"Failed to clear documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversations")
async def clear_conversations(session_id: Optional[str] = None):
    """Clear conversation history."""
    if not agent:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        agent.clear_conversation_history(session_id)
        message = f"Conversation history cleared for session: {session_id}" if session_id else "All conversation history cleared"
        return {"message": message}
    except Exception as e:
        logger.error(f"Failed to clear conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get comprehensive system statistics."""
    if not indexer or not agent:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    return {
        "indexer": indexer.get_stats(),
        "agent": agent.get_conversation_stats(),
        "crawl_jobs": {
            "total": len(crawl_status),
            "active": len([s for s in crawl_status.values() if s.status in ["running", "starting"]]),
            "completed": len([s for s in crawl_status.values() if s.status == "completed"])
        }
    }


# Predefined domain configurations
@app.get("/domains")
async def get_domain_configs():
    """Get predefined domain configurations for common documentation sites."""
    return {
        "apple_ios": {
            "name": "Apple iOS Development",
            "urls": [
                "https://developer.apple.com/documentation/uikit",
                "https://developer.apple.com/documentation/swiftui",
                "https://developer.apple.com/documentation/foundation",
                "https://developer.apple.com/documentation/swift"
            ],
            "max_depth": 3
        },
        "apple_watchos": {
            "name": "Apple watchOS Development",
            "urls": [
                "https://developer.apple.com/documentation/watchkit",
                "https://developer.apple.com/documentation/healthkit",
                "https://developer.apple.com/documentation/workoutkit",
                "https://developer.apple.com/watchos/"
            ],
            "max_depth": 3
        },
        "swift": {
            "name": "Swift Language",
            "urls": [
                "https://docs.swift.org/swift-book/",
                "https://developer.apple.com/documentation/swift"
            ],
            "max_depth": 4
        },
        "llamaindex": {
            "name": "LlamaIndex Documentation",
            "urls": [
                "https://docs.llamaindex.ai/en/stable/"
            ],
            "max_depth": 3
        },
        "openai": {
            "name": "OpenAI API Documentation",
            "urls": [
                "https://platform.openai.com/docs"
            ],
            "max_depth": 3
        },
        "nestjs": {
            "name": "NestJS Documentation",
            "urls": [
                "https://docs.nestjs.com/"
            ],
            "max_depth": 3
        },
        "nodejs": {
            "name": "Node.js Documentation",
            "urls": [
                "https://nodejs.org/docs/latest/api/"
            ],
            "max_depth": 3
        },
        "typescript": {
            "name": "TypeScript Documentation",
            "urls": [
                "https://www.typescriptlang.org/docs/"
            ],
            "max_depth": 3
        },
        "sequelize": {
            "name": "Sequelize Documentation",
            "urls": [
                "https://sequelize.org/docs/v6/"
            ],
            "max_depth": 3
        },
        "sqlserver": {
            "name": "SQL Server Documentation",
            "urls": [
                "https://learn.microsoft.com/en-us/sql/"
            ],
            "max_depth": 3
        },
        "swagger": {
            "name": "Swagger Documentation",
            "urls": [
                "https://swagger.io/docs/"
            ],
            "max_depth": 3
        },
        "redoc": {
            "name": "Redoc Documentation",
            "urls": [
                "https://redocly.com/docs/redoc/"
            ],
            "max_depth": 3
        }
    }


@app.post("/domains/{domain_name}/crawl")
async def crawl_domain(domain_name: str, background_tasks: BackgroundTasks, force_refresh: bool = False):
    """Start crawling a predefined domain."""
    domains = await get_domain_configs()
    
    if domain_name not in domains:
        raise HTTPException(status_code=404, detail="Domain configuration not found")
    
    domain_config = domains[domain_name]
    
    request = CrawlRequest(
        urls=domain_config["urls"],
        max_depth=domain_config["max_depth"],
        force_refresh=force_refresh
    )
    
    return await start_crawl(request, background_tasks)


async def _perform_crawl(crawl_id: str, urls: List[str], max_depth: int, force_refresh: bool):
    """Perform the actual crawling in the background."""
    try:
        # Update status
        crawl_status[crawl_id].status = "running"
        crawl_status[crawl_id].message = "Crawling in progress"
        
        # Initialize crawler
        async with WebCrawler(urls, max_depth) as crawler:
            # Crawl documents
            new_documents = await crawler.crawl(force_refresh=force_refresh)
            
            crawl_status[crawl_id].total_documents = len(new_documents)
            crawl_status[crawl_id].message = f"Crawled {len(new_documents)} documents, indexing..."
            
            # Index documents
            if new_documents and indexer:
                updated_count = indexer.add_documents(new_documents, force_update=force_refresh)
                crawl_status[crawl_id].documents_processed = updated_count
                
                # Save crawler state
                state_file = settings.chroma_persist_directory / f"crawler_state_{crawl_id}.json"
                crawler.save_state(state_file)
        
        # Update final status
        crawl_status[crawl_id].status = "completed"
        crawl_status[crawl_id].message = f"Successfully processed {crawl_status[crawl_id].documents_processed} documents"
        
        logger.info(f"Crawl {crawl_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Crawl {crawl_id} failed: {e}")
        crawl_status[crawl_id].status = "failed"
        crawl_status[crawl_id].message = f"Crawl failed: {str(e)}"


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )