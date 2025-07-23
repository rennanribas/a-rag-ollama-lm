"""Model Context Protocol (MCP) server implementation for AI RAG Agent."""

import asyncio
import sys
import os
from typing import Any, Dict, List

from mcp.server.fastmcp import FastMCP
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent import RAGAgent, QueryContext
from src.config import settings
from src.indexer import IncrementalIndexer
from src.crawler import WebCrawler


# Initialize FastMCP server
mcp = FastMCP("ai-rag-agent")

# Global components
indexer = None
agent = None

async def initialize_components():
    """Initialize the RAG components."""
    global indexer, agent
    try:
        logger.info("Initializing AI RAG Agent MCP Server...")
        indexer = IncrementalIndexer()
        agent = RAGAgent(indexer)
        logger.info("AI RAG Agent MCP Server initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {e}")
        raise
    
@mcp.tool()
async def query_rag(query: str, domain: str = None, session_id: str = "mcp") -> str:
    """Query the RAG agent with a question about the indexed documentation.
    
    Args:
        query: The question to ask the RAG agent
        domain: Optional domain context for the query
        session_id: Optional session ID for conversation context
    """
    global agent
    
    # Ensure components are initialized
    if not agent:
        await initialize_components()
    
    # Create query context
    query_context = QueryContext(
        query=query,
        domain=domain,
        session_id=session_id
    )
    
    # Execute query
    response = await agent.query(query_context)
    return response.answer


@mcp.tool()
async def crawl_docs(urls: List[str], max_depth: int = 2, force_refresh: bool = False) -> str:
    """Crawl and index documentation from URLs.
    
    Args:
        urls: List of URLs to crawl
        max_depth: Maximum crawl depth (default: 2)
        force_refresh: Force refresh all documents (default: false)
    """
    global indexer
    
    # Ensure indexer is initialized
    if not indexer:
        await initialize_components()
    
    # Perform crawl
    async with WebCrawler(urls, max_depth) as crawler:
        # Load previous state if exists
        state_file = settings.chroma_persist_directory / "crawler_state.json"
        crawler.load_state(state_file)
        
        # Crawl documents
        new_documents = await crawler.crawl(force_refresh=force_refresh)
        
        # Index documents
        if new_documents:
            updated_count = indexer.add_documents(new_documents, force_update=force_refresh)
            
            # Save state
            crawler.save_state(state_file)
            
            return f"Successfully crawled and indexed {updated_count} documents from {len(urls)} URLs"
        else:
            return "No new documents found to index"


@mcp.tool()
async def get_index_stats() -> str:
    """Get statistics about the current document index."""
    global indexer
    
    # Ensure indexer is initialized
    if not indexer:
        await initialize_components()
    
    stats = indexer.get_stats()
    
    return f"""Index Statistics:
- Total documents: {stats.get('total_documents', 0)}
- Total chunks: {stats.get('total_chunks', 0)}
- Index size: {stats.get('index_size', 'Unknown')}
- Last updated: {stats.get('last_updated', 'Unknown')}"""
    



if __name__ == "__main__":
    asyncio.run(initialize_components())
    mcp.run()