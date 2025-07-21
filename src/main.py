"""Main CLI interface for the AI RAG Agent."""

import asyncio
import sys
from pathlib import Path
from typing import List, Optional

import click
import uvicorn
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .agent import RAGAgent, QueryContext
from .config import settings
from .crawler import WebCrawler
from .indexer import IncrementalIndexer


console = Console()


@click.group()
@click.option('--log-level', default='INFO', help='Set logging level')
def cli(log_level: str):
    """AI RAG Agent - Domain-specific documentation assistant."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level.upper(),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    if settings.log_file:
        logger.add(
            settings.log_file,
            level=log_level.upper(),
            rotation="10 MB",
            retention="7 days"
        )


@cli.command()
@click.option('--host', default='localhost', help='API host')
@click.option('--port', default=8000, help='API port')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the RAG agent API server."""
    console.print(f"Starting AI RAG Agent API server on {host}:{port}", style="bold green")
    
    uvicorn.run(
        "src.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=settings.log_level.lower()
    )


@cli.command()
@click.argument('urls', nargs=-1, required=True)
@click.option('--max-depth', default=2, help='Maximum crawl depth')
@click.option('--force-refresh', is_flag=True, help='Force refresh all documents')
def crawl(urls: List[str], max_depth: int, force_refresh: bool):
    """Crawl documentation URLs and build the index."""
    asyncio.run(_crawl_async(list(urls), max_depth, force_refresh))


async def _crawl_async(urls: List[str], max_depth: int, force_refresh: bool):
    """Async crawl implementation."""
    console.print(f"Starting crawl of {len(urls)} URLs (max depth: {max_depth})", style="bold blue")
    
    # Initialize components
    indexer = IncrementalIndexer()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        crawl_task = progress.add_task("Crawling documents...", total=None)
        
        try:
            async with WebCrawler(urls, max_depth) as crawler:
                # Load previous state if exists
                state_file = settings.chroma_persist_directory / "crawler_state.json"
                crawler.load_state(state_file)
                
                # Perform crawl
                new_documents = await crawler.crawl(force_refresh=force_refresh)
                
                progress.update(crawl_task, description=f"Crawled {len(new_documents)} documents, indexing...")
                
                # Index documents
                if new_documents:
                    updated_count = indexer.add_documents(new_documents, force_update=force_refresh)
                    
                    progress.update(crawl_task, description=f"Indexed {updated_count} documents")
                    
                    # Save state
                    crawler.save_state(state_file)
                    
                    console.print(f"Successfully processed {updated_count} documents", style="bold green")
                else:
                    console.print("No new documents found", style="yellow")
                
                # Show stats
                stats = indexer.get_stats()
                console.print(f"Total documents in index: {stats.get('total_documents', 0)}", style="cyan")
                
        except Exception as e:
            console.print(f"Crawl failed: {e}", style="bold red")
            raise


@cli.command()
@click.argument('query')
@click.option('--domain', help='Specify domain context')
@click.option('--session-id', help='Session ID for conversation context')
def query(query: str, domain: Optional[str], session_id: Optional[str]):
    """Query the RAG agent interactively."""
    asyncio.run(_query_async(query, domain, session_id))


async def _query_async(query: str, domain: Optional[str], session_id: Optional[str]):
    """Async query implementation."""
    console.print(f"Processing query: {query}", style="bold blue")
    
    try:
        # Initialize components
        indexer = IncrementalIndexer()
        agent = RAGAgent(indexer)
        
        # Create query context
        query_context = QueryContext(
            query=query,
            domain=domain,
            session_id=session_id or "cli"
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Analyzing query and retrieving documents...", total=None)
            
            # Process query
            response = await agent.process_query(query_context)
            
            progress.update(task, description="Generating response...")
        
        # Display results
        console.print("\n" + "="*80, style="cyan")
        console.print("RESPONSE", style="bold cyan")
        console.print("="*80, style="cyan")
        console.print(response.answer)
        
        # Show confidence and sources
        console.print(f"\nConfidence: {response.confidence:.2f}", style="yellow")
        
        if response.sources:
            console.print(f"\nSources ({len(response.sources)}):", style="bold yellow")
            for i, source in enumerate(response.sources[:3], 1):
                console.print(f"  {i}. {source.get('title', 'Unknown')} (Score: {source.get('score', 0):.2f})")
                console.print(f"     {source.get('url', 'Unknown URL')}", style="dim")
        
        if response.suggestions:
            console.print(f"\nSuggestions:", style="bold green")
            for suggestion in response.suggestions:
                console.print(f"  - {suggestion}")
        
        console.print("\n" + "="*80, style="cyan")
        
    except Exception as e:
        console.print(f"Query failed: {e}", style="bold red")
        raise


@cli.command()
def interactive():
    """Start interactive query mode."""
    asyncio.run(_interactive_async())


async def _interactive_async():
    """Async interactive mode implementation."""
    console.print("AI RAG Agent - Interactive Mode", style="bold green")
    console.print("Type 'quit' or 'exit' to stop, 'clear' to clear conversation history\n")
    
    # Initialize components
    indexer = IncrementalIndexer()
    agent = RAGAgent(indexer)
    session_id = "interactive"
    
    while True:
        try:
            query = console.input("[bold blue]Query: [/bold blue]")
            
            if query.lower() in ['quit', 'exit']:
                console.print("👋 Goodbye!", style="bold green")
                break
            
            if query.lower() == 'clear':
                agent.clear_conversation_history(session_id)
                console.print("🧹 Conversation history cleared\n", style="yellow")
                continue
            
            if not query.strip():
                continue
            
            # Process query
            query_context = QueryContext(query=query, session_id=session_id)
            response = await agent.process_query(query_context)
            
            # Display response
            console.print(f"\n[bold green]Assistant:[/bold green] {response.answer}")
            console.print(f"[dim]Confidence: {response.confidence:.2f}[/dim]\n")
            
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!", style="bold green")
            break
        except Exception as e:
            console.print(f"❌ Error: {e}", style="bold red")


@cli.command()
def status():
    """Show system status and statistics."""
    try:
        indexer = IncrementalIndexer()
        stats = indexer.get_stats()
        
        # Create status table
        table = Table(title="AI RAG Agent Status", show_header=True, header_style="bold magenta")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details")
        
        table.add_row("Indexer", "✅ Ready", f"Collection: {stats.get('collection_name', 'Unknown')}")
        table.add_row("Documents", "📊 Indexed", f"Total: {stats.get('total_documents', 0)}")
        table.add_row("Storage", "💾 Available", f"Path: {stats.get('storage_directory', 'Unknown')}")
        
        console.print(table)
        
    except Exception as e:
        console.print(f"❌ Status check failed: {e}", style="bold red")


@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear all indexed documents?')
def clear():
    """Clear all indexed documents."""
    try:
        indexer = IncrementalIndexer()
        indexer.clear_index()
        console.print("🧹 All documents cleared successfully", style="bold green")
    except Exception as e:
        console.print(f"❌ Clear failed: {e}", style="bold red")


@cli.command()
def domains():
    """List predefined domain configurations."""
    domains_config = {
        "apple_ios": {
            "name": "Apple iOS Development",
            "urls": [
                "https://developer.apple.com/documentation/uikit",
                "https://developer.apple.com/documentation/swiftui",
                "https://developer.apple.com/documentation/foundation"
            ]
        },
        "apple_watchos": {
            "name": "Apple watchOS Development",
            "urls": [
                "https://developer.apple.com/documentation/watchkit",
                "https://developer.apple.com/documentation/healthkit",
                "https://developer.apple.com/documentation/workoutkit"
            ]
        },
        "swift": {
            "name": "Swift Language",
            "urls": [
                "https://docs.swift.org/swift-book/",
                "https://developer.apple.com/documentation/swift"
            ]
        }
    }
    
    table = Table(title="Available Domain Configurations", show_header=True, header_style="bold magenta")
    table.add_column("Domain", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("URLs", style="yellow")
    
    for domain_id, config in domains_config.items():
        urls_str = "\n".join(config["urls"][:2])  # Show first 2 URLs
        if len(config["urls"]) > 2:
            urls_str += f"\n... and {len(config['urls']) - 2} more"
        
        table.add_row(domain_id, config["name"], urls_str)
    
    console.print(table)
    console.print("\n💡 Use: [bold]ai-rag crawl-domain <domain_name>[/bold] to crawl a specific domain")


@cli.command()
@click.argument('domain_name')
@click.option('--force-refresh', is_flag=True, help='Force refresh all documents')
def crawl_domain(domain_name: str, force_refresh: bool):
    """Crawl a predefined domain configuration."""
    domains_config = {
        "apple_ios": {
            "urls": [
                "https://developer.apple.com/documentation/uikit",
                "https://developer.apple.com/documentation/swiftui",
                "https://developer.apple.com/documentation/foundation",
                "https://developer.apple.com/documentation/swift"
            ],
            "max_depth": 3
        },
        "apple_watchos": {
            "urls": [
                "https://developer.apple.com/documentation/watchkit",
                "https://developer.apple.com/documentation/healthkit",
                "https://developer.apple.com/documentation/workoutkit",
                "https://developer.apple.com/watchos/"
            ],
            "max_depth": 3
        },
        "swift": {
            "urls": [
                "https://docs.swift.org/swift-book/",
                "https://developer.apple.com/documentation/swift"
            ],
            "max_depth": 4
        }
    }
    
    if domain_name not in domains_config:
        console.print(f"❌ Domain '{domain_name}' not found", style="bold red")
        console.print("Available domains: " + ", ".join(domains_config.keys()), style="yellow")
        return
    
    config = domains_config[domain_name]
    console.print(f"🚀 Crawling domain: {domain_name}", style="bold green")
    
    asyncio.run(_crawl_async(config["urls"], config["max_depth"], force_refresh))


if __name__ == "__main__":
    cli()