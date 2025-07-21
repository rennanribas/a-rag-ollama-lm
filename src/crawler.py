"""Web crawler for documentation and content extraction."""

import asyncio
import hashlib
import ssl
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import requests
from bs4 import BeautifulSoup
from loguru import logger
from pydantic import BaseModel

from .config import settings


class CrawledDocument(BaseModel):
    """Represents a crawled document with metadata."""
    url: str
    title: str
    content: str
    content_hash: str
    last_modified: Optional[str] = None
    crawled_at: float
    doc_type: str = "html"
    metadata: Dict = {}


class WebCrawler:
    """Asynchronous web crawler with incremental update support."""
    
    def __init__(self, base_urls: List[str], max_depth: int = 3):
        self.base_urls = base_urls
        self.max_depth = max_depth
        self.visited_urls: Set[str] = set()
        self.crawled_documents: Dict[str, CrawledDocument] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        # Configure SSL context
        ssl_context = None
        if not settings.verify_ssl:
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            logger.warning("SSL verification disabled - this is insecure but may be needed for some sites")
        
        connector = aiohttp.TCPConnector(
            limit=settings.max_concurrent_requests,
            ssl=ssl_context
        )
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {"User-Agent": settings.user_agent}
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content to detect changes."""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled."""
        parsed = urlparse(url)
        
        # Skip non-http(s) URLs
        if parsed.scheme not in ["http", "https"]:
            return False
            
        # Skip common file types we don't want
        skip_extensions = {".pdf", ".jpg", ".png", ".gif", ".css", ".js", ".ico"}
        if any(parsed.path.lower().endswith(ext) for ext in skip_extensions):
            return False
            
        # Only crawl URLs from base domains
        base_domains = {urlparse(base_url).netloc for base_url in self.base_urls}
        return parsed.netloc in base_domains
    
    async def _fetch_page(self, url: str) -> Optional[CrawledDocument]:
        """Fetch and parse a single page."""
        if not self.session:
            raise RuntimeError("Crawler not initialized. Use async context manager.")
            
        try:
            await asyncio.sleep(settings.crawl_delay)
            
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: HTTP {response.status}")
                    return None
                    
                content = await response.text()
                last_modified = response.headers.get("Last-Modified")
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
        
        # Parse content
        soup = BeautifulSoup(content, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
        
        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text().strip() if title_tag else url
        
        # Extract main content
        main_content = soup.find("main") or soup.find("article") or soup.find("div", class_="content")
        if main_content:
            text_content = main_content.get_text(separator="\n", strip=True)
        else:
            text_content = soup.get_text(separator="\n", strip=True)
        
        # Clean up text
        lines = [line.strip() for line in text_content.split("\n") if line.strip()]
        clean_content = "\n".join(lines)
        
        if len(clean_content) < 100:  # Skip pages with minimal content
            logger.debug(f"Skipping {url}: insufficient content")
            return None
        
        content_hash = self._get_content_hash(clean_content)
        
        return CrawledDocument(
            url=url,
            title=title,
            content=clean_content,
            content_hash=content_hash,
            last_modified=last_modified,
            crawled_at=time.time(),
            metadata={"content_length": len(clean_content)}
        )
    
    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract valid links from a page."""
        links = []
        
        for link in soup.find_all("a", href=True):
            href = link["href"]
            full_url = urljoin(base_url, href)
            
            if self._is_valid_url(full_url) and full_url not in self.visited_urls:
                links.append(full_url)
        
        return links
    
    async def crawl(self, force_refresh: bool = False) -> List[CrawledDocument]:
        """Crawl all configured URLs with incremental updates."""
        logger.info(f"Starting crawl of {len(self.base_urls)} base URLs")
        
        urls_to_crawl = list(self.base_urls)
        depth = 0
        new_documents = []
        
        while urls_to_crawl and depth < self.max_depth:
            logger.info(f"Crawling depth {depth + 1}: {len(urls_to_crawl)} URLs")
            
            # Process current batch
            tasks = []
            for url in urls_to_crawl:
                if url not in self.visited_urls:
                    self.visited_urls.add(url)
                    tasks.append(self._fetch_page(url))
            
            # Execute batch with concurrency limit
            semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
            
            async def bounded_fetch(task):
                async with semaphore:
                    return await task
            
            results = await asyncio.gather(
                *[bounded_fetch(task) for task in tasks],
                return_exceptions=True
            )
            
            # Process results and extract new links
            next_urls = []
            
            for result in results:
                if isinstance(result, CrawledDocument):
                    # Check if document changed or is new
                    existing = self.crawled_documents.get(result.url)
                    
                    if (force_refresh or 
                        not existing or 
                        existing.content_hash != result.content_hash):
                        
                        self.crawled_documents[result.url] = result
                        new_documents.append(result)
                        logger.info(f"Updated document: {result.title} ({result.url})")
                        
                        # Extract links for next depth level
                        if depth < self.max_depth - 1:
                            try:
                                response = requests.get(
                                    result.url, 
                                    timeout=10,
                                    verify=settings.verify_ssl
                                )
                                soup = BeautifulSoup(response.content, "html.parser")
                                links = self._extract_links(soup, result.url)
                                next_urls.extend(links)
                            except Exception as e:
                                logger.warning(f"Failed to extract links from {result.url}: {e}")
                    else:
                        logger.debug(f"No changes detected: {result.url}")
                        
                elif isinstance(result, Exception):
                    logger.error(f"Crawl error: {result}")
            
            urls_to_crawl = list(set(next_urls))
            depth += 1
        
        logger.info(f"Crawl completed. {len(new_documents)} new/updated documents")
        return new_documents
    
    def get_all_documents(self) -> List[CrawledDocument]:
        """Get all crawled documents."""
        return list(self.crawled_documents.values())
    
    def save_state(self, file_path: Path):
        """Save crawler state to file."""
        import json
        
        state = {
            "visited_urls": list(self.visited_urls),
            "documents": {url: doc.dict() for url, doc in self.crawled_documents.items()}
        }
        
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, file_path: Path):
        """Load crawler state from file."""
        import json
        
        if not file_path.exists():
            return
            
        with open(file_path, "r") as f:
            state = json.load(f)
        
        self.visited_urls = set(state.get("visited_urls", []))
        
        documents = state.get("documents", {})
        self.crawled_documents = {
            url: CrawledDocument(**doc_data) 
            for url, doc_data in documents.items()
        }