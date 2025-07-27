#!/usr/bin/env python3
"""Test script to verify crawler functionality."""

import asyncio
from src.crawler import WebCrawler

async def test_crawler():
    """Test if the crawler can fetch and extract content."""
    url = 'https://docs.nestjs.com/first-steps'
    crawler = WebCrawler([url])
    
    print(f"Testing crawler with URL: {url}")
    
    try:
        async with crawler:
            # Test the internal _fetch_page method directly
            result = await crawler._fetch_page(url)
            
            if result:
                print(f"Title: {result.title}")
                print(f"Content length: {len(result.content)}")
                print(f"Content preview: {result.content[:300]}...")
                print("✅ Crawler is working - content extracted successfully")
            else:
                print("❌ Crawler failed - no content extracted")
                
                # Let's debug by fetching raw content
                import aiohttp
                import ssl
                from bs4 import BeautifulSoup
                
                # Create SSL context that doesn't verify certificates
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                
                connector = aiohttp.TCPConnector(ssl=ssl_context)
                async with aiohttp.ClientSession(connector=connector) as session:
                    async with session.get(url) as response:
                        content = await response.text()
                        soup = BeautifulSoup(content, "html.parser")
                        
                        # Remove script and style elements
                        for script in soup(["script", "style", "nav", "footer", "header"]):
                            script.decompose()
                        
                        text_content = soup.get_text(separator="\n", strip=True)
                        lines = [line.strip() for line in text_content.split("\n") if line.strip()]
                        clean_content = "\n".join(lines)
                        
                        print(f"Raw content length: {len(clean_content)}")
                        print(f"Raw content preview: {clean_content[:300]}...")
            
    except Exception as e:
        print(f"❌ Crawler error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_crawler())