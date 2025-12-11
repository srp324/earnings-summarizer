"""Web search tool for finding company investor relations pages."""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from ddgs import DDGS
import httpx


class WebSearchInput(BaseModel):
    """Input schema for web search."""
    query: str = Field(description="The search query to find information about a company")


class WebSearchTool(BaseTool):
    """Tool for searching the web using DuckDuckGo."""
    
    name: str = "web_search"
    description: str = """
    Search the web for information about companies and their investor relations pages.
    Use this to find the official investor relations website for a given company.
    Input should be a search query like "Apple investor relations" or "AAPL earnings reports".
    """
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str) -> str:
        """Execute web search."""
        try:
            # Try with retry logic
            max_retries = 2
            results = []
            
            for attempt in range(max_retries):
                try:
                    with DDGS() as ddgs:
                        results = list(ddgs.text(query, max_results=10))
                        break  # Success, exit retry loop
                except Exception as ddgs_err:
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)  # Wait before retry
                        continue
                    else:
                        raise ddgs_err
                
            if not results:
                return "No search results found."
            
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"{i}. **{result.get('title', 'No title')}**\n"
                    f"   URL: {result.get('href', 'No URL')}\n"
                    f"   {result.get('body', 'No description')[:200]}..."
                )
            
            return "\n\n".join(formatted_results)
        except Exception as e:
            return f"Search error: {str(e)}. The search service may be temporarily unavailable."
    
    async def _arun(self, query: str) -> str:
        """Async execution of web search."""
        return self._run(query)


class URLFetchInput(BaseModel):
    """Input schema for URL fetching."""
    url: str = Field(description="The URL to fetch content from")


class URLFetchTool(BaseTool):
    """Tool for fetching and extracting content from URLs."""
    
    name: str = "fetch_url"
    description: str = """
    Fetch the content of a webpage URL. Use this to examine the contents
    of an investor relations page to find links to earnings reports.
    """
    args_schema: Type[BaseModel] = URLFetchInput
    
    def _run(self, url: str) -> str:
        """Fetch URL content."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Extract text
            text = soup.get_text(separator='\n', strip=True)
            
            # Extract links that might be relevant to earnings
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                text_content = link.get_text(strip=True)
                # Look for earnings-related links
                keywords = ['earnings', 'annual', 'quarterly', '10-k', '10-q', 
                           'report', 'financial', 'sec', 'filing', 'pdf']
                if any(kw in href.lower() or kw in text_content.lower() for kw in keywords):
                    if not href.startswith('http'):
                        from urllib.parse import urljoin
                        href = urljoin(url, href)
                    links.append(f"- [{text_content}]({href})")
            
            result = f"**Page Content Summary:**\n{text[:3000]}...\n\n"
            if links:
                result += f"**Relevant Links Found ({len(links)}):**\n" + "\n".join(links[:20])
            
            return result
        except Exception as e:
            return f"Error fetching URL: {str(e)}"
    
    async def _arun(self, url: str) -> str:
        """Async URL fetch."""
        return self._run(url)

