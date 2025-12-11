"""Tool for finding and navigating investor relations sites."""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Optional, ClassVar
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
import re
import logging

logger = logging.getLogger(__name__)


class FindIRSiteInput(BaseModel):
    """Input schema for finding investor relations site."""
    company_name: str = Field(description="Company name or ticker symbol to find IR site for")


class InvestorRelationsTool(BaseTool):
    """Tool for intelligently finding investor relations sites."""
    
    name: str = "find_investor_relations"
    description: str = """
    Find the investor relations website for a given company.
    This tool searches for and validates the official investor relations page.
    Input should be a company name (e.g., "Apple") or ticker symbol (e.g., "AAPL").
    """
    args_schema: Type[BaseModel] = FindIRSiteInput
    
    # Common IR URL patterns
    IR_PATTERNS: ClassVar[List[str]] = [
        "investor.{domain}",
        "investors.{domain}",
        "ir.{domain}",
        "{domain}/investor-relations",
        "{domain}/investors",
        "{domain}/ir",
    ]
    
    # Known IR URLs for major companies (fallback)
    KNOWN_IR_URLS: ClassVar[Dict[str, str]] = {
        # Tech companies
        "AAPL": "https://investor.apple.com",
        "APPLE": "https://investor.apple.com",
        "MSFT": "https://www.microsoft.com/en-us/investor",
        "MICROSOFT": "https://www.microsoft.com/en-us/investor",
        "GOOGL": "https://abc.xyz/investor/",
        "GOOG": "https://abc.xyz/investor/",
        "GOOGLE": "https://abc.xyz/investor/",
        "ALPHABET": "https://abc.xyz/investor/",
        "AMZN": "https://ir.aboutamazon.com",
        "AMAZON": "https://ir.aboutamazon.com",
        "META": "https://investor.fb.com",
        "FACEBOOK": "https://investor.fb.com",
        "NVDA": "https://investor.nvidia.com",
        "NVIDIA": "https://investor.nvidia.com",
        "TSLA": "https://ir.tesla.com",
        "TESLA": "https://ir.tesla.com",
        "NFLX": "https://ir.netflix.net",
        "NETFLIX": "https://ir.netflix.net",
        "AMD": "https://ir.amd.com",
        "INTC": "https://www.intc.com/investor-relations",
        "INTEL": "https://www.intc.com/investor-relations",
        "ORCL": "https://investor.oracle.com",
        "ORACLE": "https://investor.oracle.com",
        "CRM": "https://investor.salesforce.com",
        "SALESFORCE": "https://investor.salesforce.com",
        "ADBE": "https://www.adobe.com/investor-relations.html",
        "ADOBE": "https://www.adobe.com/investor-relations.html",
        # Other major companies
        "WMT": "https://stock.walmart.com",
        "WALMART": "https://stock.walmart.com",
        "JPM": "https://www.jpmorganchase.com/ir",
        "BAC": "https://investor.bankofamerica.com",
        "V": "https://investor.visa.com",
        "VISA": "https://investor.visa.com",
        "MA": "https://investor.mastercard.com",
        "MASTERCARD": "https://investor.mastercard.com",
        "JNJ": "https://www.investor.jnj.com",
        "PG": "https://www.pginvestor.com",
        "DIS": "https://www.thewaltdisneycompany.com/investors/",
        "DISNEY": "https://www.thewaltdisneycompany.com/investors/",
        "KO": "https://investors.coca-colacompany.com",
    }
    
    def _run(self, company_name: str) -> str:
        """Find investor relations site."""
        company_upper = company_name.upper().strip()
        
        # First, check if we have a known IR URL for this company
        if company_upper in self.KNOWN_IR_URLS:
            known_url = self.KNOWN_IR_URLS[company_upper]
            logger.info(f"Using known IR URL for {company_name}: {known_url}")
            
            # Verify the URL is accessible
            try:
                headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
                with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                    response = client.head(known_url, headers=headers)
                    if response.status_code < 400:
                        return f"""**Investor Relations Site Found for '{company_name}':**

**Official IR Site:** {known_url}

This is the official investor relations website where you can find:
- Latest earnings reports (10-K, 10-Q)
- Press releases and announcements
- Financial statements
- SEC filings
- Earnings call transcripts

You can use the extract_earnings_links tool to find specific documents on this site."""
            except Exception as e:
                logger.warning(f"Known URL {known_url} not accessible: {e}")
                # Continue to search fallback
        
        # Fallback to web search
        try:
            from ddgs import DDGS
            
            logger.info(f"Searching for IR site for {company_name}")
            
            # Search for investor relations page
            search_queries = [
                f"{company_name} investor relations",
                f"{company_name} IR earnings reports",
                f"{company_name} SEC filings quarterly earnings",
            ]
            
            all_results = []
            try:
                with DDGS() as ddgs:
                    for query in search_queries:
                        try:
                            results = list(ddgs.text(query, max_results=5))
                            all_results.extend(results)
                        except Exception as search_err:
                            logger.warning(f"Search query '{query}' failed: {search_err}")
                            continue
            except Exception as ddgs_err:
                logger.error(f"DuckDuckGo search failed: {ddgs_err}")
                return f"Unable to search for investor relations page for {company_name}. The search service is temporarily unavailable. You can try providing the direct investor relations URL if you know it."
            
            if not all_results:
                return f"No investor relations page found for {company_name}. Please try providing the company's official investor relations URL directly."
            
            # Score and rank results
            scored_results = []
            ir_keywords = ['investor', 'ir', 'shareholders', 'earnings', 'financial', 'sec', 'annual-report']
            
            for result in all_results:
                url = result.get('href', '')
                title = result.get('title', '').lower()
                body = result.get('body', '').lower()
                
                score = 0
                
                # Higher score for investor-related URLs
                url_lower = url.lower()
                for keyword in ir_keywords:
                    if keyword in url_lower:
                        score += 3
                    if keyword in title:
                        score += 2
                    if keyword in body:
                        score += 1
                
                # Bonus for official domains (not aggregator sites)
                if not any(agg in url_lower for agg in ['yahoo', 'google', 'marketwatch', 'bloomberg', 'reuters', 'wikipedia']):
                    score += 5
                
                # Bonus for .com domains (often official)
                if '.com/' in url_lower or url_lower.endswith('.com'):
                    score += 1
                
                scored_results.append({
                    'url': url,
                    'title': result.get('title', ''),
                    'body': result.get('body', ''),
                    'score': score
                })
            
            # Sort by score
            scored_results.sort(key=lambda x: x['score'], reverse=True)
            
            # Deduplicate by domain
            seen_domains = set()
            unique_results = []
            for result in scored_results:
                domain = urlparse(result['url']).netloc
                if domain not in seen_domains:
                    seen_domains.add(domain)
                    unique_results.append(result)
            
            # Format output
            output = f"**Investor Relations Sites Found for '{company_name}':**\n\n"
            
            for i, result in enumerate(unique_results[:5], 1):
                output += f"{i}. **{result['title']}**\n"
                output += f"   URL: {result['url']}\n"
                output += f"   {result['body'][:200]}...\n"
                output += f"   Relevance Score: {result['score']}\n\n"
            
            if unique_results:
                best = unique_results[0]
                output += f"\n**Recommended IR Site:** {best['url']}"
            
            return output
            
        except Exception as e:
            return f"Error finding investor relations site: {str(e)}"
    
    async def _arun(self, company_name: str) -> str:
        """Async execution."""
        return self._run(company_name)


class ExtractEarningsLinksInput(BaseModel):
    """Input schema for extracting earnings links."""
    ir_url: str = Field(description="URL of the investor relations page to extract earnings links from")


class ExtractEarningsLinksTool(BaseTool):
    """Tool for extracting earnings report links from an IR page."""
    
    name: str = "extract_earnings_links"
    description: str = """
    Extract links to earnings reports, 10-K, 10-Q filings, and other financial documents
    from an investor relations page. Use this after finding the IR site.
    """
    args_schema: Type[BaseModel] = ExtractEarningsLinksInput
    
    EARNINGS_KEYWORDS: ClassVar[List[str]] = [
        'earnings', 'quarterly', 'annual', '10-k', '10-q', '10k', '10q',
        'financial results', 'fiscal', 'q1', 'q2', 'q3', 'q4',
        'fy20', 'fy21', 'fy22', 'fy23', 'fy24',
        'annual report', 'sec filing', 'press release',
        'earnings call', 'transcript', 'presentation'
    ]
    
    def _run(self, ir_url: str) -> str:
        """Extract earnings-related links from IR page."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(ir_url, headers=headers)
                response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            earnings_links: List[Dict] = []
            
            # Find all links
            for link in soup.find_all('a', href=True):
                href = link['href']
                text = link.get_text(strip=True)
                
                # Skip empty or javascript links
                if not href or href.startswith('javascript:') or href == '#':
                    continue
                
                # Make absolute URL
                if not href.startswith('http'):
                    href = urljoin(ir_url, href)
                
                # Check if earnings-related
                combined = f"{href.lower()} {text.lower()}"
                
                relevance_score = 0
                matched_keywords = []
                
                for keyword in self.EARNINGS_KEYWORDS:
                    if keyword in combined:
                        relevance_score += 1
                        matched_keywords.append(keyword)
                
                # Bonus for PDF files
                if '.pdf' in href.lower():
                    relevance_score += 2
                
                if relevance_score > 0:
                    earnings_links.append({
                        'url': href,
                        'text': text[:100] if text else 'No text',
                        'score': relevance_score,
                        'keywords': matched_keywords,
                        'is_pdf': '.pdf' in href.lower()
                    })
            
            # Sort by relevance
            earnings_links.sort(key=lambda x: x['score'], reverse=True)
            
            # Deduplicate by URL
            seen_urls = set()
            unique_links = []
            for link in earnings_links:
                if link['url'] not in seen_urls:
                    seen_urls.add(link['url'])
                    unique_links.append(link)
            
            if not unique_links:
                return f"No earnings-related links found on {ir_url}. The page may use JavaScript to load content dynamically."
            
            # Format output
            output = f"**Earnings-Related Documents Found on {ir_url}:**\n\n"
            
            # Group by type
            pdfs = [l for l in unique_links if l['is_pdf']]
            others = [l for l in unique_links if not l['is_pdf']]
            
            if pdfs:
                output += "**PDF Documents:**\n"
                for i, link in enumerate(pdfs[:15], 1):
                    output += f"{i}. [{link['text']}]({link['url']})\n"
                    output += f"   Keywords: {', '.join(link['keywords'][:5])}\n\n"
            
            if others:
                output += "\n**Other Links:**\n"
                for i, link in enumerate(others[:10], 1):
                    output += f"{i}. [{link['text']}]({link['url']})\n"
            
            return output
            
        except Exception as e:
            return f"Error extracting earnings links: {str(e)}"
    
    async def _arun(self, ir_url: str) -> str:
        """Async execution."""
        return self._run(ir_url)

