"""Tool for retrieving earnings transcripts by scraping discountingcashflows.com."""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List, Dict, Optional, Tuple
import logging
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Base URL for discounting cash flows transcripts
DCF_BASE_URL = "https://discountingcashflows.com"


class TranscriptListInput(BaseModel):
    """Input schema for listing earnings transcripts."""
    symbol: str = Field(description="Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'NVDA')")


class TranscriptListTool(BaseTool):
    """Tool for retrieving list of available earnings transcripts by scraping discountingcashflows.com."""
    
    name: str = "list_earnings_transcripts"
    description: str = """
    Get a list of available earnings call transcripts for a stock symbol by scraping discountingcashflows.com.
    Input should be a stock ticker symbol (e.g., "AAPL" for Apple, "NVDA" for NVIDIA, "MSFT" for Microsoft).
    Returns a list of available transcripts with their fiscal years, quarters, and dates.
    Use the fiscal year, quarter, and date with get_earnings_transcript to retrieve the full transcript.
    """
    args_schema: Type[BaseModel] = TranscriptListInput
    
    def _run(self, symbol: str) -> str:
        """List available earnings transcripts for a symbol."""
        try:
            symbol_upper = symbol.upper().strip()
            logger.info(f"Fetching earnings transcripts list for {symbol_upper} from discountingcashflows.com")
            
            # Construct URL for the transcripts page
            url = f"{DCF_BASE_URL}/company/{symbol_upper}/transcripts/"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find all transcript links - looking for the structure from the website
            # Transcripts are organized by fiscal year and quarter
            transcripts = []
            
            # Find all elements that might contain transcript information
            # Looking for fiscal year headers and quarter links
            transcript_sections = soup.find_all(['div', 'section'], class_=re.compile(r'transcript|quarter|fiscal', re.I))
            
            # Also look for links that contain quarter/year information
            all_links = soup.find_all('a', href=True)
            
            # Pattern to match transcript links or sections
            for link in all_links:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                
                # Check if this looks like a transcript link
                if 'transcript' in href.lower() or 'Q' in text or re.search(r'Q[1-4]', text) or re.search(r'FY \d{4}', text):
                    # Extract fiscal year and quarter from text or href
                    year_match = re.search(r'FY\s*(\d{4})', text, re.I)
                    quarter_match = re.search(r'Q([1-4])', text, re.I)
                    
                    if year_match or quarter_match:
                        year = year_match.group(1) if year_match else None
                        quarter = quarter_match.group(1) if quarter_match else None
                        
                        # Try to extract date from nearby text
                        date = None
                        parent = link.parent
                        if parent:
                            parent_text = parent.get_text()
                            date_match = re.search(r'(\w{3}\s+\d{1,2})', parent_text)
                            if date_match:
                                date = date_match.group(1)
                        
                        transcripts.append({
                            'symbol': symbol_upper,
                            'year': year,
                            'quarter': quarter,
                            'date': date,
                            'text': text,
                            'href': urljoin(url, href) if href else None
                        })
            
            # If we didn't find links, try parsing the page structure directly
            # Based on the website structure showing FY and Q sections
            if not transcripts:
                # Look for fiscal year sections
                for section in soup.find_all(['div', 'section', 'ul', 'li']):
                    section_text = section.get_text()
                    year_match = re.search(r'FY\s*(\d{4})', section_text, re.I)
                    
                    if year_match:
                        year = year_match.group(1)
                        # Look for quarters in this section
                        for item in section.find_all(['li', 'div', 'a']):
                            item_text = item.get_text(strip=True)
                            quarter_match = re.search(r'Q([1-4])', item_text, re.I)
                            date_match = re.search(r'(\w{3}\s+\d{1,2})', item_text)
                            
                            if quarter_match:
                                quarter = quarter_match.group(1)
                                date = date_match.group(1) if date_match else None
                                
                                href = None
                                link = item.find('a', href=True)
                                if link:
                                    href = urljoin(url, link['href'])
                                
                                transcripts.append({
                                    'symbol': symbol_upper,
                                    'year': year,
                                    'quarter': quarter,
                                    'date': date,
                                    'text': item_text,
                                    'href': href
                                })
            
            # Remove duplicates
            seen = set()
            unique_transcripts = []
            for t in transcripts:
                key = (t.get('year'), t.get('quarter'))
                if key not in seen and key != (None, None):
                    seen.add(key)
                    unique_transcripts.append(t)
            
            if not unique_transcripts:
                return f"No earnings transcripts found for symbol '{symbol_upper}' on discountingcashflows.com. This could mean:\n1. The symbol is invalid\n2. The company doesn't have transcripts available\n3. The website structure may have changed\n\nYou can check manually at: {url}"
            
            # Sort by year and quarter (most recent first)
            def sort_key(t):
                year = int(t.get('year', 0)) if t.get('year') and t.get('year').isdigit() else 0
                quarter = int(t.get('quarter', 0)) if t.get('quarter') else 0
                return (year, quarter)
            
            unique_transcripts.sort(key=sort_key, reverse=True)
            
            # Format the output
            output = f"**Earnings Call Transcripts Available for {symbol_upper}:**\n\n"
            output += f"Found {len(unique_transcripts)} transcript(s) on discountingcashflows.com.\n\n"
            
            # Display transcripts (limit to 20 most recent)
            for i, transcript in enumerate(unique_transcripts[:20], 1):
                year = transcript.get('year', 'N/A')
                quarter = transcript.get('quarter', 'N/A')
                date = transcript.get('date', 'N/A')
                href = transcript.get('href', '')
                
                output += f"{i}. **FY {year} Q{quarter}**\n"
                if date and date != 'N/A':
                    output += f"   Date: {date}\n"
                if href:
                    output += f"   Link: {href}\n"
                output += "\n"
            
            if len(unique_transcripts) > 20:
                output += f"\n_(Showing 20 most recent out of {len(unique_transcripts)} total transcripts)_\n"
            
            output += "\n**Next Step:** Use the `get_earnings_transcript` tool with the symbol, fiscal year, and quarter to retrieve the full transcript content."
            
            return output
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching transcripts list: {str(e)}")
            return f"**Network Error**\n\nFailed to fetch transcripts from discountingcashflows.com: {str(e)}\n\nPlease check your internet connection and try again."
            
        except Exception as e:
            logger.error(f"Error parsing transcripts list: {str(e)}")
            return f"**Error retrieving earnings transcripts**\n\n{str(e)}\n\nPlease ensure:\n1. The ticker symbol is valid (e.g., 'NVDA' not 'NVIDIA')\n2. The website is accessible\n3. The website structure hasn't changed significantly"
    
    async def _arun(self, symbol: str) -> str:
        """Async execution."""
        return self._run(symbol)


class TranscriptInput(BaseModel):
    """Input schema for retrieving a specific earnings transcript."""
    symbol: str = Field(description="Stock ticker symbol (e.g., 'AAPL')")
    fiscal_year: Optional[str] = Field(default=None, description="Fiscal year (e.g., '2025' or '2024'). If not provided, will use the most recent transcript.")
    quarter: Optional[str] = Field(default=None, description="Quarter number (1, 2, 3, or 4). If not provided, will use the most recent transcript.")


def _get_most_recent_transcript(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """Helper function to get the most recent transcript's fiscal_year and quarter."""
    try:
        symbol_upper = symbol.upper().strip()
        list_url = f"{DCF_BASE_URL}/company/{symbol_upper}/transcripts/"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(list_url, headers=headers, timeout=30)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for the transcripts list structure: ul.transcripts-list
        transcripts = []
        
        # Method 1: Parse the transcripts-list structure (ul with class transcripts-list)
        transcript_list = soup.find('ul', class_=re.compile(r'transcripts-list', re.I))
        if transcript_list:
            current_year = None
            # Iterate through li elements in order (most recent should be first)
            for li_idx, li in enumerate(transcript_list.find_all('li', recursive=False)):
                # Check for h2 with fiscal year (e.g., "FY 2026")
                h2 = li.find('h2', class_=re.compile(r'menu-title', re.I))
                if h2:
                    year_match = re.search(r'FY\s*(\d{4})', h2.get_text(strip=True), re.I)
                    if year_match:
                        current_year = year_match.group(1)
                        logger.info(f"Found fiscal year: {current_year} (at position {li_idx})")
                
                # Look for links within this li that contain quarter info
                # The first link in the first fiscal year section is likely the most recent
                if current_year:
                    links = li.find_all('a', href=True)
                    for link_idx, link in enumerate(links):
                        href = link.get('href', '')
                        text = link.get_text(strip=True)
                        
                        # Extract quarter from URL: /transcripts/{year}/{quarter}/
                        url_match = re.search(r'/transcripts/(\d{4})/(\d{1})/', href)
                        if url_match:
                            year = url_match.group(1)
                            quarter = url_match.group(2)
                            if year == current_year:  # Only add if it matches the current fiscal year section
                                transcripts.append({
                                    'year': year,
                                    'quarter': quarter,
                                    'href': href,
                                    'order': li_idx * 100 + link_idx  # Preserve order (lower = earlier in HTML = more recent)
                                })
                                #logger.info(f"Found transcript: FY{year} Q{quarter} from URL: {href} (order: {li_idx * 100 + link_idx})")
                        else:
                            # Extract quarter from text (e.g., "Q3", "Q 3")
                            quarter_match = re.search(r'Q\s*([1-4])', text, re.I)
                            if quarter_match:
                                quarter = quarter_match.group(1)
                                transcripts.append({
                                    'year': current_year,
                                    'quarter': quarter,
                                    'href': href,
                                    'order': li_idx * 100 + link_idx
                                })
                                logger.info(f"Found transcript: FY{current_year} Q{quarter} from text: {text} (order: {li_idx * 100 + link_idx})")
        
        # Method 2: Fallback - find all links with transcript URLs
        if not transcripts:
            for idx, link in enumerate(soup.find_all('a', href=True)):
                href = link.get('href', '')
                # Extract from URL pattern: /company/{symbol}/transcripts/{year}/{quarter}/
                url_match = re.search(r'/transcripts/(\d{4})/(\d{1})/', href)
                if url_match:
                    year = url_match.group(1)
                    quarter = url_match.group(2)
                    transcripts.append({
                        'year': year,
                        'quarter': quarter,
                        'href': href,
                        'order': idx  # Preserve order from HTML
                    })
        
        # Remove duplicates
        seen = set()
        unique_transcripts = []
        for t in transcripts:
            key = (t.get('year'), t.get('quarter'))
            if key not in seen and key != (None, None):
                seen.add(key)
                unique_transcripts.append(t)
        
        if not unique_transcripts:
            logger.warning("No transcripts found in the expected structure")
            return None, None
        
        # Sort by year DESC then quarter DESC (most recent first)
        # Higher year = more recent, and within same year, higher quarter = more recent
        # Also consider HTML order as a tiebreaker (lower order number = appeared earlier = more recent)
        def sort_key(t):
            year = int(t.get('year', 0)) if t.get('year') and t.get('year').isdigit() else 0
            quarter = int(t.get('quarter', 0)) if t.get('quarter') and t.get('quarter').isdigit() else 0
            order = t.get('order', 999999)  # Default to high number if no order
            return (-year, -quarter, order)  # Negative for descending order on year/quarter, ascending on order
        
        unique_transcripts.sort(key=sort_key)
        
        # Log all transcripts found for debugging
        logger.info(f"Found {len(unique_transcripts)} unique transcripts:")
        for t in unique_transcripts[:5]:  # Log first 5
            logger.info(f"  - FY{t.get('year')} Q{t.get('quarter')}")
        
        # Return ONLY the most recent one (first after sorting)
        if unique_transcripts:
            most_recent = unique_transcripts[0]
            logger.info(f"Most recent transcript selected: FY{most_recent.get('year')} Q{most_recent.get('quarter')}")
            return most_recent.get('year'), most_recent.get('quarter')
        
        return None, None
        
    except Exception as e:
        logger.error(f"Error getting most recent transcript: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None


class TranscriptTool(BaseTool):
    """Tool for retrieving the full content of an earnings transcript by scraping discountingcashflows.com."""
    
    name: str = "get_earnings_transcript"
    description: str = """
    Retrieve the full content of an earnings call transcript using symbol, fiscal year, and quarter.
    If fiscal_year or quarter is not provided, will automatically use the most recent available transcript.
    
    IMPORTANT: When the user specifies a fiscal year and quarter (e.g., "FY2025Q2", "FY 2025 Q2", "2025 Q2", "2022Q2"), 
    you MUST extract and pass them as separate parameters:
    - fiscal_year: Extract the year (e.g., "2025" from "FY2025Q2")
    - quarter: Extract the quarter number (e.g., "2" from "FY2025Q2")
    
    Input parameters:
    - symbol: Stock ticker (e.g., "NVDA", "AAPL", "MSFT")
    - fiscal_year: (Optional) Fiscal year as string (e.g., "2025", "2024"). Extract from formats like "FY2025", "2025", "FY 2025". If not provided, uses most recent.
    - quarter: (Optional) Quarter number as string "1", "2", "3", or "4". Extract from formats like "Q2", "Q 2", "2". If not provided, uses most recent.
    
    Returns the complete earnings call transcript with all speakers and content.
    """
    args_schema: Type[BaseModel] = TranscriptInput
    
    def _run(self, symbol: str, fiscal_year: Optional[str] = None, quarter: Optional[str] = None) -> str:
        """Retrieve full earnings transcript content."""
        try:
            symbol_upper = symbol.upper().strip()
            
            # Log what we received
            logger.info(f"_run received: symbol={symbol_upper}, fiscal_year={fiscal_year}, quarter={quarter}")
            
            # Handle string 'None' case - but be careful not to override valid values
            if fiscal_year:
                fiscal_year_clean = fiscal_year.strip()
                if fiscal_year_clean.lower() in ('none', 'null', ''):
                    fiscal_year = None
                else:
                    fiscal_year = fiscal_year_clean
            if quarter:
                quarter_clean = quarter.strip()
                if quarter_clean.lower() in ('none', 'null', ''):
                    quarter = None
                else:
                    quarter = quarter_clean
            
            # Try to extract fiscal_year and quarter from combined formats like "FY2025Q2", "FY 2025 Q2", "2025Q2"
            # This handles cases where the LLM might pass them in a combined format
            if fiscal_year and not quarter:
                # Check if fiscal_year contains both year and quarter (e.g., "FY2025Q2", "2025Q2")
                combined_match = re.search(r'FY?\s*(\d{4})\s*Q\s*([1-4])', fiscal_year, re.I)
                if combined_match:
                    fiscal_year = combined_match.group(1)
                    quarter = combined_match.group(2)
                    logger.info(f"Extracted from combined format: FY{fiscal_year} Q{quarter}")
            
            # If fiscal_year or quarter is still missing, get the most recent transcript
            if not fiscal_year or not quarter:
                logger.info(f"fiscal_year or quarter not provided for {symbol_upper}, fetching most recent transcript...")
                most_recent_year, most_recent_quarter = _get_most_recent_transcript(symbol_upper)
                if most_recent_year and most_recent_quarter:
                    fiscal_year = fiscal_year or most_recent_year
                    quarter = quarter or most_recent_quarter
                    logger.info(f"Using most recent transcript: FY{fiscal_year} Q{quarter}")
                else:
                    return f"**Error:** Could not find any available transcripts for {symbol_upper}. Please use list_earnings_transcripts to see available transcripts."
            
            logger.info(f"Fetching earnings transcript for {symbol_upper} FY{fiscal_year} Q{quarter} from discountingcashflows.com")
            
            # First, try to get the transcripts list page to find the specific transcript link
            list_url = f"{DCF_BASE_URL}/company/{symbol_upper}/transcripts/"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            
            response = requests.get(list_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the link for the specific transcript
            transcript_url = None
            
            # Normalize the search terms
            fiscal_year_normalized = fiscal_year.strip()
            quarter_normalized = quarter.strip()
            
            # Look for links matching the fiscal year and quarter
            # Try multiple matching strategies
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                text = link.get_text(strip=True)
                link_html = str(link)
                
                # Strategy 1: Match both year and quarter in text
                year_patterns = [
                    rf'FY\s*{fiscal_year_normalized}',
                    rf'Fiscal\s+Year\s+{fiscal_year_normalized}',
                    fiscal_year_normalized
                ]
                quarter_patterns = [
                    rf'Q{quarter_normalized}\b',
                    rf'Quarter\s+{quarter_normalized}',
                    quarter_normalized
                ]
                
                year_match = any(re.search(pattern, text, re.I) or pattern in text for pattern in year_patterns)
                quarter_match = any(re.search(pattern, text, re.I) or pattern in text for pattern in quarter_patterns)
                
                # Also check the href itself
                if not year_match or not quarter_match:
                    year_match = year_match or any(pattern in href.upper() for pattern in [f'FY{fiscal_year_normalized}', fiscal_year_normalized])
                    quarter_match = quarter_match or f'Q{quarter_normalized}' in href.upper() or quarter_normalized in href.upper()
                
                # Check parent or sibling elements for context
                if not year_match or not quarter_match:
                    parent = link.parent
                    if parent:
                        parent_text = parent.get_text()
                        year_match = year_match or any(re.search(pattern, parent_text, re.I) for pattern in year_patterns)
                        quarter_match = quarter_match or any(re.search(pattern, parent_text, re.I) for pattern in quarter_patterns)
                
                if year_match and quarter_match:
                    # Build full URL
                    if href.startswith('http'):
                        transcript_url = href
                    elif href.startswith('/'):
                        transcript_url = f"{DCF_BASE_URL}{href}"
                    else:
                        transcript_url = urljoin(list_url, href)
                    logger.info(f"Found transcript URL: {transcript_url}")
                    break
            
            # If we couldn't find a specific link, try constructing a URL pattern
            # This is a fallback in case the page structure is different
            if not transcript_url:
                # The website might use a specific URL pattern - we'll try a few common ones
                possible_patterns = [
                    f"{DCF_BASE_URL}/company/{symbol_upper}/transcripts/{fiscal_year}/q{quarter}",
                    f"{DCF_BASE_URL}/company/{symbol_upper}/transcripts/fy{fiscal_year}q{quarter}",
                    f"{DCF_BASE_URL}/company/{symbol_upper}/transcript/{fiscal_year}/{quarter}",
                ]
                
                for pattern_url in possible_patterns:
                    try:
                        test_response = requests.get(pattern_url, headers=headers, timeout=10)
                        if test_response.status_code == 200:
                            transcript_url = pattern_url
                            break
                    except Exception:
                        continue
            
            if not transcript_url:
                return f"**Transcript Not Found**\n\nCould not find a transcript link for {symbol_upper} FY{fiscal_year} Q{quarter}.\n\nPlease verify:\n1. The fiscal year and quarter are correct\n2. Use list_earnings_transcripts to see available transcripts\n3. Check manually at: {list_url}"
            
            # Fetch the transcript page
            transcript_response = requests.get(transcript_url, headers=headers, timeout=30)
            transcript_response.raise_for_status()
            
            # Check if the page is mostly empty (likely JavaScript-rendered)
            if len(transcript_response.text) < 5000:
                # If body is too small, this is likely a JS-rendered page
                # Return early and let async _arun handle it with Playwright
                logger.info(f"Page appears to be JavaScript-rendered (only {len(transcript_response.text)} bytes). Playwright will be used in async context.")
                raise ValueError("Page requires JavaScript rendering")
            
            transcript_soup = BeautifulSoup(transcript_response.text, 'html.parser')
            
            # Extract transcript content
            # Use multiple strategies to find the transcript content
            transcript_content = ""
            
            # Strategy 1: Remove unwanted elements first
            for element in transcript_soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
                element.decompose()
            
            # Also remove elements with common navigation/header classes and IDs
            unwanted_patterns = [
                r'menu|navigation|nav',
                r'header',
                r'footer',
                r'sidebar',
                r'cookie|consent',
                r'social|share',
                r'ad|advertisement',
                r'market.*quote|quote.*market',  # Market quotes/widgets
                r'after.*hours|after-hours',
                r'stock.*price|price.*stock',
            ]
            
            for pattern in unwanted_patterns:
                # Remove by class
                for elem in transcript_soup.find_all(class_=re.compile(pattern, re.I)):
                    elem.decompose()
                # Remove by id
                for elem in transcript_soup.find_all(id=re.compile(pattern, re.I)):
                    elem.decompose()
            
            # Strategy 2: Try to find content by common class/id patterns
            content_selectors = [
                # Try by class
                transcript_soup.find('div', class_=re.compile(r'transcript', re.I)),
                transcript_soup.find('div', class_=re.compile(r'content', re.I)),
                transcript_soup.find('div', class_=re.compile(r'text', re.I)),
                transcript_soup.find('section', class_=re.compile(r'transcript|content', re.I)),
                transcript_soup.find('article', class_=re.compile(r'transcript|content', re.I)),
                # Try by id
                transcript_soup.find(id=re.compile(r'transcript|content|main|text', re.I)),
                # Try common semantic elements
                transcript_soup.find('main'),
                transcript_soup.find('article'),
            ]
            
            for element in content_selectors:
                if element:
                    text = element.get_text(separator='\n', strip=True)
                    # Check if it has substantial content (more than just navigation/links)
                    if len(text) > 500 and not re.match(r'^\s*(Home|About|Contact|Login|Sign Up)', text[:100], re.I):
                        transcript_content = text
                        break
            
            # Strategy 3: If still no content, find the largest text block
            if not transcript_content or len(transcript_content) < 500:
                # Find all divs and get the one with the most text
                all_divs = transcript_soup.find_all(['div', 'section', 'article', 'main'])
                largest_content = ""
                largest_length = 0
                
                for div in all_divs:
                    text = div.get_text(separator='\n', strip=True)
                    # Skip if it looks like navigation/menu/header
                    if len(text) > largest_length and len(text) > 500:
                        # Check if it contains transcript-like content
                        if any(word in text.lower() for word in ['operator', 'analyst', 'question', 'answer', 'call', 'earnings', 'quarter', 'revenue']):
                            largest_content = text
                            largest_length = len(text)
                
                if largest_content:
                    transcript_content = largest_content
            
            # Strategy 4: Find content by looking for transcript-specific patterns
            if not transcript_content or len(transcript_content) < 500:
                # Look for elements that likely contain the actual transcript
                # Transcripts often have speaker names, questions, answers
                all_text_elements = transcript_soup.find_all(['p', 'div', 'span', 'li'])
                transcript_sections = []
                
                for elem in all_text_elements:
                    text = elem.get_text(strip=True)
                    # Look for transcript indicators
                    if text and len(text) > 20:
                        # Check if it looks like transcript content
                        # Transcripts have patterns like: "Operator:", "Analyst:", speaker names, Q&A
                        if any(pattern in text for pattern in [
                            'Operator', 'Analyst', 'Question', 'Answer', 
                            'Good morning', 'Good afternoon', 'Thank you',
                            'CEO', 'CFO', 'President', 'Revenue', 'EPS',
                            'earnings', 'guidance', 'quarter'
                        ]) or re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+:', text):  # Pattern like "John Smith:"
                            # Check it's not navigation
                            if not any(nav_word in text.lower() for nav_word in ['home', 'menu', 'login', 'sign up', 'market is']):
                                transcript_sections.append(text)
                
                if transcript_sections:
                    # Join and clean up
                    transcript_content = '\n\n'.join(transcript_sections[:200])  # Limit to avoid too much text
                    
                    # Remove duplicate content
                    lines = transcript_content.split('\n')
                    seen = set()
                    unique_lines = []
                    for line in lines:
                        line_stripped = line.strip()
                        if line_stripped and line_stripped not in seen and len(line_stripped) > 10:
                            seen.add(line_stripped)
                            unique_lines.append(line)
                    transcript_content = '\n'.join(unique_lines)
            
            # Strategy 5: Last resort - get body content but filter aggressively
            if not transcript_content or len(transcript_content) < 500:
                # Remove common non-content elements by class/id
                for element in transcript_soup.find_all(['div', 'section'], class_=re.compile(r'menu|nav|sidebar|header|footer|ad|advertisement|social|share|cookie|quote|market', re.I)):
                    element.decompose()
                
                # Also remove elements with IDs containing these words
                for element in transcript_soup.find_all(['div', 'section'], id=re.compile(r'menu|nav|sidebar|header|footer|ad|quote|market', re.I)):
                    element.decompose()
                
                body = transcript_soup.find('body')
                if body:
                    # Get all paragraph and text content, but filter out short/boilerplate text
                    paragraphs = body.find_all(['p', 'div', 'section'])
                    content_parts = []
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        # Only include substantial paragraphs that don't look like navigation
                        if (len(text) > 100 and 
                            not any(nav in text.lower() for nav in ['market is open', 'after-hours', 'last quote', 'sign up', 'login']) and
                            not re.match(r'^[A-Z\s\(\)\%]+$', text)):  # Not all caps (likely navigation)
                            content_parts.append(text)
                    
                    if content_parts:
                        transcript_content = '\n\n'.join(content_parts)
            
            # Final validation - make sure we have actual transcript content, not just page template
            if transcript_content:
                # Check if it looks like actual transcript content vs page template
                transcript_lower = transcript_content.lower()
                has_template_indicators = any(indicator in transcript_lower for indicator in [
                    'market is open', 'after-hours quote', 'last quote from', 
                    'market is closed', 'sign up', 'login', 'home', 'about', 'contact'
                ])
                
                has_transcript_indicators = any(indicator in transcript_lower for indicator in [
                    'operator', 'analyst', 'question', 'answer', 'good morning', 'good afternoon',
                    'thank you', 'ceo', 'cfo', 'president', 'revenue', 'eps', 'earnings',
                    'guidance', 'quarter', 'fiscal year'
                ]) or re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+:', transcript_content)  # Speaker pattern
                
                # If it has more template indicators than transcript indicators, it's likely wrong
                if has_template_indicators and not has_transcript_indicators:
                    logger.warning("Extracted content appears to be page template, not transcript")
                    transcript_content = ""  # Reset to try Playwright
            
            # Strategy 6: Playwright is handled in async context (_arun method)
            # Skip here in sync context to avoid async/sync conflicts
            
            # Final validation - make sure we have actual transcript content, not just page template
            if transcript_content:
                transcript_lower = transcript_content.lower()
                has_template_indicators = any(indicator in transcript_lower for indicator in [
                    'market is open', 'after-hours quote', 'last quote from', 
                    'market is closed', 'sign up', 'login', 'home', 'about', 'contact'
                ])
                
                has_transcript_indicators = any(indicator in transcript_lower for indicator in [
                    'operator', 'analyst', 'question', 'answer', 'good morning', 'good afternoon',
                    'thank you', 'ceo', 'cfo', 'president', 'revenue', 'eps', 'earnings',
                    'guidance', 'quarter', 'fiscal year'
                ]) or re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+:', transcript_content)  # Speaker pattern
                
                # If it has template indicators but no transcript indicators, it's likely wrong
                if has_template_indicators and not has_transcript_indicators:
                    logger.warning("Extracted content appears to be page template, not transcript")
                    transcript_content = ""
            
            # If we still don't have valid transcript content, return error
            if not transcript_content or len(transcript_content) < 500:
                page_title = transcript_soup.title.string if transcript_soup.title else 'No title'
                body_length = len(transcript_soup.get_text()) if transcript_soup.body else 0
                logger.warning(f"Could not extract transcript content. Page title: {page_title}, Body text length: {body_length}")
                return f"**Transcript Content Not Found**\n\nRetrieved the transcript page for {symbol_upper} FY{fiscal_year} Q{quarter} but could not extract the transcript content.\n\n**URL:** {transcript_url}\n\n**Possible Issues:**\n1. The page structure may be different than expected\n2. Content may be loaded dynamically via JavaScript\n3. The transcript may be behind a paywall or require authentication\n4. The page may require cookies or session data\n\n**Troubleshooting:**\n1. Visit the URL manually to verify the transcript is accessible\n2. Check if the page requires JavaScript enabled\n3. Consider checking if authentication is required"
            
            # Format the output
            output = "**Earnings Call Transcript:**\n\n"
            output += f"**Company:** {symbol_upper}\n"
            output += f"**Period:** FY{fiscal_year} Q{quarter}\n"
            output += f"**Source:** discountingcashflows.com\n"
            output += "\n---\n\n"
            output += transcript_content
            output += f"\n\n---\n\n_Transcript length: {len(transcript_content):,} characters_\n"
            output += f"_Source URL: {transcript_url}_"
            
            return output
            
        except ValueError as e:
            # This is expected for JavaScript-rendered pages
            # The async _arun method will handle it with Playwright
            raise e
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching transcript: {str(e)}")
            return f"**Network Error**\n\nFailed to fetch transcript from discountingcashflows.com: {str(e)}\n\nPlease check your internet connection and try again."
        
        except Exception as e:
            logger.error(f"Error parsing transcript: {str(e)}")
            return f"**Error retrieving earnings transcript**\n\n{str(e)}\n\nPlease ensure the symbol, fiscal year, and quarter are correct."
    
    async def _arun(self, *args, **kwargs) -> str:
        """Async execution with Playwright support for JavaScript-rendered pages."""
        # LangChain BaseTool should pass arguments the same way as _run
        # But handle both cases: positional args and kwargs
        symbol = None
        fiscal_year = None
        quarter = None
        
        # Try to get from positional args first (matching _run signature)
        if len(args) >= 3:
            symbol, fiscal_year, quarter = args[0], args[1], args[2]
        elif len(args) == 1 and isinstance(args[0], dict):
            # LangChain might pass tool input as a single dict argument
            tool_input = args[0]
            symbol = tool_input.get('symbol')
            fiscal_year = tool_input.get('fiscal_year')
            quarter = tool_input.get('quarter')
        
        # Also check kwargs
        symbol = symbol or kwargs.get('symbol')
        fiscal_year = fiscal_year or kwargs.get('fiscal_year')
        quarter = quarter or kwargs.get('quarter')
        
        # Log what we received for debugging
        logger.info(f"_arun received: args={args}, kwargs={kwargs}, extracted: symbol={symbol}, fiscal_year={fiscal_year}, quarter={quarter}")
        
        # Validate we have symbol
        if not symbol:
            error_msg = f"Missing required argument: symbol. args={args}, kwargs={kwargs}"
            logger.error(error_msg)
            return f"**Error:** Missing required argument: symbol. Please provide a stock ticker symbol."
        
        symbol_upper = symbol.upper().strip()
        
        # Handle string 'None' case - but be careful not to override valid values
        if fiscal_year:
            fiscal_year_clean = fiscal_year.strip()
            if fiscal_year_clean.lower() in ('none', 'null', ''):
                fiscal_year = None
            else:
                fiscal_year = fiscal_year_clean
        if quarter:
            quarter_clean = quarter.strip()
            if quarter_clean.lower() in ('none', 'null', ''):
                quarter = None
            else:
                quarter = quarter_clean
        
        # Try to extract fiscal_year and quarter from combined formats like "FY2025Q2", "FY 2025 Q2", "2025Q2"
        # This handles cases where the LLM might pass them in a combined format
        if fiscal_year and not quarter:
            # Check if fiscal_year contains both year and quarter (e.g., "FY2025Q2", "2025Q2")
            combined_match = re.search(r'FY?\s*(\d{4})\s*Q\s*([1-4])', fiscal_year, re.I)
            if combined_match:
                fiscal_year = combined_match.group(1)
                quarter = combined_match.group(2)
                logger.info(f"Extracted from combined format: FY{fiscal_year} Q{quarter}")
        
        # If fiscal_year or quarter is still missing, get the most recent transcript
        if not fiscal_year or not quarter:
            logger.info(f"fiscal_year or quarter not provided for {symbol_upper}, fetching most recent transcript...")
            most_recent_year, most_recent_quarter = _get_most_recent_transcript(symbol_upper)
            if most_recent_year and most_recent_quarter:
                fiscal_year = fiscal_year or most_recent_year
                quarter = quarter or most_recent_quarter
                logger.info(f"Using most recent transcript: FY{fiscal_year} Q{quarter}")
            else:
                return f"**Error:** Could not find any available transcripts for {symbol_upper}. Please use list_earnings_transcripts to see available transcripts."
        
        logger.info(f"=== _arun called for {symbol_upper} FY{fiscal_year} Q{quarter} ===")
        # Since discountingcashflows.com uses JavaScript rendering, use Playwright as primary method
        fiscal_year_clean = re.sub(r'[^\d]', '', fiscal_year)
        quarter_clean = quarter.strip()
        
        # Build the URL - construct directly since we know the pattern
        transcript_url = f"{DCF_BASE_URL}/company/{symbol_upper}/transcripts/{fiscal_year_clean}/{quarter_clean}/"
        
        # Use Playwright to render the page (primary method for JavaScript-rendered content)
        try:
            from playwright.async_api import async_playwright
            logger.info(f"Using Playwright to render JavaScript content for {symbol_upper} FY{fiscal_year_clean} Q{quarter_clean}...")
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                )
                page = await context.new_page()
                
                # Navigate to the full transcript page
                logger.info(f"Navigating to full transcript page: {transcript_url}")
                await page.goto(transcript_url, wait_until="domcontentloaded", timeout=30000)
                
                # Wait for the transcriptsContent element to appear
                try:
                    await page.wait_for_selector('#transcriptsContent', timeout=15000)
                    logger.info("Found transcriptsContent element after page load")
                except Exception:
                    logger.warning("transcriptsContent element not found immediately")
                
                # Wait for HTMX to load the actual content (not just the link)
                # The content is loaded dynamically via HTMX after the page loads
                await page.wait_for_timeout(3000)
                
                # Wait for the transcriptsContent to have actual content (not just a link)
                try:
                    await page.wait_for_function(
                        '''
                        () => {
                            const elem = document.getElementById("transcriptsContent");
                            if (!elem) return false;
                            // Check if it has divs, not just links
                            const divs = elem.querySelectorAll("div");
                            const links = elem.querySelectorAll("a");
                            return divs.length > 0 && divs.length > links.length;
                        }
                        ''',
                        timeout=15000
                    )
                    logger.info("Confirmed transcriptsContent has div content (not just links)")
                except Exception as playwright_err:
                    logger.warning("Timeout waiting for transcriptsContent to populate with divs")
                
                # Additional wait to ensure content is fully rendered
                await page.wait_for_timeout(2000)
                
                # Final wait for network to be idle
                await page.wait_for_load_state("networkidle", timeout=5000)
                
                # Get the rendered HTML
                rendered_html = await page.content()
                await browser.close()
                
                # Log the rendered HTML length
                logger.info(f"Rendered HTML length: {len(rendered_html)} characters")
                
                # Parse and extract content
                rendered_soup = BeautifulSoup(rendered_html, 'html.parser')
                
                # Extract transcript content BEFORE removing unwanted elements
                # This is critical because the cleanup process may remove parent elements
                def match_flex_my5_classes(class_attr):
                    if not class_attr:
                        return False
                    if isinstance(class_attr, list):
                        class_str = ' '.join(class_attr)
                    else:
                        class_str = str(class_attr)
                    return 'flex' in class_str and 'flex-col' in class_str and 'my-5' in class_str
                
                # First, try to find the main transcript container to filter out sidebar/other transcripts
                main_transcript_container = None
                transcripts_content = rendered_soup.find(id='transcriptsContent')
                if transcripts_content:
                    logger.info("Found transcriptsContent container, will extract only from within it")
                    main_transcript_container = transcripts_content
                else:
                    # Look for main content area (not sidebar)
                    main_content = rendered_soup.find('main')
                    if main_content:
                        logger.info("Found main content area, will extract from it")
                        main_transcript_container = main_content
                    else:
                        logger.info("No specific container found, will extract from all flex.flex-col.my-5 elements")
                
                # Find all div.flex.flex-col.my-5 elements BEFORE cleanup
                all_flex_my5_divs = rendered_soup.find_all('div', class_=match_flex_my5_classes)
                logger.info(f"Found {len(all_flex_my5_divs)} div.flex.flex-col.my-5 elements BEFORE removing unwanted elements")
                
                # Filter to only include elements within the main transcript container
                if main_transcript_container:
                    flex_my5_divs = []
                    for div in all_flex_my5_divs:
                        # Check if this div is a descendant of the main container
                        parent = div.parent
                        is_in_container = False
                        while parent:
                            if parent == main_transcript_container:
                                is_in_container = True
                                break
                            if parent == rendered_soup or parent.name == '[document]':
                                break
                            parent = parent.parent
                        if is_in_container:
                            flex_my5_divs.append(div)
                    logger.info(f"After filtering to main container, {len(flex_my5_divs)} div.flex.flex-col.my-5 elements remain")
                else:
                    flex_my5_divs = all_flex_my5_divs
                
                transcript_content = ""
                transcript_parts = []
                
                # Extract content from all flex.flex-col.my-5 elements
                # Use a more comprehensive approach - extract ALL text content from each div
                for i, flex_div in enumerate(flex_my5_divs):
                    # Get ALL text from this flex_div, including all nested elements
                    # This captures the full transcript content regardless of structure
                    full_text = flex_div.get_text(separator='\n', strip=True)
                    
                    # Filter out very short content (likely navigation/metadata)
                    if len(full_text.strip()) < 30:
                        continue
                    
                    # Filter out metadata headers
                    text_lower = full_text.lower()
                    # Skip if it's just a metadata header (short text with company/period info)
                    if (full_text.strip().startswith('**Company:**') or 
                        (len(full_text.strip()) < 200 and 
                         ('company:' in text_lower or 'period:' in text_lower or 'source:' in text_lower) and
                         'conference' not in text_lower and 'operator' not in text_lower)):
                        continue
                    
                    # Extract text from child elements more carefully
                    # Look for the actual content div (usually has p-4 or similar padding classes)
                    child_divs = [child for child in flex_div.children 
                                 if hasattr(child, 'name') and child.name and child.name == 'div']
                    
                    extracted_text = None
                    
                    # Strategy 1: Find the div with the most text (likely the content div)
                    max_text_length = 0
                    for child_div in child_divs:
                        child_text = child_div.get_text(separator='\n', strip=True)
                        if len(child_text.strip()) > max_text_length:
                            max_text_length = len(child_text.strip())
                            extracted_text = child_text.strip()
                    
                    # Strategy 2: If no good child div found, use the full flex_div text
                    if not extracted_text or len(extracted_text) < 50:
                        # Get text but preserve line breaks for readability
                        extracted_text = full_text
                    
                    # Clean up the text - remove excessive whitespace but keep line breaks
                    lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
                    cleaned_text = '\n'.join(lines)
                    
                    # Only add if it's substantial content (not just metadata)
                    if cleaned_text and len(cleaned_text.strip()) > 50:
                        # Additional filter: skip if it's clearly just navigation/metadata
                        if not (cleaned_text.startswith('**') and len(cleaned_text) < 300):
                            transcript_parts.append(cleaned_text)
                    
                    # Log first few for debugging
                    if i < 3:
                        logger.debug(f"  -> Flex div {i}: Extracted {len(cleaned_text) if cleaned_text else 0} chars from {len(child_divs)} child divs")
                
                # Filter transcript parts to only include content from the correct fiscal year/quarter
                # (Moved outside the loop - was incorrectly inside before)
                if transcript_parts:
                    # Look for fiscal year/quarter indicators in the page to filter correctly
                    page_text = rendered_html.lower()
                    target_year = fiscal_year_clean
                    target_quarter = quarter_clean
                    
                    # Check if we're on the correct transcript page by looking at the URL or page content
                    # The URL should already be correct, but verify the content matches
                    filtered_parts = []
                    for part in transcript_parts:
                        part_lower = part.lower()
                        # Include the part if it doesn't clearly belong to a different quarter
                        # Look for indicators that suggest it's from the wrong quarter
                        has_wrong_quarter = False
                        
                        # Check for other quarter mentions that might indicate wrong transcript
                        # But be careful - transcripts often mention other quarters in context
                        # Only filter if we see clear indicators of a different transcript section
                        
                        # For now, include all parts since we're already on the correct URL
                        # The URL construction ensures we're on the right page
                        filtered_parts.append(part)
                    
                    transcript_parts = filtered_parts
                    logger.info(f"After filtering, {len(transcript_parts)} transcript parts remain")
                    
                    # Log summary of extracted content
                    if transcript_parts:
                        total_chars = sum(len(part) for part in transcript_parts)
                        logger.info(f"Total extracted content: {total_chars:,} characters from {len(transcript_parts)} parts")
                        # Log first 200 chars of first part for verification
                        if transcript_parts[0]:
                            preview = transcript_parts[0][:200].replace('\n', ' ')
                            logger.info(f"First part preview: {preview}...")
                    
                    # Now remove unwanted elements (after extraction)
                    for element in rendered_soup(['script', 'style', 'nav', 'header', 'footer', 'aside', 'iframe', 'noscript']):
                        element.decompose()
                    
                    # Remove elements with navigation/template classes/ids (but be more careful)
                    unwanted_patterns = [
                        r'menu|navigation|nav', r'header', r'footer', r'sidebar',
                        r'cookie|consent', r'social|share', r'ad|advertisement',
                        r'market.*quote|quote.*market', r'after.*hours|after-hours',
                        r'stock.*price|price.*stock', r'tab|tabs',
                    ]
                    for pattern in unwanted_patterns:
                        for elem in rendered_soup.find_all(['div', 'section', 'aside'], class_=re.compile(pattern, re.I)):
                            # Only remove if it doesn't contain transcript content
                            elem_text = elem.get_text(strip=True)
                            if not any(keyword in elem_text.lower() for keyword in [
                                'conference operator', 'analyst', 'cfo', 'ceo', 'president',
                                'thank you', 'good morning', 'good afternoon', 'quarter',
                                'revenue', 'earnings', 'guidance'
                            ]):
                                elem.decompose()
                        for elem in rendered_soup.find_all(['div', 'section', 'aside'], id=re.compile(pattern, re.I)):
                            elem_text = elem.get_text(strip=True)
                            if not any(keyword in elem_text.lower() for keyword in [
                                'conference operator', 'analyst', 'cfo', 'ceo', 'president',
                                'thank you', 'good morning', 'good afternoon', 'quarter',
                                'revenue', 'earnings', 'guidance'
                            ]):
                                elem.decompose()
                    
                    if transcript_parts:
                        transcript_content = '\n\n'.join(transcript_parts)
                        logger.info(f"Successfully extracted {len(transcript_content)} characters from {len(transcript_parts)} p-4 divs")
                    else:
                        logger.warning("No p-4 divs found in flex.flex-col.my-5 elements")
                        # Log some debugging info
                        all_divs = rendered_soup.find_all('div')
                        logger.info(f"Total divs in rendered HTML: {len(all_divs)}")
                        if all_divs:
                            # Show classes of first few divs
                            for i, div in enumerate(all_divs[:10]):
                                classes = div.get('class', [])
                                class_str = ' '.join(classes) if isinstance(classes, list) else str(classes)
                                text_preview = div.get_text(separator=' ', strip=True)[:100] if div.get_text(strip=True) else ""
                                logger.info(f"Div {i} classes: {class_str}, text_preview: {text_preview}...")
                    
                    # If we didn't find transcriptsContent div, use fallback strategies
                    if not transcript_content:
                        # Fallback: look for elements with transcript patterns
                        all_elements = rendered_soup.find_all(['div', 'section', 'article', 'main', 'p', 'span'])
                        transcript_parts = []
                        
                        for elem in all_elements:
                            text = elem.get_text(separator=' ', strip=True)
                            
                            # Look for speaker patterns: "Name (Role)" or patterns with parentheses
                            if re.search(r'\([^)]+\)', text) and len(text) > 50:
                                # Check if it looks like a speaker line or contains transcript keywords
                                if any(keyword in text.lower() for keyword in [
                                    'conference operator', 'analyst', 'cfo', 'ceo', 'president',
                                    'thank you', 'good morning', 'good afternoon', 'quarter',
                                    'revenue', 'earnings', 'guidance', 'million', 'billion'
                                ]) or re.search(r'[A-Z][a-z]+\s+\([^)]+\)', text):
                                    # Skip if it's navigation/template
                                    if not any(nav in text.lower() for nav in [
                                        'market is open', 'after-hours', 'sign up', 'login',
                                        'download pdf', 'ai insights'
                                    ]):
                                        transcript_parts.append(text)
                        
                        # If we didn't find speaker patterns, look for large text blocks in main content
                        if not transcript_parts or sum(len(p) for p in transcript_parts) < 1000:
                            body = rendered_soup.find('body')
                            if body:
                                main_content = body.find('main') or body.find('article')
                                if not main_content:
                                    all_divs = body.find_all('div', class_=lambda x: x and not any(
                                        nav in str(x).lower() for nav in ['nav', 'sidebar', 'header', 'footer', 'menu']
                                    ))
                                    if all_divs:
                                        main_content = max(all_divs, key=lambda d: len(d.get_text(strip=True)))
                                
                                if main_content:
                                    main_text = main_content.get_text(separator='\n', strip=True)
                                    lines = [line.strip() for line in main_text.split('\n') if line.strip()]
                                    
                                    for i, line in enumerate(lines):
                                        if len(line) < 20:
                                            continue
                                        
                                        if (re.search(r'\([^)]+\)', line) or
                                            any(keyword in line.lower() for keyword in [
                                                'conference operator', 'analyst', 'cfo', 'ceo',
                                                'thank you', 'good morning', 'good afternoon'
                                            ]) or
                                            (len(line) > 100 and any(keyword in line.lower() for keyword in [
                                                'revenue', 'earnings', 'quarter', 'million', 'billion'
                                            ]))):
                                            transcript_parts.append(line)
                                            for j in range(i+1, min(i+10, len(lines))):
                                                if len(lines[j]) > 50:
                                                    transcript_parts.append(lines[j])
                                                else:
                                                    break
                        
                        # Combine transcript parts
                        if transcript_parts:
                            seen = set()
                            unique_parts = []
                            for part in transcript_parts:
                                part_clean = part.strip()
                                if part_clean and part_clean not in seen and len(part_clean) > 20:
                                    seen.add(part_clean)
                                    unique_parts.append(part_clean)
                            
                            transcript_content = '\n\n'.join(unique_parts)
                    
                    # Final validation (applies to both transcriptsContent div and fallback strategies)
                    extraction_reasons = []
                    
                    if transcript_content:
                        transcript_lower = transcript_content.lower()
                        has_template = any(indicator in transcript_lower for indicator in [
                            'market is open', 'after-hours quote', 'last quote from',
                            'sign up', 'login', 'download pdf', 'ai insights'
                        ])
                        has_transcript = any(indicator in transcript_lower for indicator in [
                            'conference operator', 'analyst', 'cfo', 'ceo',
                            'question', 'answer', 'revenue', 'earnings', 'quarter'
                        ]) or re.search(r'\([^)]+\)', transcript_content)
                        
                        # If we found p-4 divs, we can be confident it's transcript content
                        if len(transcript_content) > 500:
                            output = "**Earnings Call Transcript:**\n\n"
                            output += f"**Company:** {symbol_upper}\n"
                            output += f"**Period:** FY{fiscal_year_clean} Q{quarter_clean}\n"
                            output += f"**Source:** discountingcashflows.com\n"
                            output += "\n---\n\n"
                            output += transcript_content
                            output += f"\n\n---\n\n_Transcript length: {len(transcript_content):,} characters_\n"
                            output += f"_Source URL: {transcript_url}_"
                            return output
                        elif has_transcript and not has_template and len(transcript_content) > 500:
                            # For fallback strategies, use stricter validation
                            output = "**Earnings Call Transcript:**\n\n"
                            output += f"**Company:** {symbol_upper}\n"
                            output += f"**Period:** FY{fiscal_year_clean} Q{quarter_clean}\n"
                            output += f"**Source:** discountingcashflows.com\n"
                            output += "\n---\n\n"
                            output += transcript_content
                            output += f"\n\n---\n\n_Transcript length: {len(transcript_content):,} characters_\n"
                            output += f"_Source URL: {transcript_url}_"
                            return output
                        else:
                            extraction_reasons.append(f"Content length too short: {len(transcript_content)} chars (minimum: 500)")
                            if not has_transcript:
                                extraction_reasons.append("Content doesn't contain transcript indicators")
                            if has_template:
                                extraction_reasons.append("Content contains template/navigation text")
                            logger.warning(f"Extracted content failed validation. Reasons: {'; '.join(extraction_reasons)}")
                    else:
                        # No transcript_content was extracted - explain why
                        extraction_reasons.append(f"No transcript content extracted")
                        if len(flex_my5_divs) == 0:
                            extraction_reasons.append("No div.flex.flex-col.my-5 elements found in rendered HTML")
                        else:
                            extraction_reasons.append(f"Found {len(flex_my5_divs)} flex.flex-col.my-5 divs but no p-4 divs extracted")
                            extraction_reasons.append(f"transcript_parts count: {len(transcript_parts)}")
                    
                    if not transcript_content:
                        reasons_str = '; '.join(extraction_reasons) if extraction_reasons else "Unknown reason"
                        logger.warning(f"Could not extract transcript content from rendered page. Page length: {len(rendered_html)}. Reasons: {reasons_str}")
        except ImportError:
            logger.warning("Playwright not available, skipping JavaScript rendering")
        except Exception as playwright_err:
            logger.error(f"Playwright async rendering failed: {str(playwright_err)}")
            import traceback
            logger.error(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"Async transcript fetch failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # If we got here, Playwright extraction failed
        # Return a helpful error message
        return f"**Transcript Content Not Found**\n\nCould not extract transcript content for {symbol.upper()} FY{fiscal_year} Q{quarter}.\n\nThe page appears to be JavaScript-rendered and Playwright was unable to extract the content.\n\n**URL:** {transcript_url if 'transcript_url' in locals() else 'N/A'}\n\n**Troubleshooting:**\n1. The page structure may have changed\n2. The transcript may require authentication\n3. Check the URL manually to verify accessibility"
