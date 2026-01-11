"""
Multi-agent system for earnings report summarization using LangGraph and web scraping.

Flow:
1. Query Analyzer Agent - Identifies ticker symbol and lists available transcripts via discountingcashflows.com
2. Transcript Retriever Agent - Retrieves the full earnings call transcript by scraping discountingcashflows.com
3. Summarizer Agent - Generates comprehensive summary from the transcript
"""

from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
import json
import logging

from app.config import get_settings
from app.tools.investor_relations import TranscriptListTool, TranscriptTool
from app.rag import get_rag_service
import re

logger = logging.getLogger(__name__)


class EarningsAgentState(TypedDict):
    """State shared across all agents in the graph."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    company_query: str
    ticker_symbol: Optional[str]
    company_name: Optional[str]
    summary: Optional[str]
    current_stage: str
    error: Optional[str]
    requested_fiscal_year: Optional[str]  # Fiscal year explicitly requested by user
    requested_quarter: Optional[str]  # Quarter explicitly requested by user
    transcript_retrieved: bool  # Flag to prevent multiple transcript retrievals


def create_earnings_agent():
    """Create the multi-agent LangGraph for earnings summarization."""
    
    settings = get_settings()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )
    
    # Initialize transcript tools
    transcript_list_tool = TranscriptListTool()
    transcript_tool = TranscriptTool()
    
    all_tools = [
        transcript_list_tool,
        transcript_tool,
    ]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(all_tools)
    
    # Create tool node
    tool_node = ToolNode(all_tools)
    
    # ==================== Agent Nodes ====================
    
    def query_analyzer(state: EarningsAgentState) -> Dict[str, Any]:
        """Analyze user query and extract company information."""
        
        # Extract fiscal year and quarter from user query if present
        import re
        user_query = state.get("company_query", "")
        
        # Also check messages if company_query doesn't have it
        if not user_query and state.get("messages"):
            for msg in state["messages"]:
                if hasattr(msg, "content") and isinstance(msg.content, str):
                    # Extract from the original query, not the formatted message
                    content = msg.content
                    # Remove the prefix "Please analyze and summarize the earnings reports for: "
                    if "Please analyze and summarize" in content:
                        user_query = content.split(":", 1)[-1].strip()
                    else:
                        user_query = content
                    break
        
        requested_fiscal_year = None
        requested_quarter = None
        
        # Try to extract fiscal year and quarter from query
        if user_query:
            logger.info(f"Extracting fiscal year and quarter from user query: {user_query}")
            
            # Patterns in order of specificity:
            # 1. Combined formats: FY2025Q2, FY 2025 Q2, 2025 Q2, Q2 2025, Q2 FY 2025, Q2 FY2025, etc.
            # Use tuples to track which group is year vs quarter
            combined_patterns = [
                (r'FY\s*(\d{4})\s*Q\s*([1-4])', True),    # FY 2025 Q2, FY2025Q2 - year is group 1
                (r'Q\s*([1-4])\s*FY\s*(\d{4})', False),   # Q2 FY 2025, Q2 FY2025 - quarter is group 1, year is group 2
                (r'(\d{4})\s*Q\s*([1-4])', True),         # 2025 Q2 - year is group 1
                (r'Q\s*([1-4])\s*(\d{4})', False),        # Q2 2025 - quarter is group 1, year is group 2
            ]
            
            for pattern, year_is_first in combined_patterns:
                combined_match = re.search(pattern, user_query, re.I)
                if combined_match:
                    if year_is_first:
                        requested_fiscal_year = combined_match.group(1)
                        requested_quarter = combined_match.group(2)
                    else:
                        # Quarter first pattern
                        requested_quarter = combined_match.group(1)
                        requested_fiscal_year = combined_match.group(2)
                    logger.info(f"Extracted from pattern '{pattern}': FY{requested_fiscal_year} Q{requested_quarter}")
                    break
            
            # If no combined match, try separate patterns
            if not requested_fiscal_year or not requested_quarter:
                # Try patterns with FY prefix first
                year_match = re.search(r'FY?\s*(\d{4})', user_query, re.I)
                if year_match:
                    requested_fiscal_year = year_match.group(1)
                    logger.info(f"Extracted fiscal year: {requested_fiscal_year}")
                else:
                    # Try standalone 4-digit year (likely fiscal year if between 2000-2099)
                    standalone_year_match = re.search(r'\b(20\d{2})\b', user_query)
                    if standalone_year_match:
                        requested_fiscal_year = standalone_year_match.group(1)
                        logger.info(f"Extracted standalone fiscal year: {requested_fiscal_year}")
                
                quarter_match = re.search(r'Q\s*([1-4])', user_query, re.I)
                if quarter_match:
                    requested_quarter = quarter_match.group(1)
                    logger.info(f"Extracted quarter: {requested_quarter}")
            
            if requested_fiscal_year and requested_quarter:
                logger.info(f"Successfully extracted: FY{requested_fiscal_year} Q{requested_quarter} from query: {user_query}")
            else:
                logger.warning(f"Could not extract both fiscal_year and quarter from query: {user_query}")
        
        # Try to extract ticker symbol from user query
        ticker_symbol = state.get("ticker_symbol")
        if not ticker_symbol and user_query:
            # Look for ticker-like patterns (1-5 uppercase letters, possibly preceded by $)
            ticker_patterns = [
                r'\$([A-Z]{1,5})\b',  # $NVDA, $AAPL
                r'\b([A-Z]{1,5})\s+(?:FY|Q|fiscal|quarter)',  # NVDA FY2025, AAPL Q2
                r'\b([A-Z]{1,5})\s+\d{4}',  # NVDA 2025
                r'(?:ticker|symbol|stock):\s*([A-Z]{1,5})',  # ticker: NVDA
            ]
            
            for pattern in ticker_patterns:
                ticker_match = re.search(pattern, user_query, re.I)
                if ticker_match:
                    ticker_symbol = ticker_match.group(1).upper()
                    logger.info(f"Extracted ticker symbol from query: {ticker_symbol}")
                    break
            
            # If no pattern match, try looking for standalone uppercase words that might be tickers
            if not ticker_symbol:
                # Common company names to ticker mapping
                company_to_ticker = {
                    'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'alphabet': 'GOOGL',
                    'amazon': 'AMZN', 'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA',
                    'tesla': 'TSLA', 'netflix': 'NFLX', 'amd': 'AMD', 'intel': 'INTC',
                    'salesforce': 'CRM', 'oracle': 'ORCL', 'adobe': 'ADBE', 'ibm': 'IBM',
                    'cisco': 'CSCO',
                }
                
                query_lower = user_query.lower()
                for company, ticker in company_to_ticker.items():
                    if company in query_lower:
                        ticker_symbol = ticker
                        logger.info(f"Identified ticker {ticker_symbol} from company name: {company}")
                        break
        
        # Build system prompt based on whether year and quarter were extracted
        has_year_and_quarter = requested_fiscal_year is not None and requested_quarter is not None
        
        if has_year_and_quarter:
            system_prompt = f"""You are an expert at understanding user queries about companies and stocks.

The user query has been analyzed and BOTH fiscal year ({requested_fiscal_year}) AND quarter ({requested_quarter}) were found.

Your task is to:
1. Identify the ticker symbol from the user's query
2. You MAY use list_earnings_transcripts to verify the transcript is available, but this is optional
3. The transcript_retriever will handle fetching the specific transcript

IMPORTANT: Before calling any tools, briefly explain your reasoning in simple terms without mentioning the tool itself. For example:
- "The user provided {requested_fiscal_year} Q{requested_quarter}, so I'll verify the transcript is available for this specific period"
- "The user query mentions [company], which corresponds to ticker [TICKER]"

Common ticker symbols:
- AAPL = Apple
- MSFT = Microsoft
- GOOGL/GOOG = Google/Alphabet
- AMZN = Amazon
- META = Meta (Facebook)
- NVDA = NVIDIA
- TSLA = Tesla
- NFLX = Netflix
- AMD = AMD
- INTC = Intel
- CRM = Salesforce
- ORCL = Oracle
- ADBE = Adobe
- IBM = IBM
- CSCO = Cisco

If the user provides a company name instead of a ticker, convert it to the ticker symbol.
You may optionally use list_earnings_transcripts, but it's not required since we already know the year and quarter.
"""
        else:
            system_prompt = """You are an expert at understanding user queries about companies and stocks.

IMPORTANT: The user query does NOT contain both a fiscal year AND quarter. The user only provided a ticker symbol or company name.

Your task is to:
1. Identify the ticker symbol from the user's query
2. DO NOT use list_earnings_transcripts - skip it entirely
3. The transcript_retriever will automatically fetch the MOST RECENT transcript

IMPORTANT: Before proceeding, briefly explain your reasoning in simple terms without mention of the tools called. For example:
- "The user did not specify a fiscal quarter, so I'll fetch the most recent transcript"
- "The user query mentions [company], which corresponds to ticker [TICKER]. Since no quarter was specified, I'll proceed to get the latest transcript"

CRITICAL: Since no year/quarter was specified, DO NOT call list_earnings_transcripts. 
The transcript_retriever will handle getting the latest transcript automatically.
Proceed directly to the next step where the latest transcript will be retrieved.

Common ticker symbols:
- AAPL = Apple
- MSFT = Microsoft
- GOOGL/GOOG = Google/Alphabet
- AMZN = Amazon
- META = Meta (Facebook)
- NVDA = NVIDIA
- TSLA = Tesla
- NFLX = Netflix
- AMD = AMD
- INTC = Intel
- CRM = Salesforce
- ORCL = Oracle
- ADBE = Adobe
- IBM = IBM
- CSCO = Cisco

If the user provides a company name instead of a ticker, convert it to the ticker symbol.
DO NOT use list_earnings_transcripts - proceed directly to the next step.
"""
        
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        
        response = llm_with_tools.invoke(messages)
        
        # Try to extract ticker from LLM response if we don't have it yet
        # The LLM may mention the ticker in its reasoning
        if not ticker_symbol and hasattr(response, 'content') and response.content:
            content_str = str(response.content)
            ticker_match = re.search(r'\b([A-Z]{1,5})\b', content_str)
            if ticker_match:
                potential_ticker = ticker_match.group(1)
                # Check if it's a known ticker (common ones)
                known_tickers = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'IBM', 'CSCO']
                if potential_ticker in known_tickers:
                    ticker_symbol = potential_ticker
                    logger.info(f"Extracted ticker {ticker_symbol} from LLM response")
        
        return {
            "messages": [response],
            "current_stage": "analyzing_query",
            "requested_fiscal_year": requested_fiscal_year,
            "requested_quarter": requested_quarter,
            "ticker_symbol": ticker_symbol or state.get("ticker_symbol"),  # Preserve existing or use extracted
        }
    
    async def check_embeddings(state: EarningsAgentState) -> Dict[str, Any]:
        """Check if embeddings already exist for the requested fiscal year, quarter, and ticker."""
        try:
            # Check if transcript was already retrieved via tools (before we can check embeddings)
            messages = state.get("messages", [])
            transcript_already_retrieved = False
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', None)
                    if tool_name == "get_earnings_transcript":
                        transcript_already_retrieved = True
                        logger.info("Transcript already retrieved via tools, will route to store_embeddings")
                        break
            
            settings = get_settings()
            if not getattr(settings, "rag_enabled", True):
                logger.info("RAG is disabled, skipping embedding check")
                message = AIMessage(content="RAG disabled. Retrieving transcript from website...")
                return {
                    "messages": [message],
                    "current_stage": "retrieving_transcript",
                    "embeddings_exist": False
                }
            
            # Extract information from state
            requested_fiscal_year = state.get("requested_fiscal_year")
            requested_quarter = state.get("requested_quarter")
            ticker_symbol = state.get("ticker_symbol")
            
            # Try to extract ticker from messages or company_query if not in state
            if not ticker_symbol:
                # First try company_query
                company_query = state.get("company_query", "")
                if company_query:
                    # Remove fiscal year/quarter from query to find ticker
                    query_clean = re.sub(r'FY\s*\d{4}|Q[1-4]|\d{4}', '', company_query, flags=re.I).strip()
                    # Look for ticker-like patterns (1-5 uppercase letters, possibly preceded by $)
                    ticker_patterns = [
                        r'\$([A-Z]{1,5})\b',  # $NVDA
                        r'\b([A-Z]{1,5})\b',  # NVDA (standalone uppercase word)
                    ]
                    for pattern in ticker_patterns:
                        ticker_match = re.search(pattern, query_clean)
                        if ticker_match:
                            potential_ticker = ticker_match.group(1).upper()
                            # Basic validation: common tickers are 1-5 uppercase letters
                            if len(potential_ticker) <= 5 and potential_ticker.isalpha():
                                ticker_symbol = potential_ticker
                                logger.info(f"Extracted ticker {ticker_symbol} from company_query")
                                break
                
                # Also check messages for ticker mentions
                if not ticker_symbol:
                    messages_list = list(state["messages"])
                    for msg in reversed(messages_list):
                        if hasattr(msg, 'content') and msg.content:
                            content_str = str(msg.content)
                            # Look for ticker in transcript metadata
                            ticker_match = re.search(r'\*\*Company:\*\* ([A-Z]+)', content_str)
                            if ticker_match:
                                ticker_symbol = ticker_match.group(1)
                                logger.info(f"Extracted ticker {ticker_symbol} from message")
                                break
            
            # Only check if we have all three: ticker, fiscal_year, and quarter
            if ticker_symbol and requested_fiscal_year and requested_quarter:
                try:
                    rag_service = get_rag_service()
                    embeddings_exist = await rag_service.transcript_is_chunked(
                        ticker_symbol=ticker_symbol,
                        fiscal_year=requested_fiscal_year,
                        quarter=requested_quarter,
                    )
                    
                    if embeddings_exist:
                        logger.info(f"Embeddings already exist for {ticker_symbol} FY{requested_fiscal_year} Q{requested_quarter}, skipping transcript retrieval")
                        # Store metadata in state for use in summarizer
                        # Add a message to make the stage visible - keep it as "retrieving_transcript" stage so "Retrieving Reports" is shown
                        message = AIMessage(content=f"Transcript for {ticker_symbol} {requested_fiscal_year}Q{requested_quarter} has been fetched. Preparing for analysis...")
                        return {
                            "messages": [message],
                            "current_stage": "retrieving_transcript",  # Use retrieving_transcript so "Retrieving Reports" is shown
                            "embeddings_exist": True,
                            "ticker_symbol": ticker_symbol,
                            "requested_fiscal_year": requested_fiscal_year,  # Preserve for summarizer
                            "requested_quarter": requested_quarter,  # Preserve for summarizer
                            "skip_transcript_retrieval": True,
                        }
                    else:
                        logger.info(f"Embeddings do not exist for {ticker_symbol} FY{requested_fiscal_year} Q{requested_quarter}, will retrieve transcript")
                        # Add a message to make the stage visible
                        message = AIMessage(content=f"Fetching transcript for {ticker_symbol} {requested_fiscal_year}Q{requested_quarter}...")
                        return {
                            "messages": [message],
                            "current_stage": "retrieving_transcript",
                            "embeddings_exist": False,
                            "ticker_symbol": ticker_symbol,
                            "requested_fiscal_year": requested_fiscal_year,  # Preserve for store_embeddings
                            "requested_quarter": requested_quarter,  # Preserve for store_embeddings
                            "skip_transcript_retrieval": False,
                        }
                except Exception as e:
                    logger.error(f"Error checking embeddings: {e}", exc_info=True)
                    # On error, proceed with transcript retrieval
                    message = AIMessage(content="Error checking database. Retrieving transcript from website...")
                    return {
                        "messages": [message],
                        "current_stage": "retrieving_transcript",
                        "embeddings_exist": False,
                        "ticker_symbol": ticker_symbol,
                        "skip_transcript_retrieval": False,
                    }
            else:
                # Missing information - need to proceed with transcript retrieval
                # This happens when user doesn't provide fiscal year/quarter (to get latest)
                # OR when transcript was already retrieved via tools
                if transcript_already_retrieved:
                    logger.info(f"Transcript already retrieved via tools, routing to store_embeddings")
                    # Build user-friendly message with ticker/year/quarter if available
                    if ticker_symbol and requested_fiscal_year and requested_quarter:
                        message_content = f"Transcript for {ticker_symbol} {requested_fiscal_year}Q{requested_quarter} has been fetched. Preparing for analysis..."
                    elif ticker_symbol:
                        message_content = f"Latest earnings report transcript for {ticker_symbol} has been fetched. Preparing for analysis..."
                    else:
                        message_content = "Latest earnings report transcript has been retrieved. Preparing for analysis..."
                    message = AIMessage(content=message_content)
                    return {
                        "messages": [message],
                        "current_stage": "retrieving_transcript",  # Keep as retrieving_transcript to show "Retrieving Reports"
                        "embeddings_exist": False,
                        "ticker_symbol": ticker_symbol,
                        "requested_fiscal_year": requested_fiscal_year,  # May be None
                        "requested_quarter": requested_quarter,  # May be None
                        "skip_transcript_retrieval": True,  # Signal that transcript is already retrieved
                    }
                else:
                    logger.info(f"Cannot check embeddings: missing info. ticker={ticker_symbol}, year={requested_fiscal_year}, quarter={requested_quarter}. Will retrieve transcript.")
                    # Add a message to make the stage visible
                    message = AIMessage(content="Retrieving the latest earnings transcript from website...")
                    return {
                        "messages": [message],
                        "current_stage": "retrieving_transcript",
                        "embeddings_exist": False,
                        "ticker_symbol": ticker_symbol,
                        "requested_fiscal_year": requested_fiscal_year,  # May be None
                        "requested_quarter": requested_quarter,  # May be None
                        "skip_transcript_retrieval": False,
                    }
        except Exception as e:
            logger.error(f"Error in check_embeddings node: {e}", exc_info=True)
            # On error, proceed with transcript retrieval
            return {
                "current_stage": "retrieving_transcript",
                "embeddings_exist": False,
                "skip_transcript_retrieval": False,
            }
    
    async def store_embeddings(state: EarningsAgentState) -> Dict[str, Any]:
        """Store transcript embeddings in database for RAG retrieval."""
        try:
            settings = get_settings()
            if not getattr(settings, "rag_enabled", True):
                logger.info("RAG is disabled, skipping embedding storage")
                return {"current_stage": "storing_embeddings"}
            
            # Extract transcript information from messages
            messages_list = list(state["messages"])
            
            ticker_symbol = state.get("ticker_symbol")
            requested_fiscal_year = state.get("requested_fiscal_year")
            requested_quarter = state.get("requested_quarter")
            
            # Find transcript content in tool messages
            transcript_content = None
            source_url = None
            
            for msg in reversed(messages_list):
                if hasattr(msg, 'content') and msg.content:
                    content_str = str(msg.content)
                    # Look for transcript indicators
                    if "Earnings Call Transcript" in content_str or "discountingcashflows.com" in content_str:
                        if len(content_str) > 1000 and not any(template in content_str.lower() for template in [
                            'market is open', 'after-hours quote', 'last quote from'
                        ]):
                            transcript_content = content_str
                            # Extract source URL if present
                            url_match = re.search(r'_Source URL: (https?://[^\s_]+)_', content_str)
                            if url_match:
                                source_url = url_match.group(1)
                            break
            
            # Extract transcript content (remove metadata headers)
            transcript_body = None
            if transcript_content:
                # The transcript format is:
                # **Earnings Call Transcript:**
                # **Company:** NVDA
                # **Period:** FY2026 Q3
                # **Source:** discountingcashflows.com
                # ---
                # [ACTUAL TRANSCRIPT CONTENT]
                # ---
                # _Transcript length: ..._
                # _Source URL: ..._
                
                # Find the content between the two "---" separators
                parts = transcript_content.split('---')
                if len(parts) >= 3:
                    # Content is between first and second "---"
                    transcript_body = parts[1].strip()
                else:
                    # Fallback: try to find content after "Earnings Call Transcript" header
                    lines = transcript_content.split('\n')
                    start_idx = 0
                    found_separator = False
                    for i, line in enumerate(lines):
                        if "---" in line:
                            if not found_separator:
                                start_idx = i + 1
                                found_separator = True
                            else:
                                # Second separator - end here
                                transcript_body = '\n'.join(lines[start_idx:i]).strip()
                                break
                    if not transcript_body:
                        # Last resort: take everything after first separator
                        transcript_body = '\n'.join(lines[start_idx:]).strip()
                        # Remove trailing metadata if present
                        if "_Transcript length:" in transcript_body:
                            transcript_body = transcript_body.split("_Transcript length:")[0].strip()
                        if "_Source URL:" in transcript_body:
                            transcript_body = transcript_body.split("_Source URL:")[0].strip()
                
                # Log the extracted transcript length for debugging
                if transcript_body:
                    logger.info(f"Extracted transcript body: {len(transcript_body):,} characters")
                else:
                    logger.warning(f"Failed to extract transcript body from content ({len(transcript_content):,} chars)")
            
            # Try to extract ticker, fiscal year, quarter from transcript content if not in state
            if not ticker_symbol and transcript_content:
                # Look for ticker in transcript
                ticker_match = re.search(r'\*\*Company:\*\* ([A-Z]+)', transcript_content)
                if ticker_match:
                    ticker_symbol = ticker_match.group(1)
            
            if not requested_fiscal_year and transcript_content:
                period_match = re.search(r'\*\*Period:\*\* FY(\d{4}) Q([1-4])', transcript_content)
                if period_match:
                    requested_fiscal_year = period_match.group(1)
                    requested_quarter = period_match.group(2)
            
            # Check if embeddings already exist before storing
            if transcript_body and ticker_symbol and requested_fiscal_year and requested_quarter:
                try:
                    rag_service = get_rag_service()
                    
                    # Check if embeddings already exist
                    embeddings_exist = await rag_service.transcript_is_chunked(
                        ticker_symbol=ticker_symbol,
                        fiscal_year=requested_fiscal_year,
                        quarter=requested_quarter,
                    )
                    
                    if embeddings_exist:
                        logger.info(f"Embeddings already exist for {ticker_symbol} FY{requested_fiscal_year} Q{requested_quarter}, skipping storage")
                        return {"current_stage": "storing_embeddings"}
                    
                    # Get company name (default to ticker if not available)
                    company_name = state.get("company_name") or ticker_symbol
                    
                    # Store embeddings (this will delete existing chunks first)
                    chunk_count = await rag_service.chunk_and_store_transcript(
                        transcript_content=transcript_body,
                        ticker_symbol=ticker_symbol,
                        company_name=company_name,
                        fiscal_year=requested_fiscal_year,
                        quarter=requested_quarter,
                        source_url=source_url or "",
                    )
                    logger.info(f"Stored {chunk_count} chunks with embeddings for {ticker_symbol} FY{requested_fiscal_year} Q{requested_quarter}")
                except Exception as e:
                    logger.error(f"Error storing embeddings: {e}", exc_info=True)
                    # Continue even if embedding storage fails
            else:
                logger.warning(f"Cannot store embeddings: missing data. ticker={ticker_symbol}, year={requested_fiscal_year}, quarter={requested_quarter}, has_content={transcript_body is not None}")
            
            return {"current_stage": "storing_embeddings"}
        except Exception as e:
            logger.error(f"Error in store_embeddings node: {e}", exc_info=True)
            # Return state to continue to summarizer even if embedding storage fails
            return {"current_stage": "storing_embeddings", "error": str(e)}
    
    def transcript_retriever(state: EarningsAgentState) -> Dict[str, Any]:
        """Retrieve the earnings transcript content."""
        
        # Get requested fiscal year and quarter from state if available
        requested_fiscal_year = state.get("requested_fiscal_year")
        requested_quarter = state.get("requested_quarter")
        
        logger.info(f"transcript_retriever: requested_fiscal_year={requested_fiscal_year}, requested_quarter={requested_quarter}")
        
        # Check if we have a fiscal year but no quarter - ask user to specify quarter
        if requested_fiscal_year and not requested_quarter:
            try:
                # Extract ticker symbol from state or try to extract from company_query
                ticker_symbol = state.get("ticker_symbol")
                company_name = state.get("company_name")
                
                # If ticker not in state, try to extract from company_query
                if not ticker_symbol:
                    company_query = state.get("company_query", "")
                    # Try to find a ticker-like pattern (uppercase letters, 1-5 chars)
                    ticker_match = re.search(r'\b([A-Z]{1,5})\b', company_query)
                    if ticker_match:
                        ticker_symbol = ticker_match.group(1)
                
                # Use ticker or company name, fallback to generic
                display_name = company_name or ticker_symbol or "the company"
                
                if ticker_symbol:
                    clarification_message = f"""I found that you're looking for {display_name}'s earnings report for fiscal year {requested_fiscal_year}, but I need you to specify which quarter (Q1, Q2, Q3, or Q4) you'd like me to analyze.

Please provide the quarter, for example:
- "{ticker_symbol} {requested_fiscal_year} Q1"
- "{ticker_symbol} {requested_fiscal_year} Q2"
- "{ticker_symbol} {requested_fiscal_year} Q3"
- "{ticker_symbol} {requested_fiscal_year} Q4"

Or you can simply say "Q1", "Q2", "Q3", or "Q4" and I'll use the fiscal year {requested_fiscal_year}."""
                else:
                    clarification_message = f"""I found that you're looking for {display_name}'s earnings report for fiscal year {requested_fiscal_year}, but I need you to specify which quarter (Q1, Q2, Q3, or Q4) you'd like me to analyze.

Please provide the quarter, for example:
- "{requested_fiscal_year} Q1"
- "{requested_fiscal_year} Q2"
- "{requested_fiscal_year} Q3"
- "{requested_fiscal_year} Q4"

Or you can simply say "Q1", "Q2", "Q3", or "Q4" and I'll use the fiscal year {requested_fiscal_year}."""
                
                logger.info(f"Returning clarification message for fiscal year {requested_fiscal_year} without quarter")
                return {
                    "messages": [AIMessage(content=clarification_message)],
                    "current_stage": "complete",  # End here to return the clarification
                    "summary": clarification_message,  # Set as summary so it's returned to user
                }
            except Exception as e:
                logger.error(f"Error generating clarification message: {e}", exc_info=True)
                # Fall through to normal processing if clarification fails
        
        # Build system prompt with explicit values if available
        if requested_fiscal_year and requested_quarter:
            system_prompt = f"""You are an expert at retrieving earnings transcripts from discountingcashflows.com.

CRITICAL: The user has explicitly requested FY{requested_fiscal_year} Q{requested_quarter}. You MUST use these exact values.

IMPORTANT: Before calling the tool, briefly explain your reasoning. For example:
- "The user has requested FY{requested_fiscal_year} Q{requested_quarter} for [TICKER]. I'll fetch this specific transcript."
- "I need to retrieve the transcript for [TICKER] for fiscal year {requested_fiscal_year}, quarter {requested_quarter}."

RULES:
1. Call get_earnings_transcript EXACTLY ONCE with:
   - symbol: The ticker symbol (extract from the conversation)
   - fiscal_year: "{requested_fiscal_year}" (use this EXACT value - do NOT use "None" or leave it empty)
   - quarter: "{requested_quarter}" (use this EXACT value - do NOT use "None" or leave it empty)

2. Do NOT call the tool multiple times
3. Do NOT use different fiscal year or quarter values
4. Do NOT fetch multiple transcripts
5. Do NOT pass "None" as a string - pass the actual values "{requested_fiscal_year}" and "{requested_quarter}"

IMPORTANT: When calling get_earnings_transcript, you MUST provide:
- fiscal_year: "{requested_fiscal_year}"
- quarter: "{requested_quarter}"

After calling get_earnings_transcript ONCE with fiscal_year="{requested_fiscal_year}" and quarter="{requested_quarter}", stop immediately.
"""
        else:
            system_prompt = """You are an expert at retrieving earnings transcripts from discountingcashflows.com.

IMPORTANT: Before calling the tool, briefly explain your reasoning. For example:
- "The user did not specify a fiscal quarter, so I'll fetch the most recent transcript for [TICKER]"
- "No quarter was specified, so I'll retrieve the latest available transcript"
- "I need to get the most recent earnings transcript for [TICKER] since no specific quarter was provided"

CRITICAL RULES - READ CAREFULLY:
1. Call get_earnings_transcript EXACTLY ONCE - NEVER call it multiple times
2. When the user specifies a fiscal year and quarter, you MUST extract them correctly and use ONLY those values
3. Do NOT fetch multiple transcripts - fetch ONLY the one the user requested
4. IGNORE any transcript lists you may have seen - do NOT try to fetch multiple transcripts from a list

Examples of user queries and how to extract:
- "FY2025Q2" or "FY 2025 Q2" → fiscal_year: "2025", quarter: "2" → Call tool ONCE with these values
- "2025 Q2" → fiscal_year: "2025", quarter: "2" → Call tool ONCE with these values
- "Q2 2025" → fiscal_year: "2025", quarter: "2" → Call tool ONCE with these values
- "second quarter 2025" → fiscal_year: "2025", quarter: "2" → Call tool ONCE with these values

Extraction rules:
- Look at the user's ORIGINAL query to find the fiscal year and quarter
- Extract the year number (e.g., "2025" from "FY2025" or "2025")
- Extract the quarter number (e.g., "2" from "Q2" or "2")
- If the user says "FY2025Q2", extract fiscal_year="2025" and quarter="2"

Once you have extracted the fiscal year and quarter from the user's query:
- Call get_earnings_transcript tool EXACTLY ONCE with:
  - symbol: The ticker symbol (e.g., "NVDA")
  - fiscal_year: The fiscal year as a string (e.g., "2025") - extract ONLY the year number
  - quarter: The quarter number as a string "1", "2", "3", or "4" - extract ONLY the number

If the user does NOT specify a fiscal year and quarter:
- Call get_earnings_transcript with ONLY the symbol parameter
- Omit fiscal_year and quarter parameters (or pass None)
- The tool will automatically retrieve the MOST RECENT transcript available
- Still call the tool EXACTLY ONCE

CRITICAL: When the user only provides a ticker symbol (e.g., "NVDA", "Apple", "MSFT") without any year or quarter:
- DO NOT look at any transcript lists you may have seen
- DO NOT try to pick a specific quarter from a list
- DO NOT call the tool multiple times for different quarters
- Call: get_earnings_transcript(symbol="NVDA", fiscal_year=None, quarter=None) EXACTLY ONCE
- OR call: get_earnings_transcript(symbol="NVDA") without fiscal_year and quarter parameters EXACTLY ONCE
- The tool will automatically select the latest/most recent transcript (the most recent quarter of the most recent year)
- After calling ONCE, STOP. Do not make any more tool calls.

ABSOLUTELY FORBIDDEN:
- Calling the tool multiple times
- Fetching multiple transcripts
- Iterating through quarters
- Using different values than what the user specified
- Trying to guess or extract a fiscal year/quarter if the user didn't provide one
- Looking at transcript lists and trying to fetch multiple transcripts from them

After calling get_earnings_transcript EXACTLY ONCE, stop immediately. Do not make any more tool calls.
"""
        
        messages_list = list(state["messages"])
        messages = [SystemMessage(content=system_prompt)] + messages_list
        
        response = llm_with_tools.invoke(messages)
        
        # Check if the response contains tool calls for get_earnings_transcript
        transcript_retrieved = False
        if hasattr(response, "tool_calls") and response.tool_calls:
            # Count how many get_earnings_transcript calls are being made
            transcript_calls = [tc for tc in response.tool_calls if tc.get("name") == "get_earnings_transcript"]
            if len(transcript_calls) > 1:
                # If multiple calls, keep only the first one
                logger.warning(f"Agent attempted to make {len(transcript_calls)} get_earnings_transcript calls. Limiting to one.")
                # Filter to keep only the first get_earnings_transcript call
                other_calls = [tc for tc in response.tool_calls if tc.get("name") != "get_earnings_transcript"]
                response.tool_calls = other_calls + [transcript_calls[0]] if transcript_calls else other_calls
            if transcript_calls:
                transcript_retrieved = True
        
        return {
            "messages": [response],
            "current_stage": "retrieving_transcript",
            "transcript_retrieved": transcript_retrieved
        }
    
    
    async def summarizer(state: EarningsAgentState) -> Dict[str, Any]:
        """Generate comprehensive summary of the earnings report using RAG when available."""
        
        settings = get_settings()
        use_rag = getattr(settings, "rag_enabled", True)
        
        # Extract metadata
        ticker_symbol = state.get("ticker_symbol")
        requested_fiscal_year = state.get("requested_fiscal_year")
        requested_quarter = state.get("requested_quarter")
        
        # Try to extract from messages if not in state
        messages_list = list(state["messages"])
        if not ticker_symbol or not requested_fiscal_year or not requested_quarter:
            for msg in reversed(messages_list):
                if hasattr(msg, 'content') and msg.content:
                    content_str = str(msg.content)
                    # Extract ticker
                    if not ticker_symbol:
                        ticker_match = re.search(r'\*\*Company:\*\* ([A-Z]+)', content_str)
                        if ticker_match:
                            ticker_symbol = ticker_match.group(1)
                    # Extract fiscal period
                    if not requested_fiscal_year or not requested_quarter:
                        period_match = re.search(r'\*\*Period:\*\* FY(\d{4}) Q([1-4])', content_str)
                        if period_match:
                            requested_fiscal_year = period_match.group(1)
                            requested_quarter = period_match.group(2)
                    if ticker_symbol and requested_fiscal_year and requested_quarter:
                        break
        
        # Try to use RAG retrieval if enabled and we have the metadata
        relevant_chunks = []
        if use_rag and ticker_symbol and requested_fiscal_year and requested_quarter:
            try:
                rag_service = get_rag_service()
                
                # Check if transcript is already chunked
                is_chunked = await rag_service.transcript_is_chunked(
                    ticker_symbol=ticker_symbol,
                    fiscal_year=requested_fiscal_year,
                    quarter=requested_quarter,
                )
                
                if is_chunked:
                    # Use RAG: retrieve relevant chunks for comprehensive summary
                    # Use a broad query to get chunks covering all major topics
                    summary_query = "financial highlights revenue earnings per share EPS profitability margins free cash flow business segments performance metrics operational highlights management commentary strategic outlook guidance risks challenges announcements growth catalysts"
                    
                    relevant_chunks = await rag_service.retrieve_relevant_chunks(
                        query=summary_query,
                        ticker_symbol=ticker_symbol,
                        fiscal_year=requested_fiscal_year,
                        quarter=requested_quarter,
                        top_k=rag_service.top_k,
                    )
                    logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for summarization using RAG")
            except Exception as e:
                logger.warning(f"RAG retrieval failed, falling back to full transcript: {e}", exc_info=True)
                relevant_chunks = []
        
        # Build transcript content for summarization
        if relevant_chunks:
            # Use RAG chunks
            transcript_content = "\n\n---\n\n".join([
                f"[Chunk {chunk['chunk_index']}] {chunk['content']}"
                for chunk in relevant_chunks
            ])
            logger.info(f"Using {len(relevant_chunks)} RAG chunks (total {len(transcript_content)} chars) for summarization")
        else:
            # Fallback: use full transcript from messages
            transcript_content = None
            for msg in reversed(messages_list):
                if hasattr(msg, 'content') and msg.content:
                    content_str = str(msg.content)
                    # Look for transcript indicators
                    if "Earnings Call Transcript" in content_str or "discountingcashflows.com" in content_str:
                        if len(content_str) > 1000 and not any(template in content_str.lower() for template in [
                            'market is open', 'after-hours quote', 'last quote from'
                        ]):
                            transcript_content = content_str
                            # Remove metadata headers
                            lines = content_str.split('\n')
                            start_idx = 0
                            for i, line in enumerate(lines):
                                if "---" in line:
                                    start_idx = i + 1
                                    break
                            if "---" in transcript_content:
                                transcript_content = '\n'.join(transcript_content.split("---")[1:-1]).strip()
                            break
            
            if not transcript_content:
                # Check if we have any substantial content
                for msg in reversed(messages_list):
                    if hasattr(msg, 'content') and msg.content:
                        content_str = str(msg.content)
                        if len(content_str) > 500:
                            transcript_content = content_str
                            break
        
        system_prompt = """You are an expert financial analyst specializing in earnings call transcript analysis.

You have been provided with an earnings call transcript in the conversation history. Your task is to analyze the transcript and generate a comprehensive, well-structured summary that highlights the KEY POINTS and most important information from the earnings call.

IMPORTANT: Look through the conversation history above to find the earnings call transcript. The transcript content should be in one of the previous tool execution results. If you see content that looks like page navigation or template (e.g., "Market is Open", "After-Hours Quote"), that is NOT the transcript - keep looking for the actual transcript content which should contain speaker names, questions, answers, financial metrics, etc.

## Summary Structure

Generate a clear, structured summary with the following sections:

### 📊 Executive Summary
- Start with a brief 2-3 sentence overview highlighting the most critical points
- Include the fiscal period (e.g., "Q4 FY2025")

### 💰 Key Financial Highlights
Extract and highlight:
- **Revenue**: Total revenue and year-over-year (YoY) or quarter-over-quarter (QoQ) growth percentages
- **Earnings Per Share (EPS)**: Actual vs. estimates if mentioned
- **Profitability**: Net income, gross margin, operating margin
- **Free Cash Flow**: If mentioned
- **Financial guidance**: Forward-looking revenue/EPS guidance

### 🚀 Business Segment Performance
- Performance breakdown by business segment or product line
- Which segments/products showed strong growth
- Which segments/products declined or faced challenges
- Notable numbers and percentages for each segment

### 📈 Key Metrics & Operational Highlights
- Company-specific metrics (e.g., for NVIDIA: data center revenue, gaming revenue, automotive revenue)
- Customer/user growth numbers
- Market share or competitive positioning mentions
- Operational efficiency metrics

### 💬 Management Commentary & Strategic Outlook
- **Key quotes** from the CEO/CFO highlighting important points
- Strategic initiatives or pivots mentioned
- Forward guidance and outlook for next quarter/year
- Major strategic announcements or direction changes

### ⚠️ Risks & Challenges Discussed
- Risk factors mentioned by management
- Competitive pressures discussed
- Regulatory or market challenges
- Supply chain or operational issues

### 🎯 Notable Events & Announcements
- New product launches or announcements
- Acquisitions, partnerships, or strategic deals
- Leadership changes or organizational updates
- Significant customer wins or contracts

## Instructions:
1. **Extract specific numbers**: Include actual dollar amounts, percentages, and growth rates mentioned in the transcript
2. **Use quotes strategically**: Include important quotes from executives that highlight key points
3. **Be comprehensive**: Cover all major topics discussed in the call
4. **Highlight the most important points**: If revenue grew 200%, make that prominent
5. **Format clearly**: Use bullet points, bold text (**like this**), and clear section headers
6. **Be accurate**: Only include information that was actually mentioned in the transcript

If certain information (like specific financial metrics) is not available in the transcript, you can note that, but still summarize what was discussed.

IMPORTANT: Look through the conversation history above to find the earnings call transcript. It should have been retrieved using the get_earnings_transcript tool. The transcript content will be in one of the previous tool execution results. Once you find it, analyze it thoroughly and generate the comprehensive summary following the structure above.

Make sure to highlight KEY POINTS with specific numbers, percentages, and important quotes from executives."""
        
        # Prepare messages for summarization
        if relevant_chunks:
            # When using RAG, provide the chunks directly
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"""Please analyze the following earnings call transcript chunks and generate a comprehensive summary. The transcript has been segmented into {len(relevant_chunks)} key sections:

{transcript_content}

Generate a comprehensive summary covering all the key sections above.""")
            ]
        else:
            # Fallback: use full transcript from messages
            messages_list = list(state["messages"])
            messages = [SystemMessage(content=system_prompt)] + messages_list
            messages.append(HumanMessage(content="Please analyze the earnings call transcript that was retrieved above and generate a comprehensive summary with all key points, financial highlights, and management commentary."))
        
        # Use regular LLM (not with tools) since we're just summarizing
        response = llm.invoke(messages)
        
        return {
            "messages": [response],
            "summary": response.content,
            "current_stage": "complete"
        }
    
    def should_use_tools(state: EarningsAgentState) -> str:
        """Determine if we should continue with tools or move to next stage."""
        
        messages = state["messages"]
        last_message = messages[-1]
        
        # If LLM wants to use tools, route to tool node
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        
        # Otherwise, determine next stage based on current stage
        current_stage = state.get("current_stage", "analyzing_query")
        
        if current_stage == "analyzing_query":
            return "check_embeddings"
        elif current_stage == "checking_embeddings":
            # Check if transcript was already retrieved via tools (before embeddings check)
            messages = state.get("messages", [])
            transcript_already_retrieved = False
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', None)
                    if tool_name == "get_earnings_transcript":
                        transcript_already_retrieved = True
                        break
            
            # Check if we should skip transcript retrieval
            skip_transcript_retrieval = state.get("skip_transcript_retrieval", False)
            if skip_transcript_retrieval:
                if transcript_already_retrieved:
                    # Transcript was just retrieved via tools, need to store embeddings
                    return "store_embeddings"
                else:
                    # Embeddings exist in DB, skip to summarizer
                    return "summarizer"
            else:
                # Embeddings don't exist, proceed with transcript retrieval
                return "transcript_retriever"
        elif current_stage == "retrieving_transcript":
            # Check if we should skip transcript retrieval (embeddings exist)
            # Note: check_embeddings sets current_stage to "retrieving_transcript" when embeddings exist
            skip_transcript_retrieval = state.get("skip_transcript_retrieval", False)
            if skip_transcript_retrieval:
                # Embeddings exist, skip to summarizer
                return "summarizer"
            
            # Check if transcript was already retrieved
            transcript_retrieved = False
            for msg in reversed(messages):
                if isinstance(msg, ToolMessage):
                    tool_name = getattr(msg, 'name', None)
                    if tool_name == "get_earnings_transcript":
                        transcript_retrieved = True
                        break
            
            if transcript_retrieved:
                # Transcript was retrieved, go to store_embeddings
                return "store_embeddings"
            else:
                # No transcript yet, but no tool calls either - go to summarizer as fallback
                # (This shouldn't normally happen, but prevents errors)
                return "summarizer"
        elif current_stage == "storing_embeddings":
            return "summarizer"
        else:
            return "end"
    
    def after_tools(state: EarningsAgentState) -> str:
        """Determine next step after tool execution."""
        current_stage = state.get("current_stage", "analyzing_query")
        
        # Check if we've already retrieved a transcript
        transcript_retrieved = state.get("transcript_retrieved", False)
        
        # Check messages for ToolMessage from get_earnings_transcript
        # ToolMessage objects are created after tool execution
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                # ToolMessage has a 'name' attribute indicating which tool was called
                tool_name = getattr(msg, 'name', None)
                logger.debug(f"Found ToolMessage with name: {tool_name}")
                if tool_name == "get_earnings_transcript":
                    transcript_retrieved = True
                    logger.info("Detected transcript retrieval, routing to store_embeddings")
                    break
        
        # Stay in current stage to process tool results
        if current_stage == "analyzing_query":
            # After query_analyzer tools complete, route to check_embeddings
            # This allows check_embeddings to run and emit "Retrieving Reports" stage update
            # even when transcript was already retrieved via tools
            # (check_embeddings will handle the case when info is missing and route appropriately)
            return "check_embeddings"
        
        # If transcript was retrieved (after check_embeddings), go to store_embeddings
        if transcript_retrieved:
            return "store_embeddings"
        elif current_stage == "checking_embeddings":
            # After tools in checking_embeddings, go back to check_embeddings to re-evaluate
            return "check_embeddings"
        elif current_stage == "retrieving_transcript":
            # If we're in retrieving_transcript stage and haven't retrieved yet, stay here
            # But if we've already retrieved, go to store_embeddings
            if transcript_retrieved:
                return "store_embeddings"
            else:
                return "transcript_retriever"
        else:
            # Default: go back to query_analyzer if we don't know what to do
            logger.warning(f"Unknown current_stage: {current_stage}, routing to query_analyzer")
            return "query_analyzer"
    
    # ==================== Build Graph ====================
    
    workflow = StateGraph(EarningsAgentState)
    
    # Add nodes
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("check_embeddings", check_embeddings)
    workflow.add_node("transcript_retriever", transcript_retriever)
    workflow.add_node("store_embeddings", store_embeddings)
    workflow.add_node("summarizer", summarizer)
    workflow.add_node("tools", tool_node)
    
    # Set entry point
    workflow.set_entry_point("query_analyzer")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "query_analyzer",
        should_use_tools,
        {
            "tools": "tools",
            "check_embeddings": "check_embeddings",
            "end": END
        }
    )
    
    # Add conditional edges from check_embeddings
    workflow.add_conditional_edges(
        "check_embeddings",
        should_use_tools,
        {
            "summarizer": "summarizer",  # Embeddings exist in DB, skip transcript retrieval
            "store_embeddings": "store_embeddings",  # Transcript was just retrieved via tools, need to store
            "transcript_retriever": "transcript_retriever",  # Embeddings don't exist, proceed
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "transcript_retriever",
        should_use_tools,
        {
            "tools": "tools",
            "store_embeddings": "store_embeddings",
            "summarizer": "summarizer",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "tools",
        after_tools,
        {
            "query_analyzer": "query_analyzer",
            "check_embeddings": "check_embeddings",
            "transcript_retriever": "transcript_retriever",
            "store_embeddings": "store_embeddings",
        }
    )
    
    # Store embeddings goes to summarizer
    workflow.add_edge("store_embeddings", "summarizer")
    
    # Summarizer goes to END
    workflow.add_edge("summarizer", END)
    
    # Compile graph
    app = workflow.compile()
    
    return app


async def run_earnings_analysis(company_query: str) -> Dict[str, Any]:
    """Run the earnings analysis pipeline for a company."""
    
    agent = create_earnings_agent()
    
    initial_state: EarningsAgentState = {
        "messages": [HumanMessage(content=f"Please analyze and summarize the earnings reports for: {company_query}")],
        "company_query": company_query,
        "ticker_symbol": None,
        "company_name": None,
        "summary": None,
        "current_stage": "analyzing_query",
        "error": None,
        "requested_fiscal_year": None,
        "requested_quarter": None,
        "transcript_retrieved": False,
    }
    
    # Run the agent
    result = await agent.ainvoke(initial_state)
    
    # Process messages to include all types (AIMessage, HumanMessage, ToolMessage, etc.)
    processed_messages = []
    for m in result.get("messages", []):
        # Handle different message types
        if isinstance(m, AIMessage):
            processed_messages.append({
                "role": "assistant",
                "content": m.content if hasattr(m, 'content') else str(m)
            })
        elif isinstance(m, HumanMessage):
            processed_messages.append({
                "role": "user",
                "content": m.content if hasattr(m, 'content') else str(m)
            })
        elif isinstance(m, SystemMessage):
            processed_messages.append({
                "role": "system",
                "content": m.content if hasattr(m, 'content') else str(m)
            })
        elif isinstance(m, ToolMessage):
            # ToolMessage contains tool execution results - show these as assistant messages
            content = m.content if hasattr(m, 'content') else str(m)
            if content and content.strip():
                # Truncate very long tool results for display, but keep them
                display_content = content[:2000] + "..." if len(content) > 2000 else content
                processed_messages.append({
                    "role": "assistant",
                    "content": f"[Tool Result] {display_content}"
                })
        else:
            # Handle other message types
            if hasattr(m, 'content') and m.content:
                # Tool results should be shown as assistant messages
                processed_messages.append({
                    "role": "assistant",
                    "content": str(m.content) if m.content else ""
                })
            elif hasattr(m, '__str__'):
                # Fallback: convert to string
                content = str(m)
                if content and content.strip():
                    processed_messages.append({
                        "role": "assistant",
                        "content": content
                    })
    
    return {
        "summary": result.get("summary"),
        "messages": processed_messages,
        "stage": result.get("current_stage"),
        "error": result.get("error"),
        "ticker_symbol": result.get("ticker_symbol"),
        "company_name": result.get("company_name"),
        "requested_fiscal_year": result.get("requested_fiscal_year"),
        "requested_quarter": result.get("requested_quarter"),
        "company_query": company_query,  # Also include original query for reference
    }


async def stream_earnings_analysis(company_query: str):
    """Stream the earnings analysis process, yielding updates at each step."""
    
    agent = create_earnings_agent()
    
    initial_state: EarningsAgentState = {
        "messages": [HumanMessage(content=f"Please analyze and summarize the earnings reports for: {company_query}")],
        "company_query": company_query,
        "ticker_symbol": None,
        "company_name": None,
        "summary": None,
        "current_stage": "analyzing_query",
        "error": None,
        "requested_fiscal_year": None,
        "requested_quarter": None,
        "transcript_retrieved": False,
    }
    
    # Stream the agent execution
    final_result = None
    current_state = initial_state.copy()
    async for event in agent.astream(initial_state):
        # Yield each step's updates
        for node_name, node_output in event.items():
            # Update current state with node output (state is cumulative)
            for key, value in node_output.items():
                if key == "messages" and value:
                    # Messages are appended, so merge them
                    if isinstance(value, list):
                        existing_messages = current_state.get(key, [])
                        current_state[key] = list(existing_messages) + value
                else:
                    current_state[key] = value
            
            current_stage = node_output.get("current_stage") or current_state.get("current_stage", "processing")
            
            # Special handling for tools node: if it's retrieving transcripts, use retrieving_transcript stage
            if node_name == "tools" and current_stage == "analyzing_query":
                # Check if previous message (from query_analyzer) has transcript-related tool calls
                # Also check new messages from tools node (ToolMessage) to detect which tools were executed
                all_messages = current_state.get("messages", [])
                new_messages = node_output.get("messages", [])
                
                # Check for transcript-related tool calls in AIMessage
                for msg in reversed(all_messages):
                    if isinstance(msg, AIMessage):
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tc in msg.tool_calls:
                                tool_name = None
                                if isinstance(tc, dict):
                                    tool_name = tc.get('name', '')
                                elif hasattr(tc, 'name'):
                                    tool_name = tc.name
                                if tool_name in ['get_earnings_transcript', 'list_earnings_transcripts']:
                                    # Transcript retrieval/listing is happening, use retrieving_transcript stage
                                    current_stage = "retrieving_transcript"
                                    logger.info(f"Detected transcript operation ({tool_name}) in tools node, using retrieving_transcript stage")
                                    break
                        if current_stage == "retrieving_transcript":
                            break
                
                # Also check ToolMessages from tools node execution to detect transcript operations
                if current_stage != "retrieving_transcript":
                    for msg in new_messages:
                        if isinstance(msg, ToolMessage):
                            tool_name = getattr(msg, 'name', None)
                            if tool_name in ['get_earnings_transcript', 'list_earnings_transcripts']:
                                current_stage = "retrieving_transcript"
                                logger.info(f"Detected transcript operation ({tool_name}) in tools node output, using retrieving_transcript stage")
                                break
            
            summary = node_output.get("summary") or current_state.get("summary")
            
            # Log node execution for debugging
            logger.debug(f"Processing node: {node_name}, current_stage: {current_stage}, has_messages: {bool(node_output.get('messages'))}")
            
            # Extract reasoning from the AI message specific to this node
            # Each node produces its own AIMessage, so we need to find the one from this node
            reasoning = None
            
            # Strategy: Check node_output messages first - these are the messages just produced by this node
            new_messages = node_output.get("messages", [])
            
            # Priority 1: Check new messages from this node (these are definitely from this node)
            messages_to_check = []
            if new_messages:
                # Filter to only AIMessages
                for msg in new_messages:
                    if isinstance(msg, AIMessage):
                        messages_to_check.append(msg)
                        logger.debug(f"Found AIMessage in node_output for {node_name}")
            
            # Priority 2: If no AIMessage in node_output, look in state messages
            # But be smart about which messages belong to which node
            if not messages_to_check:
                all_messages = current_state.get("messages", [])
                
                if node_name == "query_analyzer":
                    # query_analyzer produces the first AIMessage (after HumanMessage)
                    # Look for first AIMessage that doesn't have get_earnings_transcript tool calls
                    for msg in all_messages:
                        if isinstance(msg, AIMessage):
                            # Check if it has transcript tool calls (if so, it's from transcript_retriever)
                            has_transcript_tool = False
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    tool_name = None
                                    if isinstance(tc, dict):
                                        tool_name = tc.get('name', '')
                                    elif hasattr(tc, 'name'):
                                        tool_name = tc.name
                                    if tool_name == 'get_earnings_transcript':
                                        has_transcript_tool = True
                                        break
                            if not has_transcript_tool:
                                messages_to_check.append(msg)
                                break
                
                elif node_name == "transcript_retriever":
                    # transcript_retriever produces AIMessage with get_earnings_transcript tool calls
                    # Priority: Check node_output messages first (these are from this node)
                    if new_messages:
                        for msg in new_messages:
                            if isinstance(msg, AIMessage):
                                messages_to_check.append(msg)
                                logger.debug(f"Found transcript_retriever AIMessage in node_output")
                                break
                    
                    # Fallback: Check state messages for AIMessage with get_earnings_transcript tool calls
                    if not messages_to_check:
                        for msg in reversed(all_messages):
                            if isinstance(msg, AIMessage):
                                # Check if this message has get_earnings_transcript tool calls
                                has_transcript_tool = False
                                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        tool_name = None
                                        if isinstance(tc, dict):
                                            tool_name = tc.get('name', '')
                                        elif hasattr(tc, 'name'):
                                            tool_name = tc.name
                                        if tool_name == 'get_earnings_transcript':
                                            has_transcript_tool = True
                                            break
                                if has_transcript_tool:
                                    messages_to_check.append(msg)
                                    logger.debug(f"Found transcript_retriever AIMessage in state messages")
                                    break
                
                elif node_name == "summarizer":
                    # summarizer produces the last AIMessage without tool calls
                    # Check node_output first (most recent)
                    for msg in reversed(all_messages):
                        if isinstance(msg, AIMessage):
                            # Summarizer messages typically don't have tool calls
                            if not (hasattr(msg, 'tool_calls') and msg.tool_calls):
                                messages_to_check.append(msg)
                                # For summarizer, we want to show a brief reasoning, not the full summary
                                # Extract just the first part as reasoning
                                break
                
                elif node_name == "store_embeddings":
                    # store_embeddings doesn't produce AIMessages, skip reasoning
                    pass
            
            logger.debug(f"Checking {len(messages_to_check)} messages for {node_name}, node_output had {len(new_messages)} messages")
            
            for msg in messages_to_check:
                if isinstance(msg, AIMessage):
                    # Extract reasoning/thinking from AI message content
                    content = msg.content if hasattr(msg, 'content') else ""
                    content_str = str(content) if content else ""
                    
                    # Check if this message has tool calls
                    has_tool_calls = hasattr(msg, 'tool_calls') and msg.tool_calls and len(msg.tool_calls) > 0
                    
                    # Extract tool names if present
                    tool_names = []
                    if has_tool_calls:
                        for tc in msg.tool_calls:
                            if isinstance(tc, dict):
                                tool_name = tc.get('name', 'tool')
                                if tool_name:
                                    tool_names.append(tool_name)
                            elif hasattr(tc, 'name'):
                                if tc.name:
                                    tool_names.append(tc.name)
                    
                    # Build reasoning from content and/or tool calls
                    if content_str and content_str.strip():
                        # Skip very long content (likely tool results)
                        if len(content_str) > 5000:
                            continue
                        
                        # For messages with reasoning but no tool calls
                        if not has_tool_calls:
                            # Plain reasoning message
                            if node_name == "summarizer":
                                # For summarizer, extract a brief reasoning from the start of the summary
                                # Take first 500 chars as reasoning (the summary itself is the output)
                                if len(content_str) > 500:
                                    reasoning = content_str[:500] + "...\n\n[Generating comprehensive summary from transcript]"
                                else:
                                    reasoning = content_str
                            elif len(content_str) < 3000:
                                reasoning = content_str
                        else:
                            # Message with tool calls - prioritize the reasoning content
                            # The content should contain the reasoning, tool calls are just metadata
                            if len(content_str) < 3000:
                                # Use the content as reasoning (it should contain the LLM's explanation)
                                reasoning = content_str
                            elif len(content_str) >= 3000:
                                # Very long content - might be tool results, skip
                                continue
                    elif has_tool_calls and tool_names:
                        # No content but has tool calls - this shouldn't happen with updated prompts
                        # but if it does, show what tools are being called
                        reasoning = f"Preparing to call tools: {', '.join(tool_names)}"
                    
                    # Only use the first reasonable message we find
                    if reasoning:
                        logger.info(f"Extracted reasoning for {node_name} stage {current_stage}: {reasoning[:100]}...")
                        break
            
            # Map internal stages to user-friendly stages
            stage_mapping = {
                "analyzing_query": "analyzing_query",
                "checking_embeddings": "checking_embeddings",
                "retrieving_transcript": "retrieving_transcript",
                "storing_embeddings": "storing_embeddings",
                "complete": "complete",
            }
            
            update = {
                "node": node_name,
                "stage": stage_mapping.get(current_stage, current_stage),
                "has_summary": summary is not None,
                "summary": summary,
                "messages": node_output.get("messages", []),
                "reasoning": reasoning,  # Add reasoning to update
            }
            
            # Always yield update for all nodes (tools node is filtered out in routes.py)
            # This ensures we capture transcript_retriever even if reasoning extraction fails
            logger.info(f"Yielding update for {node_name}: stage={update['stage']}, current_stage={current_stage}, has_reasoning={bool(reasoning)}")
            yield update
            
            # Keep track of final result
            if summary:
                final_result = update

