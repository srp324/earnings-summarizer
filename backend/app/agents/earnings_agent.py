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
    ir_url: Optional[str]
    earnings_links: List[Dict[str, str]]
    parsed_documents: List[Dict[str, str]]
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
        
        system_prompt = """You are an expert at understanding user queries about companies and stocks.
        
Your task is to:
1. Identify the ticker symbol from the user's query
2. Use the list_earnings_transcripts tool with the ticker symbol to get available transcripts from discountingcashflows.com

Common ticker symbols:
- AAPL = Apple
- MSFT = Microsoft
- GOOGL/GOOG = Google/Alphabet
- AMZN = Amazon
- META = Meta (Facebook)
- NVDA = NVIDIA
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
Then immediately use list_earnings_transcripts to get the available earnings transcripts.
"""
        
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "current_stage": "analyzing_query",
            "requested_fiscal_year": requested_fiscal_year,
            "requested_quarter": requested_quarter
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
                # Remove the header metadata
                lines = transcript_content.split('\n')
                start_idx = 0
                for i, line in enumerate(lines):
                    if "---" in line or "Earnings Call Transcript" in line:
                        # Start content after the separator
                        start_idx = i + 1
                        break
                transcript_body = '\n'.join(lines[start_idx:])
                
                # Remove trailing metadata
                if "---" in transcript_body:
                    transcript_body = transcript_body.split("---")[0].strip()
            
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
            
            # If we have all required information, store embeddings
            if transcript_body and ticker_symbol and requested_fiscal_year and requested_quarter:
                try:
                    rag_service = get_rag_service()
                    
                    # Get company name (default to ticker if not available)
                    company_name = state.get("company_name") or ticker_symbol
                    
                    # Store embeddings
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

CRITICAL RULES:
1. Call get_earnings_transcript EXACTLY ONCE - do NOT call it multiple times
2. When the user specifies a fiscal year and quarter, you MUST extract them correctly and use ONLY those values
3. Do NOT fetch multiple transcripts - fetch ONLY the one the user requested

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
- Use the most recent transcript (first one in the list)
- Still call the tool EXACTLY ONCE

DO NOT:
- Call the tool multiple times
- Fetch multiple transcripts
- Iterate through quarters
- Use different values than what the user specified

After calling get_earnings_transcript ONCE, stop. Do not make any more tool calls.
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
                    summary_query = "financial highlights revenue earnings per share EPS profitability margins free cash flow business segments performance metrics operational highlights management commentary strategic outlook guidance risks challenges announcements"
                    
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
            return "transcript_retriever"
        elif current_stage == "retrieving_transcript":
            return "store_embeddings"
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
        
        # If transcript was retrieved, go to store_embeddings first, then summarizer
        if transcript_retrieved:
            return "store_embeddings"
        
        # Stay in current stage to process tool results
        if current_stage == "analyzing_query":
            return "query_analyzer"
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
            "transcript_retriever": "transcript_retriever",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "transcript_retriever",
        should_use_tools,
        {
            "tools": "tools",
            "summarizer": "summarizer",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "tools",
        after_tools,
        {
            "query_analyzer": "query_analyzer",
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
        "ir_url": None,
        "earnings_links": [],
        "parsed_documents": [],
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
    }


async def stream_earnings_analysis(company_query: str):
    """Stream the earnings analysis process, yielding updates at each step."""
    
    agent = create_earnings_agent()
    
    initial_state: EarningsAgentState = {
        "messages": [HumanMessage(content=f"Please analyze and summarize the earnings reports for: {company_query}")],
        "company_query": company_query,
        "ticker_symbol": None,
        "company_name": None,
        "ir_url": None,
        "earnings_links": [],
        "parsed_documents": [],
        "summary": None,
        "current_stage": "analyzing_query",
        "error": None,
        "requested_fiscal_year": None,
        "requested_quarter": None,
        "transcript_retrieved": False,
    }
    
    # Stream the agent execution
    final_result = None
    async for event in agent.astream(initial_state):
        # Yield each step's updates
        for node_name, node_output in event.items():
            current_stage = node_output.get("current_stage", "processing")
            summary = node_output.get("summary")
            
            # Map internal stages to user-friendly stages
            stage_mapping = {
                "analyzing_query": "analyzing_query",
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
            }
            
            yield update
            
            # Keep track of final result
            if summary:
                final_result = update

