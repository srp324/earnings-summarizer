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
                year_match = re.search(r'FY?\s*(\d{4})', user_query, re.I)
                quarter_match = re.search(r'Q\s*([1-4])', user_query, re.I)
                if year_match:
                    requested_fiscal_year = year_match.group(1)
                    logger.info(f"Extracted fiscal year: {requested_fiscal_year}")
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
    
    def transcript_retriever(state: EarningsAgentState) -> Dict[str, Any]:
        """Retrieve the earnings transcript content."""
        
        # Get requested fiscal year and quarter from state if available
        requested_fiscal_year = state.get("requested_fiscal_year")
        requested_quarter = state.get("requested_quarter")
        
        logger.info(f"transcript_retriever: requested_fiscal_year={requested_fiscal_year}, requested_quarter={requested_quarter}")
        
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
    
    
    def summarizer(state: EarningsAgentState) -> Dict[str, Any]:
        """Generate comprehensive summary of the earnings report."""
        
        # Get all messages to find transcript content
        messages_list = list(state["messages"])
        
        # Find transcript content in the messages
        transcript_content = None
        for msg in reversed(messages_list):
            if hasattr(msg, 'content') and msg.content:
                content_str = str(msg.content)
                # Look for transcript indicators
                if "Earnings Call Transcript" in content_str or "discountingcashflows.com" in content_str:
                    # Make sure it's not just the header/template
                    if len(content_str) > 1000 and not any(template in content_str.lower() for template in [
                        'market is open', 'after-hours quote', 'last quote from'
                    ]):
                        transcript_content = content_str
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
        
        # Include all messages (which contain the transcript from tool execution)
        messages_list = list(state["messages"])
        messages = [SystemMessage(content=system_prompt)] + messages_list
        
        # Add explicit instruction to summarize
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
            return "summarizer"
        else:
            return "end"
    
    def after_tools(state: EarningsAgentState) -> str:
        """Determine next step after tool execution."""
        current_stage = state.get("current_stage", "analyzing_query")
        
        # Check if we've already retrieved a transcript
        transcript_retrieved = state.get("transcript_retrieved", False)
        
        # Check if any tool calls were for get_earnings_transcript
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        # If we just retrieved a transcript, mark it and move to summarizer
        if last_message and hasattr(last_message, "tool_calls"):
            for tool_call in getattr(last_message, "tool_calls", []):
                if tool_call.get("name") == "get_earnings_transcript":
                    transcript_retrieved = True
                    break
        
        # If transcript was retrieved, go to summarizer
        if transcript_retrieved:
            return "summarizer"
        
        # Stay in current stage to process tool results
        if current_stage == "analyzing_query":
            return "query_analyzer"
        elif current_stage == "retrieving_transcript":
            # If we're in retrieving_transcript stage and haven't retrieved yet, stay here
            # But if we've already retrieved, go to summarizer
            return "summarizer" if transcript_retrieved else "transcript_retriever"
        else:
            return "summarizer"
    
    # ==================== Build Graph ====================
    
    workflow = StateGraph(EarningsAgentState)
    
    # Add nodes
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("transcript_retriever", transcript_retriever)
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
            "summarizer": "summarizer",
        }
    )
    
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


def stream_earnings_analysis(company_query: str):
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
    for event in agent.stream(initial_state):
        # Yield each step's updates
        for node_name, node_output in event.items():
            yield {
                "node": node_name,
                "stage": node_output.get("current_stage", "processing"),
                "has_summary": node_output.get("summary") is not None,
            }
    
    # Return final result
    return agent.invoke(initial_state)

