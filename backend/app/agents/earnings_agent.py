"""
Multi-agent system for earnings report summarization using LangGraph.

Flow:
1. Query Analyzer Agent - Understands user query and extracts company info
2. IR Finder Agent - Finds the investor relations site
3. Document Extractor Agent - Extracts earnings report links
4. Document Parser Agent - Parses and extracts content from reports
5. Summarizer Agent - Generates comprehensive summary
"""

from typing import TypedDict, Annotated, Sequence, List, Optional, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
import operator
import json

from app.config import get_settings
from app.tools.web_search import WebSearchTool, URLFetchTool
from app.tools.investor_relations import InvestorRelationsTool, ExtractEarningsLinksTool
from app.tools.document_parser import DocumentParserTool, HTMLDocumentTool


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


def create_earnings_agent():
    """Create the multi-agent LangGraph for earnings summarization."""
    
    settings = get_settings()
    
    # Initialize LLM
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key,
    )
    
    # Initialize tools
    web_search_tool = WebSearchTool()
    url_fetch_tool = URLFetchTool()
    ir_finder_tool = InvestorRelationsTool()
    earnings_links_tool = ExtractEarningsLinksTool()
    pdf_parser_tool = DocumentParserTool()
    html_parser_tool = HTMLDocumentTool()
    
    all_tools = [
        web_search_tool,
        url_fetch_tool,
        ir_finder_tool,
        earnings_links_tool,
        pdf_parser_tool,
        html_parser_tool,
    ]
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(all_tools)
    
    # Create tool node
    tool_node = ToolNode(all_tools)
    
    # ==================== Agent Nodes ====================
    
    def query_analyzer(state: EarningsAgentState) -> Dict[str, Any]:
        """Analyze user query and extract company information."""
        
        system_prompt = """You are an expert at understanding user queries about companies and stocks.
        
Your task is to:
1. Identify the company name from the user's query
2. Identify the ticker symbol if provided
3. Determine if the user wants a specific type of earnings report (quarterly, annual, recent, etc.)

If the user only provides a ticker symbol, you should recognize common ones:
- AAPL = Apple
- MSFT = Microsoft
- GOOGL/GOOG = Google/Alphabet
- AMZN = Amazon
- META = Meta (Facebook)
- NVDA = NVIDIA
- TSLA = Tesla

Respond with a brief analysis and then use the find_investor_relations tool to locate the company's IR site.
"""
        
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "current_stage": "analyzing_query"
        }
    
    def ir_finder(state: EarningsAgentState) -> Dict[str, Any]:
        """Find the investor relations site for the company."""
        
        system_prompt = """You are an expert at finding investor relations websites.

Based on the company identified, use the find_investor_relations tool to locate the official IR site.
Then use the extract_earnings_links tool to find the earnings reports.

Focus on finding:
1. The official investor relations page (usually investor.company.com or company.com/investors)
2. Links to recent earnings reports (10-K, 10-Q, quarterly earnings)
3. Earnings call transcripts if available

Prioritize the most recent reports (current fiscal year).
"""
        
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "current_stage": "finding_ir_site"
        }
    
    def document_parser(state: EarningsAgentState) -> Dict[str, Any]:
        """Parse and extract content from earnings documents."""
        
        system_prompt = """You are an expert at parsing financial documents.

Based on the earnings links found, use the parse_pdf tool to extract content from PDF earnings reports.
If the document is HTML-based, use the parse_html_document tool instead.

Focus on parsing:
1. The most recent quarterly earnings report (10-Q)
2. The most recent annual report (10-K) if available
3. Any earnings call transcripts

Parse ONE document at a time to avoid overwhelming the system.
Start with the most recent quarterly report if available.
"""
        
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "current_stage": "parsing_documents"
        }
    
    def summarizer(state: EarningsAgentState) -> Dict[str, Any]:
        """Generate comprehensive summary of the earnings report."""
        
        system_prompt = """You are an expert financial analyst specializing in earnings report analysis.

Based on the parsed earnings documents, provide a comprehensive summary covering:

## Summary Structure

### 1. Company Overview
- Brief description of what the company does
- Fiscal period covered in the report

### 2. Key Financial Highlights
- Revenue (total and year-over-year change)
- Net Income / Earnings Per Share (EPS)
- Gross Margin and Operating Margin
- Free Cash Flow

### 3. Business Segment Performance
- Performance breakdown by business segment/product line
- Notable growth or decline areas

### 4. Key Metrics & KPIs
- Company-specific metrics that matter (e.g., MAU for social media, subscribers for streaming)
- Customer/user growth
- Market share information if available

### 5. Management Commentary & Outlook
- Key points from management discussion
- Forward guidance if provided
- Major initiatives or strategic changes

### 6. Risks & Challenges
- Key risk factors mentioned
- Competitive pressures
- Regulatory or market challenges

### 7. Notable Events
- Acquisitions, divestitures, or restructuring
- New product launches
- Leadership changes

Provide specific numbers and percentages where available. Be concise but comprehensive.
If certain information is not available in the parsed documents, note that.
"""
        
        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        
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
            return "ir_finder"
        elif current_stage == "finding_ir_site":
            return "document_parser"
        elif current_stage == "parsing_documents":
            return "summarizer"
        else:
            return "end"
    
    def after_tools(state: EarningsAgentState) -> str:
        """Determine next step after tool execution."""
        current_stage = state.get("current_stage", "analyzing_query")
        
        # Stay in current stage to process tool results
        if current_stage == "analyzing_query":
            return "query_analyzer"
        elif current_stage == "finding_ir_site":
            return "ir_finder"
        elif current_stage == "parsing_documents":
            return "document_parser"
        else:
            return "summarizer"
    
    # ==================== Build Graph ====================
    
    workflow = StateGraph(EarningsAgentState)
    
    # Add nodes
    workflow.add_node("query_analyzer", query_analyzer)
    workflow.add_node("ir_finder", ir_finder)
    workflow.add_node("document_parser", document_parser)
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
            "ir_finder": "ir_finder",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "ir_finder",
        should_use_tools,
        {
            "tools": "tools",
            "document_parser": "document_parser",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "document_parser",
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
            "ir_finder": "ir_finder",
            "document_parser": "document_parser",
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
    }
    
    # Run the agent
    result = await agent.ainvoke(initial_state)
    
    return {
        "summary": result.get("summary"),
        "messages": [
            {"role": "assistant" if isinstance(m, AIMessage) else "user" if isinstance(m, HumanMessage) else "system", 
             "content": m.content if hasattr(m, 'content') else str(m)}
            for m in result.get("messages", [])
            if hasattr(m, 'content') and m.content
        ],
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

