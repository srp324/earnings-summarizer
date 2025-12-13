"""
Multi-agent system for earnings report summarization using LangGraph and web scraping.

Flow:
1. Query Analyzer Agent - Identifies ticker symbol and lists available transcripts via discountingcashflows.com
2. Transcript Retriever Agent - Retrieves the full earnings call transcript by scraping discountingcashflows.com
3. Summarizer Agent - Generates comprehensive summary from the transcript
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
from app.tools.investor_relations import TranscriptListTool, TranscriptTool


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
            "current_stage": "analyzing_query"
        }
    
    def transcript_retriever(state: EarningsAgentState) -> Dict[str, Any]:
        """Retrieve the earnings transcript content."""
        
        system_prompt = """You are an expert at retrieving earnings transcripts from discountingcashflows.com.

Once you have the list of available transcripts, select the most appropriate one based on the user's request:
- If they want the most recent, choose the first one (most recent fiscal year and quarter)
- If they specify a quarter/year, choose the matching one
- Use the get_earnings_transcript tool with:
  - symbol: The ticker symbol (e.g., "NVDA")
  - fiscal_year: The fiscal year as a string (e.g., "2025")
  - quarter: The quarter number as a string "1", "2", "3", or "4"

The transcript will contain the complete earnings call with all speakers and their statements scraped from discountingcashflows.com.

After retrieving the transcript, you do not need to do anything else - the system will automatically summarize it.
"""
        
        messages_list = list(state["messages"])
        messages = [SystemMessage(content=system_prompt)] + messages_list
        
        response = llm_with_tools.invoke(messages)
        
        return {
            "messages": [response],
            "current_stage": "retrieving_transcript"
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
        
        # Stay in current stage to process tool results
        if current_stage == "analyzing_query":
            return "query_analyzer"
        elif current_stage == "retrieving_transcript":
            return "transcript_retriever"
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

