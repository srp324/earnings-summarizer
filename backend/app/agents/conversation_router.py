"""
Conversation Router Agent - Orchestrates between chat and analysis modes.

This agent determines whether user input should:
1. Trigger a new earnings analysis (tool-using agent)
2. Be handled as a conversational follow-up (chat mode)
3. Be a clarification or refinement of an existing analysis
"""

from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import re
import logging

from app.config import get_settings
from app.tools.investor_relations import TranscriptListTool

logger = logging.getLogger(__name__)


class IntentClassification(BaseModel):
    """Classification of user intent."""
    intent: str = Field(
        description="One of: 'new_analysis', 'follow_up_question', 'clarification', 'general_chat'"
    )
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    reasoning: str = Field(description="Explanation of the classification")
    extracted_company: Optional[str] = Field(
        None,
        description="Extracted company name or ticker if intent is 'new_analysis'"
    )


class ConversationRouter:
    """Routes conversations between analysis and chat modes."""
    
    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.1,  # Low temperature for consistent classification
            api_key=settings.openai_api_key,
        )
        
        self.classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a conversation router for an earnings analysis system.

Your job is to classify user input into one of these categories:

1. **new_analysis**: User wants to analyze a NEW company's earnings
   - Examples: "Apple", "Analyze Tesla", "NVDA", "Tell me about Microsoft's earnings"
   - Also includes when user provides incomplete information that should be combined with previous context
   - Examples: "Q1" after "NVDA 2022" should be classified as new_analysis with full context "NVDA 2022 Q1"
   - Extract the company name/ticker, and if the current input is incomplete (like just "Q1"), 
     combine it with information from the conversation history

2. **follow_up_question**: User is asking for MORE DETAIL about the CURRENT analysis
   - Examples: "Tell me about business segments", "What were the key metrics?", 
     "What did management say?", "yes" (when asked if they want more details)
   - This assumes there's already an analysis in the session

3. **clarification**: User wants to REFINE or CHANGE parameters of current analysis
   - Examples: "Actually, I meant the Q2 report", "Can you include more detail?"

4. **general_chat**: General conversation not related to earnings, OR sector/industry queries
   - Examples: "How are you?", "What can you do?", "Help"
   - Sector names: "technology", "energy", "finance", "healthcare", "retail", "energy", etc.
   - When user responds to a question asking about sectors (for example, "energy" after being asked 
     "which sector?") = general_chat
   - Questions about which companies have recent earnings in a sector = general_chat
   - "Who had recent earnings reports?" with sector specification = general_chat

CRITICAL CONTEXT AWARENESS:
- ALWAYS look at the conversation history to understand context
- Sector names (technology, energy, finance, healthcare, retail, etc.) are NOT companies - 
  classify as 'general_chat' NOT 'new_analysis'
- If the previous assistant message asked "which sector?" or "what sector are you interested in?" 
  and user responds with a sector name (e.g., "energy", "technology"), classify as 'general_chat'
- If user says "Q1", "Q2", etc. and previous messages mention a company and year, 
  this is a continuation/new_analysis that should combine the information
- If user says "NVDA 2022" and then "Q1", the second message should be classified as 
  new_analysis with extracted_company="NVDA 2022 Q1"
- If the previous assistant message asked for a quarter, and user provides just "Q1", 
  extract the company and year from earlier messages and combine them
- If a user answers "yes", "sure", "okay" etc. to a question about deeper analysis, 
  classify as 'follow_up_question' NOT 'new_analysis'
- Be contextually aware - "yes" after asking about deeper dive = follow_up_question
- Look for company names, ticker symbols for new_analysis (NOT sector names)
- Short affirmative responses during conversation = follow_up_question

Return your classification with confidence and reasoning. If the current input is incomplete 
(like just "Q1"), extract the full context from conversation history and include it in extracted_company."""),
            ("human", """Conversation History:
{conversation_history}

Previous assistant message: {previous_message}

Current user input: {user_input}

Based on the conversation history above, classify the user's intent. If the current input is incomplete 
(like "Q1" or "Q2"), look at previous messages to find the company name/ticker and year, and combine 
them in your extracted_company field (e.g., "NVDA 2022 Q1").""")
        ])
    
    async def classify_intent(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]] = None,
        has_active_analysis: bool = False,
        previous_message: Optional[str] = None
    ) -> IntentClassification:
        """
        Classify the user's intent.
        
        Args:
            user_input: The user's message
            conversation_history: Previous conversation messages
            has_active_analysis: Whether there's an active analysis in the session
            previous_message: The immediately previous assistant message (if any)
        
        Returns:
            IntentClassification with intent type and metadata
        """
        
        # Structure the classification request
        structured_llm = self.llm.with_structured_output(IntentClassification)
        
        # Build context from conversation history
        prev_msg = previous_message or "No previous message"
        
        # Format conversation history for the prompt
        history_text = "No previous messages"
        if conversation_history:
            history_lines = []
            for msg in conversation_history[-10:]:  # Last 10 messages for context
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                history_lines.append(f"{role.upper()}: {content}")
            history_text = "\n".join(history_lines)
        
        result = await structured_llm.ainvoke(
            self.classifier_prompt.format_messages(
                user_input=user_input,
                previous_message=prev_msg,
                conversation_history=history_text
            )
        )
        
        # Adjust classification based on context
        if result.intent == "new_analysis" and has_active_analysis:
            # Check if it might actually be a follow-up
            short_affirmative = user_input.lower().strip() in [
                "yes", "yeah", "sure", "ok", "okay", "yep", "yup", "please", "y"
            ]
            if short_affirmative and previous_message and "deeper dive" in previous_message.lower():
                result.intent = "follow_up_question"
                result.reasoning += " (Adjusted: Short affirmative after follow-up question)"
        
        return result
    
    async def route_conversation(
        self,
        user_input: str,
        session_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Route the conversation to appropriate handler.
        
        Args:
            user_input: User's message
            session_data: Current session state including:
                - conversation_history: List of previous messages
                - last_analysis: Most recent analysis result
                - session_id: Session identifier
        
        Returns:
            Dict with:
                - action: 'analyze' or 'chat'
                - intent_classification: IntentClassification object
                - context: Relevant context for the action
        """
        
        conversation_history = session_data.get("conversation_history", [])
        last_analysis = session_data.get("last_analysis")
        has_active_analysis = last_analysis is not None
        
        # Get previous assistant message for context
        previous_message = None
        if conversation_history:
            for msg in reversed(conversation_history):
                if msg.get("role") == "assistant":
                    previous_message = msg.get("content")
                    break
        
        # Classify intent
        classification = await self.classify_intent(
            user_input=user_input,
            conversation_history=conversation_history,
            has_active_analysis=has_active_analysis,
            previous_message=previous_message
        )
        
        # Heuristic fix: reclassify certain general_chat messages as follow-ups
        # when we clearly have an active analysis and the user is asking about
        # metrics/financial details without specifying a new company.
        user_lower = user_input.lower()
        if classification.intent == "general_chat" and has_active_analysis:
            metrics_keywords = ["metric", "metrics", "financial metrics", "latest metrics"]
            if any(keyword in user_lower for keyword in metrics_keywords):
                logger.info(
                    "Reclassifying general_chat as follow_up_question based on "
                    "metrics-related query and existing last_analysis"
                )
                classification.intent = "follow_up_question"

        # Determine action
        if classification.intent == "new_analysis":
            action = "analyze"
            
            # Build complete company query from context
            company_query = user_input
            
            # If extracted_company is provided and different from user_input, use it
            # This handles cases where LLM combined context (e.g., "NVDA 2022 Q1" from "Q1" + history)
            if classification.extracted_company and classification.extracted_company != user_input:
                company_query = classification.extracted_company
            else:
                # Fallback: Try to extract context from conversation history if current input is incomplete
                # Check if user_input looks incomplete (just quarter, just year, etc.)
                user_lower = user_input.strip().upper()
                
                # Patterns that suggest incomplete input
                is_just_quarter = bool(re.match(r'^Q[1-4]$', user_lower))
                is_just_year = bool(re.match(r'^(20\d{2}|FY\s*20\d{2})$', user_lower))
                is_quarter_year = bool(re.match(r'^Q[1-4]\s+(20\d{2}|FY\s*20\d{2})$', user_lower))
                
                if is_just_quarter or is_just_year or is_quarter_year:
                    # If user didn't specify a new ticker, use the ticker from last analysis
                    # This allows queries like "Q2" or "2025 Q2" to use the previous company
                    ticker = None
                    year = None
                    
                    # PRIORITY 1: Use ticker from last_analysis (most reliable)
                    if last_analysis and last_analysis.get('ticker_symbol'):
                        ticker = last_analysis.get('ticker_symbol')
                        logger.info(f"Using ticker_symbol '{ticker}' from last_analysis for incomplete query: {user_input}")
                    
                    # PRIORITY 2: Extract year from last_analysis if available
                    if not year and last_analysis:
                        # Try to get year from last analysis (might not have it, but worth checking)
                        if last_analysis.get('requested_fiscal_year'):
                            year = last_analysis.get('requested_fiscal_year')
                    
                    # PRIORITY 3: Fallback - Look for company/ticker and year in previous user messages
                    if not ticker:
                        for msg in reversed(conversation_history):
                            if msg.get("role") == "user":
                                prev_content = msg.get("content", "").strip()
                                # Extract ticker (1-5 uppercase letters, common ticker pattern)
                                ticker_match = re.search(r'\b([A-Z]{1,5})\b', prev_content.upper())
                                if ticker_match:
                                    ticker = ticker_match.group(1)
                                    break
                    
                    # Extract year from user input or conversation history if not from last_analysis
                    if not year:
                        for msg in reversed(conversation_history):
                            if msg.get("role") == "user":
                                prev_content = msg.get("content", "").strip()
                                # Extract year
                                year_match = re.search(r'(?:FY\s*)?(20\d{2})', prev_content)
                                if year_match:
                                    year = year_match.group(1)
                                    break
                    
                    # Build complete query if we found missing pieces
                    if ticker:
                        if is_just_quarter and year:
                            company_query = f"{ticker} {year} {user_input}"
                        elif is_just_quarter:
                            company_query = f"{ticker} {user_input}"
                        elif is_just_year:
                            company_query = f"{ticker} {user_input}"
                        elif is_quarter_year:
                            # User provided quarter and year, just need ticker
                            company_query = f"{ticker} {user_input}"
                    else:
                        # No ticker found - user needs to specify one
                        # But this should be handled by the classifier/agent, so log warning
                        logger.warning(f"Could not find ticker symbol for incomplete query: {user_input}. User may need to specify company/ticker.")
            
            # Use full company_query to preserve fiscal year/quarter information
            # The agent will extract the company name and fiscal details separately
            context = {
                "company_query": company_query,  # Use full input to preserve "NVDA 2022 Q1" format
                "is_new_analysis": True
            }
        elif classification.intent == "follow_up_question":
            action = "chat"
            context = {
                "last_analysis": last_analysis,
                "question": user_input,
                "conversation_history": conversation_history
            }
        elif classification.intent == "clarification":
            action = "chat"  # Handle as chat for now, could be enhanced
            context = {
                "last_analysis": last_analysis,
                "clarification": user_input,
                "conversation_history": conversation_history
            }
        else:  # general_chat
            action = "chat"
            context = {
                "conversation_history": conversation_history
            }
        
        return {
            "action": action,
            "classification": classification,
            "context": context
        }


async def create_chat_response(
    user_input: str,
    context: Dict[str, Any],
    intent: str
) -> str:
    """
    Generate a chat response based on context.
    
    Args:
        user_input: User's message
        context: Context including analysis results and history
        intent: Classified intent type
    
    Returns:
        AI-generated response
    """
    settings = get_settings()
    
    last_analysis = context.get("last_analysis")
    conversation_history = context.get("conversation_history", [])
    
    # Initialize tools for chat responses (to fetch real earnings data)
    transcript_list_tool = TranscriptListTool()
    chat_tools = [transcript_list_tool]
    
    # Bind tools to LLM for general_chat and default cases
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.7,
        api_key=settings.openai_api_key,
    )
    llm_with_tools = llm.bind_tools(chat_tools) if chat_tools else llm
    
    if intent == "follow_up_question" and last_analysis:
        # Use stored metadata from the analysis (ticker, fiscal year, quarter, company name)
        ticker_symbol = last_analysis.get("ticker_symbol")
        company_name = last_analysis.get("company_name")
        requested_fiscal_year = last_analysis.get("requested_fiscal_year")
        requested_quarter = last_analysis.get("requested_quarter")
        summary = last_analysis.get("summary", "")
        
        # Build context description from stored metadata
        context_parts = []
        if ticker_symbol:
            context_parts.append(f"ticker symbol {ticker_symbol}")
        if company_name:
            context_parts.append(f"company {company_name}")
        elif ticker_symbol:
            # If no company_name but we have ticker, use ticker as identifier
            context_parts.append(f"company ({ticker_symbol})")
        
        if requested_fiscal_year and requested_quarter:
            context_parts.append(f"fiscal year {requested_fiscal_year}, quarter {requested_quarter}")
        elif requested_fiscal_year:
            context_parts.append(f"fiscal year {requested_fiscal_year}")
        
        context_desc = ", ".join(context_parts) if context_parts else last_analysis.get("company_query", "")
        
        # Answer based on existing analysis if we have a summary
        if summary:
            context_info = f"\n\nThe analysis is for {context_desc}." if context_desc else ""
            system_msg = f"""You are a helpful assistant analyzing earnings reports.

There is an earnings report analysis available with the following summary:

{summary}
{context_info}

IMPORTANT: Answer the user's question based on this analysis. You already have the context 
- the company ({context_desc}), the fiscal period, and the detailed summary above.

Answer the user's question directly using information from this analysis. Be specific and 
reference details from the analysis. Do NOT ask which company or earnings report they're 
referring to - you already have this context.

If the question asks for information not in the analysis, politely explain that and offer 
to provide what information is available from the analysis."""
        else:
            # No summary available - use conversation history instead
            system_msg = """You are a helpful assistant analyzing earnings reports.

Answer the user's question based on the conversation history below. Use the context from 
previous messages to understand what earnings report is being discussed."""
        
        messages = [SystemMessage(content=system_msg)]
        
        # Add recent conversation history for context
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=user_input))
        
        # Use regular LLM for follow-up questions (no tools needed)
        response = await llm.ainvoke(messages)
        return response.content
        
    elif intent == "general_chat":
        # General conversation with tool access
        system_msg = """You are a helpful AI assistant for an earnings analysis system.
        
You can help users:
- Analyze earnings reports for any public company (when they provide a company name or ticker)
- Answer questions about earnings summaries (if they've already been generated)
- Provide deeper insights into specific aspects of earnings reports (if they've already been analyzed)
- Find which companies have recent earnings reports available

IMPORTANT: You have access to the list_earnings_transcripts tool that can fetch real earnings 
transcript data from discountingcashflows.com for any ticker symbol.

When asked about recent earnings reports (e.g., "Who had recent earning reports?" or "technology companies"):
1. Use the list_earnings_transcripts tool to check recent earnings for relevant companies
2. For sector-based questions (like "technology"), identify major companies in that sector:
   - Technology: AAPL, MSFT, GOOGL, NVDA, META, TSLA, NFLX, AMD, INTC, CRM, ORCL, ADBE, IBM, CSCO
   - Finance: JPM, BAC, GS, MS, C, WFC
   - Retail: AMZN, WMT, TGT, COST
   - Healthcare: JNJ, PFE, UNH, ABBV, MRK
   - etc.
3. Call list_earnings_transcripts for each relevant company to get real, up-to-date data
4. Aggregate the results and present which companies have recent earnings reports

CRITICAL RULES:
1. DO NOT make up or guess specific earnings report dates - always use the tool to get real relevant data
2. When asked about recent earnings, use list_earnings_transcripts to fetch actual data
3. If you don't know which companies to check, ask the user to specify or check major companies in the mentioned sector
4. Present the real data from the tool results, not fabricated information
5. When offering additional information about a company's earnings, refer to generating a "summary" or 
   "comprehensive summary" of the earnings report. DO NOT mention providing "the full transcript" - 
   instead, say you can generate a comprehensive summary if they provide the company name or ticker symbol

Be friendly and concise. Use tools to get real data rather than guessing."""
        
        messages = [SystemMessage(content=system_msg)]
        
        # Add recent history
        for msg in conversation_history[-4:]:  # Last 4 messages
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=user_input))
        
        # Use LLM with tools and handle tool calls
        max_iterations = 5  # Limit tool call iterations
        iteration = 0
        while iteration < max_iterations:
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            
            # Check if LLM wants to use tools
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Execute tool calls
                for tool_call in response.tool_calls:
                    # Handle both dict and object formats
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args", {})
                        tool_call_id = tool_call.get("id", "")
                    else:
                        tool_name = getattr(tool_call, "name", "")
                        tool_args = getattr(tool_call, "args", {})
                        tool_call_id = getattr(tool_call, "id", "")
                    
                    if tool_name == "list_earnings_transcripts":
                        # Extract symbol from args
                        if isinstance(tool_args, dict):
                            symbol = tool_args.get("symbol", "")
                        else:
                            symbol = str(tool_args) if tool_args else ""
                        
                        try:
                            tool_result = transcript_list_tool.run(symbol)
                            tool_message = ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                            messages.append(tool_message)
                        except Exception as e:
                            error_message = ToolMessage(
                                content=f"Error calling tool: {str(e)}",
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                            messages.append(error_message)
                
                iteration += 1
            else:
                # No more tool calls, return the response
                return response.content
        
        # If we hit max iterations, return the last response
        return messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])
    
    else:
        # Default response with conversation history and tool access
        system_msg = """You are a helpful earnings analysis assistant.

You have access to the list_earnings_transcripts tool to fetch real earnings transcript 
data for any ticker symbol. When asked about recent earnings reports, use this tool to 
get real data rather than making up dates or information.

When offering to provide more information about a company's earnings, refer to generating 
a "summary" or "comprehensive summary" - DO NOT mention providing "the full transcript"."""
        
        messages = [SystemMessage(content=system_msg)]
        
        # Add recent conversation history for context
        for msg in conversation_history[-6:]:  # Last 6 messages for context
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=user_input))
        
        # Use LLM with tools and handle tool calls
        max_iterations = 5  # Limit tool call iterations
        iteration = 0
        while iteration < max_iterations:
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)
            
            # Check if LLM wants to use tools
            if hasattr(response, "tool_calls") and response.tool_calls:
                # Execute tool calls
                for tool_call in response.tool_calls:
                    # Handle both dict and object formats
                    if isinstance(tool_call, dict):
                        tool_name = tool_call.get("name", "")
                        tool_args = tool_call.get("args", {})
                        tool_call_id = tool_call.get("id", "")
                    else:
                        tool_name = getattr(tool_call, "name", "")
                        tool_args = getattr(tool_call, "args", {})
                        tool_call_id = getattr(tool_call, "id", "")
                    
                    if tool_name == "list_earnings_transcripts":
                        # Extract symbol from args
                        if isinstance(tool_args, dict):
                            symbol = tool_args.get("symbol", "")
                        else:
                            symbol = str(tool_args) if tool_args else ""
                        
                        try:
                            tool_result = transcript_list_tool.run(symbol)
                            tool_message = ToolMessage(
                                content=str(tool_result),
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                            messages.append(tool_message)
                        except Exception as e:
                            error_message = ToolMessage(
                                content=f"Error calling tool: {str(e)}",
                                tool_call_id=tool_call_id,
                                name=tool_name
                            )
                            messages.append(error_message)
                
                iteration += 1
            else:
                # No more tool calls, return the response
                return response.content
        
        # If we hit max iterations, return the last response
        return messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])




