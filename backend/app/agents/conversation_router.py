"""
Conversation Router Agent - Orchestrates between chat and analysis modes.

This agent determines whether user input should:
1. Trigger a new earnings analysis (tool-using agent)
2. Be handled as a conversational follow-up (chat mode)
3. Be a clarification or refinement of an existing analysis
"""

from typing import Dict, Any, Optional, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.config import get_settings


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
   - Extract the company name/ticker

2. **follow_up_question**: User is asking for MORE DETAIL about the CURRENT analysis
   - Examples: "Tell me about business segments", "What were the key metrics?", 
     "What did management say?", "yes" (when asked if they want more details)
   - This assumes there's already an analysis in the session

3. **clarification**: User wants to REFINE or CHANGE parameters of current analysis
   - Examples: "Actually, I meant the Q2 report", "Can you include more detail?"

4. **general_chat**: General conversation not related to earnings
   - Examples: "How are you?", "What can you do?", "Help"

IMPORTANT: 
- If a user answers "yes", "sure", "okay" etc. to a question about deeper analysis, 
  classify as 'follow_up_question' NOT 'new_analysis'
- Be contextually aware - "yes" after asking about deeper dive = follow_up_question
- Look for company names, ticker symbols for new_analysis
- Short affirmative responses during conversation = follow_up_question

Return your classification with confidence and reasoning."""),
            ("human", "Previous assistant message: {previous_message}\n\nUser input: {user_input}")
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
        
        # Build context
        prev_msg = previous_message or "No previous message"
        
        result = await structured_llm.ainvoke(
            self.classifier_prompt.format_messages(
                user_input=user_input,
                previous_message=prev_msg
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
                    # Look for company/ticker and year in previous user messages
                    ticker = None
                    year = None
                    
                    for msg in reversed(conversation_history):
                        if msg.get("role") == "user":
                            prev_content = msg.get("content", "").strip()
                            # Extract ticker (1-5 uppercase letters, common ticker pattern)
                            if not ticker:
                                ticker_match = re.search(r'\b([A-Z]{1,5})\b', prev_content.upper())
                                if ticker_match:
                                    ticker = ticker_match.group(1)
                            
                            # Extract year
                            if not year:
                                year_match = re.search(r'(?:FY\s*)?(20\d{2})', prev_content)
                                if year_match:
                                    year = year_match.group(1)
                            
                            # If we found both, we can build the query
                            if ticker and year:
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
    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=0.7,
        api_key=settings.openai_api_key,
    )
    
    last_analysis = context.get("last_analysis")
    conversation_history = context.get("conversation_history", [])
    
    if intent == "follow_up_question" and last_analysis:
        # Answer based on existing analysis
        system_msg = f"""You are a helpful assistant analyzing earnings reports.

You previously analyzed a company and provided this summary:

{last_analysis.get('summary', 'No summary available')}

Based on this analysis, answer the user's follow-up question. Be specific and reference 
details from the analysis. If the question asks for information not in the analysis, 
politely explain that and offer to provide what information is available."""
        
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_input)
        ]
        
    elif intent == "general_chat":
        # General conversation
        system_msg = """You are a helpful AI assistant for an earnings analysis system.
        
You can help users:
- Analyze earnings reports for any public company
- Answer questions about earnings summaries
- Provide deeper insights into specific aspects of earnings reports

Be friendly and concise. If users want to analyze a company, ask for the company 
name or ticker symbol."""
        
        messages = [SystemMessage(content=system_msg)]
        
        # Add recent history
        for msg in conversation_history[-4:]:  # Last 4 messages
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        
        messages.append(HumanMessage(content=user_input))
    
    else:
        # Default response
        messages = [
            SystemMessage(content="You are a helpful earnings analysis assistant."),
            HumanMessage(content=user_input)
        ]
    
    response = await llm.ainvoke(messages)
    return response.content




