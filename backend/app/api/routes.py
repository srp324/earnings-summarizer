"""API routes for earnings summarization."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
import json
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator, Optional

from app.schemas import (
    EarningsRequest, 
    EarningsResponse, 
    HealthResponse,
    AnalysisStatus,
    StreamUpdate,
    Message,
    SessionListResponse,
    ChatRequest,
    ChatResponse,
    SearchHistoryResponse,
    SearchHistoryEntry,
)
from app.database import get_db, SearchSession
from app.agents.earnings_agent import run_earnings_analysis, stream_earnings_analysis
from app.agents.conversation_router import ConversationRouter, create_chat_response
from app.session_manager import get_session_manager
from app.services.financial_scraper import scrape_company_financials
from app.services.metrics_service import store_scraped_metrics, get_metrics, get_metrics_history
from langchain_core.messages import AIMessage
import asyncio


async def scrape_and_store_metrics(
    ticker_symbol: str,
    company_name: str,
    session_id: str
):
    """Helper function to scrape and store metrics in background."""
    try:
        logger.info(f"Starting background scrape for {ticker_symbol}")
        # Create a new database session for the background task
        from app.database import AsyncSessionLocal
        async with AsyncSessionLocal() as db:
            scraped_data = await scrape_company_financials(ticker_symbol.upper())
            if scraped_data and any(scraped_data.values()):
                await store_scraped_metrics(
                    db=db,
                    ticker_symbol=ticker_symbol.upper(),
                    company_name=company_name,
                    scraped_data=scraped_data,
                    session_id=session_id
                )
                logger.info(f"Successfully scraped and stored metrics for {ticker_symbol}")
    except Exception as e:
        logger.error(f"Error in background metrics scrape for {ticker_symbol}: {e}", exc_info=True)

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize conversation router
conversation_router = ConversationRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse()


@router.post("/analyze", response_model=EarningsResponse)
async def analyze_earnings(
    request: EarningsRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Analyze earnings reports for a company.
    
    This endpoint triggers the multi-agent analysis pipeline:
    1. Identifies the company from the query
    2. Finds the investor relations website
    3. Extracts links to earnings reports
    4. Parses the documents
    5. Generates a comprehensive summary
    """
    session_id = str(uuid.uuid4())
    
    try:
        # Create session record
        session = SearchSession(
            session_id=session_id,
            company_query=request.company_query,
            status="searching",
        )
        db.add(session)
        await db.commit()
        
        # Run the analysis
        result = await run_earnings_analysis(request.company_query)
        
        # Update session with results
        session.status = "complete" if result.get("summary") else "error"
        session.summary = result.get("summary")
        await db.commit()
        
        return EarningsResponse(
            session_id=session_id,
            company_query=request.company_query,
            status=AnalysisStatus.COMPLETE if result.get("summary") else AnalysisStatus.ERROR,
            summary=result.get("summary"),
            messages=[
                Message(role=m["role"], content=m["content"])
                for m in result.get("messages", [])
            ],
            error=result.get("error"),
        )
        
    except Exception as e:
        # Update session with error
        session.status = "error"
        await db.commit()
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze/stream")
async def analyze_earnings_stream(
    request: EarningsRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Stream earnings analysis with real-time updates.
    
    Returns Server-Sent Events (SSE) with progress updates.
    """
    session_id = str(uuid.uuid4())
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events during analysis."""
        try:
            # Initial event
            yield f"data: {json.dumps({'session_id': session_id, 'stage': 'starting', 'message': 'Starting analysis...'})}\n\n"
            
            # Create session record
            session = SearchSession(
                session_id=session_id,
                company_query=request.company_query,
                status="searching",
            )
            db.add(session)
            await db.commit()
            
            # Run analysis (non-streaming endpoint - consider using /chat/stream for real-time updates)
            try:
                result = await run_earnings_analysis(request.company_query)
                
                # Send final result
                final_update = {
                    "session_id": session_id,
                    "stage": "complete",
                    "message": "Analysis complete!",
                    "summary": result.get("summary"),
                    "is_complete": True,
                }
                yield f"data: {json.dumps(final_update)}\n\n"
                
                # Update database
                session.status = "complete"
                session.summary = result.get("summary")
                await db.commit()
                
            except Exception as e:
                error_update = {
                    "session_id": session_id,
                    "stage": "error",
                    "message": str(e),
                    "error": str(e),
                    "is_complete": True,
                }
                yield f"data: {json.dumps(error_update)}\n\n"
                
                session.status = "error"
                await db.commit()
                
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e), 'is_complete': True})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(
    limit: int = 20,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """List recent analysis sessions."""
    result = await db.execute(
        select(SearchSession)
        .order_by(SearchSession.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    sessions = result.scalars().all()
    
    return SessionListResponse(
        sessions=[
            {
                "session_id": s.session_id,
                "company_query": s.company_query,
                "status": s.status,
                "created_at": s.created_at.isoformat() if s.created_at else None,
            }
            for s in sessions
        ],
        total=len(sessions),
    )


@router.get("/sessions/{session_id}", response_model=EarningsResponse)
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Get a specific analysis session."""
    result = await db.execute(
        select(SearchSession).where(SearchSession.session_id == session_id)
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return EarningsResponse(
        session_id=session.session_id,
        company_query=session.company_query,
        status=AnalysisStatus(session.status) if session.status in [s.value for s in AnalysisStatus] else AnalysisStatus.PENDING,
        summary=session.summary,
        messages=[],
        created_at=session.created_at,
    )


@router.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Streaming version of chat endpoint that provides real-time updates during analysis.
    
    Returns Server-Sent Events (SSE) with progress updates.
    """
    
    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events during chat/analysis."""
        try:
            # Get or create session
            # If request.session_id is None, a new session will be created with fresh state
            # This happens when user goes back to homepage and starts a new conversation
            session_manager = get_session_manager()
            
            # If session_id is None or empty, always create a new session
            # This ensures each new conversation gets its own session and search history
            if not request.session_id:
                session = session_manager.get_or_create_session(None)  # None forces new session creation
                logger.info(f"Created new session for new conversation: {session.session_id}")
            else:
                session = session_manager.get_or_create_session(request.session_id)
                logger.info(f"Using existing session: {session.session_id}")
            
            # Add user message to history
            session.add_message("user", request.message)
            
            # Send initial event
            yield f"data: {json.dumps({'type': 'status', 'session_id': session.session_id, 'stage': 'routing', 'message': 'Analyzing your request...'})}\n\n"
            
            # Route the conversation
            routing_result = await conversation_router.route_conversation(
                user_input=request.message,
                session_data={
                    "conversation_history": session.conversation_history,
                    "last_analysis": session.last_analysis,
                    "session_id": session.session_id,
                }
            )
            
            action = routing_result["action"]
            classification = routing_result["classification"]
            context = routing_result["context"]
            
            if action == "analyze":
                # Trigger new earnings analysis with streaming
                company_query = context.get("company_query", request.message)
                
                # Check if database session already exists
                result = await db.execute(
                    select(SearchSession).where(SearchSession.session_id == session.session_id)
                )
                db_session = result.scalar_one_or_none()
                
                if db_session:
                    # Update existing session
                    db_session.company_query = company_query
                    db_session.status = "searching"
                    db_session.summary = None  # Clear previous summary
                    db_session.updated_at = datetime.utcnow()
                else:
                    # Create new database session record
                    db_session = SearchSession(
                        session_id=session.session_id,
                        company_query=company_query,
                        status="searching",
                    )
                    db.add(db_session)
                
                await db.commit()
                
                # Map agent stages to frontend stages
                stage_mapping = {
                    "analyzing_query": {"id": "analyzing", "label": "Analyzing Query"},
                    "checking_embeddings": {"id": "searching", "label": "Retrieving Reports"},
                    "retrieving_transcript": {"id": "searching", "label": "Retrieving Reports"},
                    "storing_embeddings": {"id": "summarizing", "label": "Generating Summary"},
                    "complete": {"id": "summarizing", "label": "Generating Summary"},
                }
                
                # Stream analysis with real-time updates
                # If user didn't specify a new ticker, use ticker from last analysis
                # The conversation router should have already added it to company_query, but we'll pass it to the agent as well
                initial_ticker = None
                if session.last_analysis and session.last_analysis.get('ticker_symbol'):
                    # Check if company_query actually contains a ticker (not just year/quarter patterns)
                    # If query looks like just period info (e.g., "2025 Q2", "Q2"), use last_analysis ticker
                    import re
                    query_upper = company_query.upper().strip()
                    # Patterns that indicate just period without ticker
                    is_just_period = bool(re.match(r'^(FY\s*)?\d{4}\s*(Q[1-4])?$|^Q[1-4](\s+(FY\s*)?\d{4})?$', query_upper))
                    if is_just_period:
                        initial_ticker = session.last_analysis.get('ticker_symbol')
                        logger.info(f"Company query '{company_query}' appears to be just period without ticker, will use '{initial_ticker}' from last_analysis in agent state")
                    else:
                        # Query has something else - check if it's actually a ticker pattern
                        # Look for ticker-like patterns (1-5 uppercase letters, not part of FY/Q pattern)
                        ticker_match = re.search(r'\b([A-Z]{1,5})\b', query_upper)
                        if not ticker_match:
                            initial_ticker = session.last_analysis.get('ticker_symbol')
                            logger.info(f"Company query '{company_query}' doesn't contain clear ticker, will use '{initial_ticker}' from last_analysis in agent state")
                
                final_result = None
                last_mapped_stage = None  # Track the last mapped stage (with id and label)
                current_active_stage_id = None  # Track the currently active stage ID
                ticker_from_tool_call = None  # Capture ticker from actual tool invocation
                stage_reasoning = {}  # Track reasoning for each stage: {stage_id: reasoning}
                async for update in stream_earnings_analysis(company_query, initial_ticker_symbol=initial_ticker):
                    # Extract ticker from tool calls in updates (most reliable - what tool actually used)
                    if update.get("messages"):
                        import re
                        from langchain_core.messages import ToolMessage
                        for msg in update.get("messages", []):
                            # Check AIMessage with tool_calls for get_earnings_transcript
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
                                    if tool_name == 'get_earnings_transcript':
                                        tool_args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                                        symbol = tool_args.get('symbol') if isinstance(tool_args, dict) else None
                                        if symbol:
                                            ticker_from_tool_call = symbol.upper()
                                            logger.info(f"Captured ticker_symbol '{ticker_from_tool_call}' from get_earnings_transcript tool call in stream update")
                                            break
                                if ticker_from_tool_call:
                                    break
                    
                    # Preserve ticker_symbol from any update that has it
                    if update.get("ticker_symbol") and not final_result:
                        final_result = {}
                    stage = update.get("stage", "processing")
                    node = update.get("node", "")
                    reasoning = update.get("reasoning")  # Extract reasoning from update
                    
                    # Log all updates from stream for debugging
                    logger.info(f"Received update from stream: node={node}, stage={stage}")
                    
                    # For "tools" node, only skip if it's not retrieving transcripts
                    # Transcript retrieval should show "Retrieving Reports" stage
                    if node == "tools":
                        # Check if stage is "retrieving_transcript" - if so, process it to show "Retrieving Reports"
                        if stage == "retrieving_transcript":
                            # This is transcript retrieval, process it to show "Retrieving Reports"
                            logger.debug(f"Processing tools node update for transcript retrieval (stage: {stage})")
                            # Continue processing - don't skip
                        else:
                            # Not transcript retrieval, skip it
                            logger.debug(f"Skipping stage update for tools node (stage: {stage}, previous stage '{current_active_stage_id}' remains active)")
                            continue
                    
                    # Skip if stage is "processing" and not in our mapping (invalid stage)
                    if stage == "processing" and stage not in stage_mapping:
                        logger.debug(f"Skipping stage update for invalid stage: {stage}")
                        continue
                    
                    # Map stage to frontend stage
                    mapped_stage = stage_mapping.get(stage, {"id": stage, "label": stage.replace("_", " ").title()})
                    current_stage_id = mapped_stage.get('id')
                    
                    # Mark previous stage as complete when moving to a new stage
                    if last_mapped_stage and last_mapped_stage.get('id') != current_stage_id:
                        complete_update_data = {
                            'type': 'stage_update',
                            'session_id': session.session_id,
                            'stage_id': last_mapped_stage.get('id'),
                            'stage_label': last_mapped_stage.get('label'),
                            'status': 'complete'
                        }
                        yield f"data: {json.dumps(complete_update_data)}\n\n"
                    
                    # Update last mapped stage and current active stage
                    last_mapped_stage = mapped_stage
                    current_active_stage_id = mapped_stage.get('id')
                    
                    # Log all stage updates for debugging
                    logger.info(f"Processing stage update: node={node}, stage={stage}, mapped_stage={mapped_stage.get('id')}, mapped_label={mapped_stage.get('label')}, has_reasoning={bool(reasoning)}")
                    
                    # Enhance reasoning for "Retrieving Reports" stage with fiscal year and quarter if available
                    enhanced_reasoning = reasoning
                    if mapped_stage.get('id') == 'searching':
                        requested_fiscal_year = update.get("requested_fiscal_year")
                        requested_quarter = update.get("requested_quarter")
                        current_ticker_symbol = update.get("ticker_symbol")
                        
                        # Replace any old ticker symbol in the reasoning with the current one
                        if current_ticker_symbol and enhanced_reasoning:
                            import re
                            # First, check if the reasoning already contains the current ticker
                            # If it does and it's correct, don't do any replacements to avoid duplication
                            current_ticker_pattern = rf'\b{re.escape(current_ticker_symbol)}\b'
                            if re.search(current_ticker_pattern, enhanced_reasoning, re.I):
                                # Current ticker is already in the reasoning, check if it's in the right format
                                # Look for patterns like "for TSLA", "TSLA FY", etc.
                                correct_patterns = [
                                    rf'\bfor\s+{re.escape(current_ticker_symbol)}\b',
                                    rf'\b{re.escape(current_ticker_symbol)}\s+FY',
                                    rf'\breport\s+{re.escape(current_ticker_symbol)}\b',
                                ]
                                has_correct_format = any(re.search(pattern, enhanced_reasoning, re.I) for pattern in correct_patterns)
                                
                                if has_correct_format:
                                    # Ticker is already correctly formatted, skip replacement to avoid duplication
                                    logger.debug(f"Current ticker '{current_ticker_symbol}' already correctly formatted in reasoning, skipping replacement")
                                else:
                                    # Ticker exists but might be in wrong format, only replace if there's a different ticker
                                    # Match patterns like "for OLD_TICKER" or "report OLD_TICKER" where OLD_TICKER != current
                                    ticker_pattern = r'\b(for|report)\s+([A-Z]{1,5})\b(?=\s|FY|Q|$)'
                                    matches = re.findall(ticker_pattern, enhanced_reasoning, re.I)
                                    for prefix, old_ticker in matches:
                                        # Only replace if it's a different ticker
                                        if old_ticker.upper() != current_ticker_symbol.upper():
                                            enhanced_reasoning = re.sub(
                                                rf'\b{re.escape(prefix)}\s+{re.escape(old_ticker)}\b',
                                                f'{prefix} {current_ticker_symbol}',
                                                enhanced_reasoning,
                                                count=1,
                                                flags=re.I
                                            )
                                            logger.info(f"Replaced old ticker '{old_ticker}' with current ticker '{current_ticker_symbol}' in reasoning")
                                            break  # Only replace first occurrence
                            else:
                                # Current ticker not found, look for any ticker to replace
                                ticker_pattern = r'\b(for|report)\s+([A-Z]{1,5})\b(?=\s|FY|Q|$)'
                                matches = re.findall(ticker_pattern, enhanced_reasoning, re.I)
                                for prefix, old_ticker in matches:
                                    # Replace with current ticker
                                    enhanced_reasoning = re.sub(
                                        rf'\b{re.escape(prefix)}\s+{re.escape(old_ticker)}\b',
                                        f'{prefix} {current_ticker_symbol}',
                                        enhanced_reasoning,
                                        count=1,
                                        flags=re.I
                                    )
                                    logger.info(f"Replaced ticker '{old_ticker}' with current ticker '{current_ticker_symbol}' in reasoning")
                                    break  # Only replace first occurrence
                        
                        fiscal_info_parts = []
                        if requested_fiscal_year:
                            fiscal_info_parts.append(f"FY{requested_fiscal_year}")
                        if requested_quarter:
                            fiscal_info_parts.append(f"Q{requested_quarter}")
                        
                        if fiscal_info_parts:
                            fiscal_info = " ".join(fiscal_info_parts)
                            if enhanced_reasoning and enhanced_reasoning.strip():
                                # Append fiscal info to existing reasoning if not already present
                                if fiscal_info not in enhanced_reasoning:
                                    enhanced_reasoning = f"{enhanced_reasoning}\n\nRetrieving reports for {fiscal_info}."
                            else:
                                # Create reasoning if it doesn't exist
                                if current_ticker_symbol:
                                    enhanced_reasoning = f"Retrieving reports for {current_ticker_symbol} {fiscal_info}."
                                else:
                                    enhanced_reasoning = f"Retrieving reports for {fiscal_info}."
                    
                    # Send reasoning event first if it exists (as a separate visible step)
                    if enhanced_reasoning and enhanced_reasoning.strip():
                        # Store reasoning for this stage
                        stage_reasoning[mapped_stage['id']] = enhanced_reasoning
                        
                        reasoning_data = {
                            'type': 'reasoning',
                            'session_id': session.session_id,
                            'stage_id': mapped_stage['id'],
                            'stage_label': mapped_stage['label'],
                            'node': node,
                            'reasoning': enhanced_reasoning
                        }
                        logger.info(f"Sending reasoning event for stage {mapped_stage['id']}: {enhanced_reasoning[:100]}...")
                        yield f"data: {json.dumps(reasoning_data)}\n\n"
                    
                    # Send stage update
                    stage_update_data = {
                        'type': 'stage_update',
                        'session_id': session.session_id,
                        'stage_id': mapped_stage['id'],
                        'stage_label': mapped_stage['label'],
                        'node': node,
                        'status': 'active',
                    }
                    
                    # Also include reasoning in stage_update for backwards compatibility
                    if enhanced_reasoning and enhanced_reasoning.strip():
                        stage_update_data['reasoning'] = enhanced_reasoning
                        # Store reasoning for this stage
                        stage_reasoning[mapped_stage['id']] = enhanced_reasoning
                    elif reasoning and reasoning.strip():
                        stage_update_data['reasoning'] = reasoning
                        # Store reasoning for this stage
                        stage_reasoning[mapped_stage['id']] = reasoning
                    
                    yield f"data: {json.dumps(stage_update_data)}\n\n"
                    
                    # Track final result (keep updating as we get more complete data)
                    if update.get("has_summary") or update.get("summary"):
                        # Merge updates to preserve all fields including ticker_symbol
                        if final_result:
                            final_result.update(update)
                        else:
                            final_result = update.copy()
                        # Don't mark as complete here - wait until stream is finished
                        # The stage will be marked complete when we actually move to a different stage
                        # or when the analysis is fully complete (handled after the stream loop)
                
                # Use final result from stream if available, otherwise run full analysis
                if final_result and final_result.get("summary"):
                    # Prioritize ticker_symbol from tool call (most reliable - what tool actually used)
                    ticker_symbol_for_result = ticker_from_tool_call or final_result.get("ticker_symbol")
                    result = {
                        "summary": final_result.get("summary"),
                        "messages": final_result.get("messages", []),
                        "error": None,
                        "ticker_symbol": ticker_symbol_for_result,
                        "company_name": final_result.get("company_name"),
                        "requested_fiscal_year": final_result.get("requested_fiscal_year"),
                        "requested_quarter": final_result.get("requested_quarter"),
                    }
                else:
                    # Fallback: run full analysis if stream didn't provide complete result
                    result = await run_earnings_analysis(company_query)
                    # If we have ticker from tool call, use it (more reliable)
                    if ticker_from_tool_call and not result.get("ticker_symbol"):
                        result["ticker_symbol"] = ticker_from_tool_call
                
                # Update database
                db_session.status = "complete" if result.get("summary") else "error"
                db_session.summary = result.get("summary")
                # Update ticker_symbol in database session if available
                if result.get("ticker_symbol"):
                    db_session.ticker_symbol = result.get("ticker_symbol")
                await db.commit()
                
                # Preserve previous ticker_symbol for scraping (before overwriting with new analysis)
                # Use previous ticker_symbol if it exists, as it's more reliable than extracting from new query
                previous_ticker_symbol = None
                previous_company_name = None
                if session.last_analysis:
                    previous_ticker_symbol = session.last_analysis.get('ticker_symbol')
                    previous_company_name = session.last_analysis.get('company_name')
                    if previous_ticker_symbol:
                        logger.info(f"Found previous ticker_symbol '{previous_ticker_symbol}' from last_analysis - will use for scraping if current result doesn't have one")
                
                # Get the summary
                summary = result.get("summary")
                
                # If no summary but we have messages, check the last assistant message
                if not summary:
                    messages = result.get("messages", [])
                    for msg in reversed(messages):
                        if isinstance(msg, dict):
                            role = msg.get("role", "")
                            content = msg.get("content", "")
                            # Accept assistant messages (clarification messages may be shorter)
                            if role == "assistant" and content and len(content) > 10:
                                summary = content
                                break
                        elif hasattr(msg, 'content') and hasattr(msg, 'role'):
                            if hasattr(msg, 'role') and (msg.role == "assistant" or isinstance(msg, AIMessage)):
                                if hasattr(msg, 'content') and msg.content and len(str(msg.content)) > 10:
                                    summary = str(msg.content)
                                    break
                
                # Fallback message if still no summary (only for actual analysis failures, not clarifications)
                if not summary or (len(summary.strip()) < 50 and "specify which quarter" not in summary.lower() and "need you to specify" not in summary.lower()):
                    summary = "Unable to generate a comprehensive summary. The transcript may not have been fully extracted, or there was an error processing it."
                
                # Add assistant message to history BEFORE storing in search history
                # This ensures the summary message is included in the stored conversation snapshot
                session.add_message("assistant", summary)
                
                # Store analysis in session (AFTER adding summary message to conversation_history)
                analysis_data = {
                    "company_query": company_query,
                    "summary": summary,
                    "messages": result.get("messages", []),
                    "timestamp": datetime.utcnow().isoformat(),
                    "ticker_symbol": result.get("ticker_symbol"),
                    "company_name": result.get("company_name"),
                    "requested_fiscal_year": result.get("requested_fiscal_year"),
                    "requested_quarter": result.get("requested_quarter"),
                    "stage_reasoning": stage_reasoning,  # Store reasoning for each stage
                }
                session.set_analysis(analysis_data)
                
                # Mark the last active stage as complete when stream finishes
                if last_mapped_stage and last_mapped_stage.get('id'):
                    complete_update_data = {
                        'type': 'stage_update',
                        'session_id': session.session_id,
                        'stage_id': last_mapped_stage.get('id'),
                        'stage_label': last_mapped_stage.get('label'),
                        'status': 'complete'
                    }
                    yield f"data: {json.dumps(complete_update_data)}\n\n"
                
                # Send final result with ticker symbol and company name
                complete_data = {
                    'type': 'complete',
                    'session_id': session.session_id,
                    'message': summary,
                    'action_taken': 'analysis_triggered',
                    'intent': classification.intent,
                    'ticker_symbol': result.get('ticker_symbol'),
                    'company_name': result.get('company_name'),
                    'is_complete': True
                }
                yield f"data: {json.dumps(complete_data)}\n\n"
                
                # Automatically trigger scraping if we have a ticker symbol
                # For NEW analyses (action == "analyze"), prioritize current result's ticker_symbol
                # Also try to extract ticker from transcript source URL (most reliable - comes from transcript tool)
                # Only use previous ticker_symbol if current result doesn't have one
                # This ensures new analyses for different companies use the correct ticker
                ticker_symbol_for_scraping = None
                company_name_for_storage = None
                
                # Extract ticker symbol from the actual tool call/invocation (most reliable)
                # The transcript tool was called with the correct ticker symbol (e.g., 'SQ' for Block)
                # This is more reliable than extracting from URLs or query text
                ticker_from_tool = ticker_from_tool_call  # Use ticker captured from stream updates first
                import re
                from langchain_core.messages import ToolMessage
                
                # Check messages in result and final_result for tool calls/invocations
                messages_to_check = []
                if result.get('messages'):
                    messages_to_check.extend(result.get('messages', []))
                if final_result and final_result.get('messages'):
                    messages_to_check.extend(final_result.get('messages', []))
                
                if messages_to_check:
                    logger.info(f"Checking {len(messages_to_check)} messages for tool calls/invocations...")
                    
                    # Method 1: Check AIMessage tool_calls for get_earnings_transcript invocations
                    for idx, msg in enumerate(messages_to_check):
                        # Check if it's an AIMessage with tool_calls
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            for tool_call in msg.tool_calls:
                                tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
                                if tool_name == 'get_earnings_transcript':
                                    tool_args = tool_call.get('args', {}) if isinstance(tool_call, dict) else getattr(tool_call, 'args', {})
                                    symbol = tool_args.get('symbol') if isinstance(tool_args, dict) else None
                                    if symbol:
                                        ticker_from_tool = symbol.upper()
                                        logger.info(f"Extracted ticker_symbol '{ticker_from_tool}' from get_earnings_transcript tool call (message {idx})")
                                        break
                            if ticker_from_tool:
                                break
                    
                    # Method 2: Extract from transcript source URL in ToolMessage content
                    if not ticker_from_tool:
                        for idx, msg in enumerate(messages_to_check):
                            # Check ToolMessage content for source URL
                            if isinstance(msg, ToolMessage) or (hasattr(msg, 'name') and getattr(msg, 'name', '') == 'get_earnings_transcript'):
                                content = None
                                if isinstance(msg, dict):
                                    content = msg.get('content', '')
                                elif hasattr(msg, 'content'):
                                    content = str(msg.content)
                                
                                if content and len(content) > 100:
                                    # Look for source URL in transcript content
                                    patterns = [
                                        r'_Source URL: https?://[^/\s]+/company/([A-Z]+)/',  # Standard format
                                        r'/company/([A-Z]+)/transcripts/',  # URL pattern anywhere in content
                                        r'company/([A-Z]+)/transcripts/\d{4}/\d',  # More specific pattern
                                    ]
                                    
                                    for pattern in patterns:
                                        url_match = re.search(pattern, content)
                                        if url_match:
                                            ticker_from_tool = url_match.group(1).upper()
                                            logger.info(f"Extracted ticker_symbol '{ticker_from_tool}' from transcript source URL using pattern '{pattern}' (message {idx})")
                                            break
                                    
                                    if ticker_from_tool:
                                        break
                
                # Prioritize ticker from tool call/invocation (most reliable - what the tool actually used)
                # ticker_from_tool may have been captured from stream updates (ticker_from_tool_call)
                # or extracted from messages below
                # Then try transcript URL extraction, then current result's ticker_symbol
                # This ensures that if user analyzes "Block" after "TSLA", we use "SQ" not "BLOCK" or "TSLA"
                if ticker_from_tool:
                    ticker_symbol_for_scraping = ticker_from_tool
                    company_name_for_storage = result.get('company_name') or ticker_from_tool
                    logger.info(f"Using ticker_symbol '{ticker_symbol_for_scraping}' from transcript tool call for scraping (most reliable - captured from stream or messages)")
                elif result.get('ticker_symbol'):
                    ticker_symbol_for_scraping = result.get('ticker_symbol')
                    company_name_for_storage = result.get('company_name') or result.get('ticker_symbol')
                    logger.info(f"Using ticker_symbol '{ticker_symbol_for_scraping}' from current analysis result for scraping")
                # Fallback to previous ticker_symbol only if current result doesn't have one
                # This handles cases where current analysis didn't extract a ticker but previous one did
                elif previous_ticker_symbol:
                    ticker_symbol_for_scraping = previous_ticker_symbol
                    company_name_for_storage = previous_company_name or previous_ticker_symbol
                    logger.info(f"Using ticker_symbol '{ticker_symbol_for_scraping}' from previous analysis for scraping (current result has no ticker_symbol)")
                
                if ticker_symbol_for_scraping:
                    logger.info(f"Ticker symbol found: {ticker_symbol_for_scraping}, sending metrics dashboard message")
                    try:
                        # Send loading message first, before starting the scrape
                        loading_message = {
                            'type': 'metrics_dashboard_loading',
                            'session_id': session.session_id,
                            'ticker_symbol': ticker_symbol_for_scraping,
                            'company_name': company_name_for_storage or ticker_symbol_for_scraping,
                            'message': 'Loading financial metric charts...',
                            'is_complete': False
                        }
                        logger.info(f"Sending metrics_dashboard_loading message for {ticker_symbol_for_scraping}")
                        yield f"data: {json.dumps(loading_message)}\n\n"
                        
                        # Trigger scraping in background (don't wait for it)
                        # ticker_symbol is what's used for scraping (URL construction)
                        # company_name is just for database storage/display
                        asyncio.create_task(
                            scrape_and_store_metrics(
                                ticker_symbol=ticker_symbol_for_scraping,
                                company_name=company_name_for_storage or ticker_symbol_for_scraping,
                                session_id=session.session_id
                            )
                        )
                        
                        # Send a separate message for the metrics dashboard (after loading message)
                        metrics_message = {
                            'type': 'metrics_dashboard',
                            'session_id': session.session_id,
                            'ticker_symbol': ticker_symbol_for_scraping,
                            'company_name': company_name_for_storage or ticker_symbol_for_scraping,
                            'is_complete': True
                        }
                        logger.info(f"Sending metrics_dashboard message for {ticker_symbol_for_scraping}")
                        yield f"data: {json.dumps(metrics_message)}\n\n"
                    except Exception as e:
                        logger.error(f"Error triggering metrics scrape: {e}", exc_info=True)
                else:
                    logger.warning(f"No ticker symbol found in previous analysis or current result. Previous analysis: {session.last_analysis.get('ticker_symbol') if session.last_analysis else 'None'}, Current result keys: {result.keys() if result else 'None'}")
            
            else:  # action == "chat"
                # Generate chat response
                yield f"data: {json.dumps({'type': 'status', 'session_id': session.session_id, 'stage': 'thinking', 'message': 'Generating response...'})}\n\n"
                
                # Add search entry for first chat message (if it's a new conversation)
                if not session.search_history and not session.last_analysis:
                    session.add_search_entry(
                        query=request.message,
                        action="chat"
                    )
                
                response_message = await create_chat_response(
                    user_input=request.message,
                    context=context,
                    intent=classification.intent
                )
                
                # Add assistant message to history
                session.add_message("assistant", response_message)
                
                # Send final result
                complete_data = {
                    'type': 'complete',
                    'session_id': session.session_id,
                    'message': response_message,
                    'action_taken': 'chat',
                    'intent': classification.intent,
                    'is_complete': True
                }
                yield f"data: {json.dumps(complete_data)}\n\n"
        
        except Exception as e:
            error_message = f"Sorry, I encountered an error: {str(e)}"
            session_id = None
            if 'session' in locals():
                session.add_message("assistant", error_message)
                session_id = session.session_id
            
            error_data = {
                'type': 'error',
                'session_id': session_id,
                'error': str(e),
                'message': error_message,
                'is_complete': True
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Conversational endpoint that intelligently routes between chat and analysis.
    
    This endpoint:
    1. Maintains conversation context within a session
    2. Classifies user intent (new analysis vs follow-up question)
    3. Routes to appropriate handler (analysis agent vs chat)
    4. Returns contextual responses
    
    If the intent is to analyze a new company, it triggers the full analysis pipeline.
    If it's a follow-up question, it answers based on existing analysis in the session.
    """
    
    # Get or create session
    session_manager = get_session_manager()
    session = session_manager.get_or_create_session(request.session_id)
    
    # Add user message to history
    session.add_message("user", request.message)
    
    try:
        # Route the conversation
        routing_result = await conversation_router.route_conversation(
            user_input=request.message,
            session_data={
                "conversation_history": session.conversation_history,
                "last_analysis": session.last_analysis,
                "session_id": session.session_id,
            }
        )
        
        action = routing_result["action"]
        classification = routing_result["classification"]
        context = routing_result["context"]
        
        if action == "analyze":
            # Trigger new earnings analysis
            company_query = context.get("company_query", request.message)
            
            # Check if database session already exists
            result = await db.execute(
                select(SearchSession).where(SearchSession.session_id == session.session_id)
            )
            db_session = result.scalar_one_or_none()
            
            if db_session:
                # Update existing session
                db_session.company_query = company_query
                db_session.status = "searching"
                db_session.summary = None  # Clear previous summary
                db_session.updated_at = datetime.utcnow()
            else:
                # Create new database session record
                db_session = SearchSession(
                    session_id=session.session_id,
                    company_query=company_query,
                    status="searching",
                )
                db.add(db_session)
            
            await db.commit()
            
            # Run analysis
            result = await run_earnings_analysis(company_query)
            
            # Update database
            db_session.status = "complete" if result.get("summary") else "error"
            db_session.summary = result.get("summary")
            await db.commit()
            
            # Preserve previous ticker_symbol for scraping (before overwriting with new analysis)
            # Use previous ticker_symbol if it exists, as it's more reliable than extracting from new query
            previous_ticker_symbol = None
            previous_company_name = None
            if session.last_analysis:
                previous_ticker_symbol = session.last_analysis.get('ticker_symbol')
                previous_company_name = session.last_analysis.get('company_name')
                if previous_ticker_symbol:
                    logger.info(f"Found previous ticker_symbol '{previous_ticker_symbol}' from last_analysis - will use for scraping if current result doesn't have one")
            
            # Store analysis in session
            analysis_data = {
                "company_query": company_query,
                "summary": result.get("summary"),
                "messages": result.get("messages", []),
                "timestamp": datetime.utcnow().isoformat(),
                "ticker_symbol": result.get("ticker_symbol"),
                "company_name": result.get("company_name"),
                "requested_fiscal_year": result.get("requested_fiscal_year"),
                "requested_quarter": result.get("requested_quarter"),
            }
            session.set_analysis(analysis_data)
            
            # Get the summary - prioritize the actual summary from the agent
            summary = result.get("summary")
            
            # If no summary but we have messages, check the last assistant message
            if not summary:
                messages = result.get("messages", [])
                # Look for the last assistant/system message that might be a summary
                for msg in reversed(messages):
                    if isinstance(msg, dict):
                        role = msg.get("role", "")
                        content = msg.get("content", "")
                        # Accept assistant messages (clarification messages may be shorter)
                        if role == "assistant" and content and len(content) > 10:
                            summary = content
                            break
                    elif hasattr(msg, 'content') and hasattr(msg, 'role'):
                        if hasattr(msg, 'role') and (msg.role == "assistant" or isinstance(msg, AIMessage)):
                            if hasattr(msg, 'content') and msg.content and len(str(msg.content)) > 10:
                                summary = str(msg.content)
                                break
            
            # Fallback message if still no summary (only for actual analysis failures, not clarifications)
            if not summary or (len(summary.strip()) < 50 and "specify which quarter" not in summary.lower() and "need you to specify" not in summary.lower()):
                summary = "Unable to generate a comprehensive summary. The transcript may not have been fully extracted, or there was an error processing it."
            
            # Add assistant message to history
            session.add_message("assistant", summary)
            
            return ChatResponse(
                session_id=session.session_id,
                message=summary,
                action_taken="analysis_triggered",
                intent=classification.intent,
                analysis_result=EarningsResponse(
                    session_id=session.session_id,
                    company_query=company_query,
                    status=AnalysisStatus.COMPLETE if summary and len(summary.strip()) > 50 else AnalysisStatus.ERROR,
                    summary=summary,
                    messages=[
                        Message(role=m.get("role") if isinstance(m, dict) else ("assistant" if isinstance(m, AIMessage) else "user"), 
                               content=m.get("content") if isinstance(m, dict) else (m.content if hasattr(m, 'content') else str(m)))
                        for m in result.get("messages", [])
                        if (isinstance(m, dict) and m.get("content")) or (hasattr(m, 'content') and m.content)
                    ],
                    error=result.get("error"),
                )
            )
        
        else:  # action == "chat"
            # Add search entry for first chat message (if it's a new conversation)
            if not session.search_history and not session.last_analysis:
                session.add_search_entry(
                    query=request.message,
                    action="chat"
                )
            
            # Generate chat response
            response_message = await create_chat_response(
                user_input=request.message,
                context=context,
                intent=classification.intent
            )
            
            # Add assistant message to history
            session.add_message("assistant", response_message)
            
            return ChatResponse(
                session_id=session.session_id,
                message=response_message,
                action_taken="chat",
                intent=classification.intent,
                analysis_result=None
            )
    
    except Exception as e:
        error_message = f"Sorry, I encountered an error: {str(e)}"
        session.add_message("assistant", error_message)
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/metrics/{ticker_symbol}/scrape")
async def scrape_financial_metrics(
    ticker_symbol: str,
    company_name: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Scrape financial metrics from discountingcashflows.com for a ticker symbol.
    
    This endpoint:
    1. Scrapes income statement, balance sheet, and cash flow statements
    2. Stores the metrics in the database
    3. Returns the scraped data
    """
    try:
        # Scrape financial data
        scraped_data = await scrape_company_financials(ticker_symbol.upper())
        
        # Store in database
        if scraped_data and any(scraped_data.values()):
            stored_metrics = await store_scraped_metrics(
                db=db,
                ticker_symbol=ticker_symbol.upper(),
                company_name=company_name or ticker_symbol.upper(),
                scraped_data=scraped_data
            )
            
            return {
                "ticker_symbol": ticker_symbol.upper(),
                "company_name": company_name or ticker_symbol.upper(),
                "periods_scraped": len(stored_metrics),
                "metrics": [
                    {
                        "fiscal_year": m.fiscal_year,
                        "fiscal_quarter": m.fiscal_quarter,
                        "period": f"FY{m.fiscal_year} Q{m.fiscal_quarter}",
                        "revenue": m.revenue,
                        "net_income": m.net_income,
                        "eps": m.eps,
                    }
                    for m in stored_metrics
                ]
            }
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"No financial data found for ticker {ticker_symbol}"
            )
            
    except Exception as e:
        logger.error(f"Error scraping metrics for {ticker_symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/{ticker_symbol}")
async def get_company_metrics(
    ticker_symbol: str,
    fiscal_year: Optional[str] = None,
    fiscal_quarter: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """Get financial metrics for a company."""
    metrics = await get_metrics(db, ticker_symbol.upper(), fiscal_year, fiscal_quarter)
    
    if not metrics:
        raise HTTPException(status_code=404, detail="Metrics not found")
    
    # Parse segment_data JSON if present
    segment_data = None
    if metrics.segment_data:
        try:
            segment_data = json.loads(metrics.segment_data)
        except:
            pass
    
    return {
        "ticker_symbol": metrics.ticker_symbol,
        "company_name": metrics.company_name,
        "fiscal_year": metrics.fiscal_year,
        "fiscal_quarter": metrics.fiscal_quarter,
        "period": f"FY{metrics.fiscal_year} Q{metrics.fiscal_quarter}",
        "revenue": metrics.revenue,
        "revenue_qoq_change": metrics.revenue_qoq_change,
        "revenue_yoy_change": metrics.revenue_yoy_change,
        "eps": metrics.eps,
        "eps_actual": metrics.eps_actual,
        "eps_estimate": metrics.eps_estimate,
        "eps_beat_miss": metrics.eps_beat_miss,
        "net_income": metrics.net_income,
        "gross_margin": metrics.gross_margin,
        "operating_margin": metrics.operating_margin,
        "net_margin": metrics.net_margin,
        "free_cash_flow": metrics.free_cash_flow,
        "operating_cash_flow": metrics.operating_cash_flow,
        "total_assets": metrics.total_assets,
        "total_liabilities": metrics.total_liabilities,
        "total_equity": metrics.total_equity,
        "current_assets": metrics.current_assets,
        "current_liabilities": metrics.current_liabilities,
        "revenue_guidance": metrics.revenue_guidance,
        "eps_guidance": metrics.eps_guidance,
        "segment_data": segment_data,
        "report_date": metrics.report_date.isoformat() if metrics.report_date else None,
    }


@router.get("/metrics/{ticker_symbol}/history")
async def get_metrics_history_endpoint(
    ticker_symbol: str,
    limit: int = 8,
    db: AsyncSession = Depends(get_db),
):
    """Get historical financial metrics for a company."""
    try:
        history = await get_metrics_history(db, ticker_symbol.upper(), limit)
        logger.info(f"Fetched {len(history)} metrics records for {ticker_symbol}")
        
        # Log sample data for debugging
        if history:
            sample = history[0]
            logger.debug(f"Sample metric for {ticker_symbol}: revenue={sample.revenue}, eps={sample.eps}, "
                        f"gross_margin={sample.gross_margin}, operating_margin={sample.operating_margin}, "
                        f"free_cash_flow={sample.free_cash_flow}, operating_cash_flow={sample.operating_cash_flow}")
    except Exception as e:
        logger.error(f"Error fetching metrics history for {ticker_symbol}: {e}", exc_info=True)
        # Return empty history instead of crashing
        return {"ticker_symbol": ticker_symbol.upper(), "history": []}
    
    results = []
    for metrics in history:
        segment_data = None
        if metrics.segment_data:
            try:
                segment_data = json.loads(metrics.segment_data)
            except:
                pass
        
        results.append({
            "fiscal_year": metrics.fiscal_year,
            "fiscal_quarter": metrics.fiscal_quarter,
            "period": f"FY{metrics.fiscal_year} Q{metrics.fiscal_quarter}",
            "revenue": metrics.revenue,
            "revenue_qoq_change": metrics.revenue_qoq_change,
            "revenue_yoy_change": metrics.revenue_yoy_change,
            "eps": metrics.eps,
            "eps_actual": metrics.eps_actual,
            "eps_estimate": metrics.eps_estimate,
            "net_income": metrics.net_income,
            "gross_margin": metrics.gross_margin,
            "operating_margin": metrics.operating_margin,
            "net_margin": metrics.net_margin,
            "free_cash_flow": metrics.free_cash_flow,
            "operating_cash_flow": metrics.operating_cash_flow,
            "total_assets": metrics.total_assets,
            "total_liabilities": metrics.total_liabilities,
            "total_equity": metrics.total_equity,
            "report_date": metrics.report_date.isoformat() if metrics.report_date else None,
        })
    
    return {"ticker_symbol": ticker_symbol.upper(), "history": results}


@router.get("/sessions/{session_id}/history", response_model=SearchHistoryResponse)
async def get_session_search_history(
    session_id: str,
    all_sessions: bool = False,  # Query parameter to get history from all sessions
):
    """
    Get search history for a session.
    
    If all_sessions=true, returns history from ALL sessions (not just the specified one).
    This allows users to see all their previous searches across different conversations.
    """
    session_manager = get_session_manager()
    
    if all_sessions:
        # Get history from all sessions
        history = session_manager.get_all_search_history()
        logger.info(f"Fetching search history from all sessions: {len(history)} total entries")
    else:
        # Get history from specific session
        session = session_manager.get_session(session_id)
        
        if not session:
            # Return empty history if session doesn't exist
            return SearchHistoryResponse(
                session_id=session_id,
                searches=[],
                total=0
            )
        
        history = session.get_search_history()
        logger.info(f"Fetching search history for session {session_id}: {len(history)} entries")
    
    # Convert to SearchHistoryEntry objects (include messages and reasoning for restoration)
    search_entries = [
        SearchHistoryEntry(
            id=entry["id"],
            timestamp=entry["timestamp"],
            ticker_symbol=entry.get("ticker_symbol"),
            company_name=entry.get("company_name"),
            fiscal_year=entry.get("fiscal_year"),
            fiscal_quarter=entry.get("fiscal_quarter"),
            query=entry.get("query", ""),
            action=entry.get("action", "chat"),
            message_count=entry.get("message_count"),
            messages=entry.get("messages"),  # Include messages for restoration
            stage_reasoning=entry.get("stage_reasoning")  # Include reasoning for restoration
        )
        for entry in history
    ]
    
    return SearchHistoryResponse(
        session_id=session_id if not all_sessions else "all",
        searches=search_entries,
        total=len(search_entries)
    )


@router.get("/sessions/{session_id}/history/{search_id}/messages")
async def get_search_entry_messages(
    session_id: str,
    search_id: str,
):
    """Get messages for a specific search entry to restore conversation state."""
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session.get_search_entry_messages(search_id)
    
    if messages is None:
        raise HTTPException(status_code=404, detail="Search entry not found")
    
    return {
        "session_id": session_id,
        "search_id": search_id,
        "messages": messages
    }


@router.post("/sessions/{session_id}/save-conversation")
async def save_current_conversation(
    session_id: str,
):
    """
    Save the current conversation to search history.
    This is useful when the user wants to save their conversation before navigating away.
    """
    session_manager = get_session_manager()
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Only save if there are messages in the conversation
    if not session.conversation_history:
        return {
            "session_id": session_id,
            "saved": False,
            "message": "No conversation to save"
        }
    
    # Check if this conversation has already been saved (to avoid duplicates)
    # We'll check if the last search entry has the same message count
    existing_history = session.get_search_history()
    if existing_history:
        last_entry = existing_history[0]  # Most recent entry
        if last_entry.get("message_count") == len(session.conversation_history):
            # Same number of messages - likely already saved
            return {
                "session_id": session_id,
                "saved": False,
                "message": "Conversation already saved"
            }
    
    # Extract query from first user message
    first_user_message = None
    for msg in session.conversation_history:
        if msg.get("role") == "user":
            first_user_message = msg.get("content", "")
            break
    
    # Save as a chat entry
    session.add_search_entry(
        query=first_user_message or "Conversation",
        action="chat"
    )
    
    return {
        "session_id": session_id,
        "saved": True,
        "message": "Conversation saved successfully"
    }
