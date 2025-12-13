"""API routes for earnings summarization."""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
import json
import asyncio
from datetime import datetime
from typing import AsyncGenerator

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
)
from app.database import get_db, SearchSession
from app.agents.earnings_agent import run_earnings_analysis, stream_earnings_analysis
from app.agents.conversation_router import ConversationRouter, create_chat_response
from app.session_manager import get_session_manager
from langchain_core.messages import AIMessage

router = APIRouter()

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
            
            # Stage messages
            stage_messages = {
                "analyzing_query": "Analyzing your query and identifying the company...",
                "finding_ir_site": "Searching for the investor relations website...",
                "parsing_documents": "Downloading and parsing earnings reports...",
                "summarizing": "Generating comprehensive summary...",
                "complete": "Analysis complete!",
            }
            
            # Run analysis with streaming
            try:
                result = await run_earnings_analysis(request.company_query)
                
                # Send stage updates
                for stage, message in stage_messages.items():
                    yield f"data: {json.dumps({'session_id': session_id, 'stage': stage, 'message': message})}\n\n"
                    await asyncio.sleep(0.1)  # Small delay for UI updates
                
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
            session_manager = get_session_manager()
            session = session_manager.get_or_create_session(request.session_id)
            
            # Add user message to history
            session.add_message("user", request.message)
            
            # Send initial event
            yield f"data: {json.dumps({'type': 'status', 'stage': 'routing', 'message': 'Analyzing your request...'})}\n\n"
            
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
                
                # Create database session record
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
                    "retrieving_transcript": {"id": "searching", "label": "Retrieving Reports"},
                    "storing_embeddings": {"id": "parsing", "label": "Storing Embeddings"},
                    "complete": {"id": "summarizing", "label": "Generating Summary"},
                }
                
                # Stream analysis with real-time updates
                final_result = None
                last_stage = None
                async for update in stream_earnings_analysis(company_query):
                    stage = update.get("stage", "processing")
                    node = update.get("node", "")
                    last_stage = stage
                    
                    # Map stage to frontend stage
                    mapped_stage = stage_mapping.get(stage, {"id": stage, "label": stage.replace("_", " ").title()})
                    
                    # Send stage update
                    stage_update_data = {
                        'type': 'stage_update',
                        'stage_id': mapped_stage['id'],
                        'stage_label': mapped_stage['label'],
                        'node': node,
                        'status': 'active'
                    }
                    yield f"data: {json.dumps(stage_update_data)}\n\n"
                    
                    # Track final result (keep updating as we get more complete data)
                    if update.get("has_summary") or update.get("summary"):
                        final_result = update
                        # Mark summarizing stage as complete
                        complete_update_data = {
                            'type': 'stage_update',
                            'stage_id': 'summarizing',
                            'stage_label': 'Generating Summary',
                            'status': 'complete'
                        }
                        yield f"data: {json.dumps(complete_update_data)}\n\n"
                
                # Use final result from stream if available, otherwise run full analysis
                if final_result and final_result.get("summary"):
                    result = {
                        "summary": final_result.get("summary"),
                        "messages": final_result.get("messages", []),
                        "error": None,
                    }
                else:
                    # Fallback: run full analysis if stream didn't provide complete result
                    result = await run_earnings_analysis(company_query)
                
                # Update database
                db_session.status = "complete" if result.get("summary") else "error"
                db_session.summary = result.get("summary")
                await db.commit()
                
                # Store analysis in session
                analysis_data = {
                    "company_query": company_query,
                    "summary": result.get("summary"),
                    "messages": result.get("messages", []),
                    "timestamp": datetime.utcnow().isoformat(),
                }
                session.set_analysis(analysis_data)
                
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
                
                # Add assistant message to history
                session.add_message("assistant", summary)
                
                # Send final result
                complete_data = {
                    'type': 'complete',
                    'session_id': session.session_id,
                    'message': summary,
                    'action_taken': 'analysis_triggered',
                    'intent': classification.intent,
                    'is_complete': True
                }
                yield f"data: {json.dumps(complete_data)}\n\n"
            
            else:  # action == "chat"
                # Generate chat response
                yield f"data: {json.dumps({'type': 'status', 'stage': 'thinking', 'message': 'Generating response...'})}\n\n"
                
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
            if 'session' in locals():
                session.add_message("assistant", error_message)
            
            error_data = {
                'type': 'error',
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
            
            # Create database session record
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
            
            # Store analysis in session
            analysis_data = {
                "company_query": company_query,
                "summary": result.get("summary"),
                "messages": result.get("messages", []),
                "timestamp": datetime.utcnow().isoformat(),
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

