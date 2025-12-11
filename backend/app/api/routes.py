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
)
from app.database import get_db, SearchSession
from app.agents.earnings_agent import run_earnings_analysis, stream_earnings_analysis

router = APIRouter()


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

