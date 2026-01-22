"""Pydantic schemas for API request/response models."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class AnalysisStatus(str, Enum):
    """Status of an earnings analysis."""
    PENDING = "pending"
    SEARCHING = "searching"
    PARSING = "parsing"
    SUMMARIZING = "summarizing"
    COMPLETE = "complete"
    ERROR = "error"


class EarningsRequest(BaseModel):
    """Request to analyze earnings for a company."""
    company_query: str = Field(
        ...,
        description="Company name or ticker symbol",
        example="Apple",
        min_length=1,
        max_length=255
    )


class Message(BaseModel):
    """A message in the analysis conversation."""
    role: str = Field(..., description="Role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = None


class EarningsResponse(BaseModel):
    """Response containing earnings analysis results."""
    session_id: str = Field(..., description="Unique session identifier")
    company_query: str = Field(..., description="Original company query")
    status: AnalysisStatus = Field(..., description="Current analysis status")
    summary: Optional[str] = Field(None, description="Generated earnings summary")
    messages: List[Message] = Field(default_factory=list, description="Conversation messages")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class StreamUpdate(BaseModel):
    """Real-time update during analysis."""
    session_id: str
    stage: str
    message: str
    progress: Optional[int] = None  # 0-100
    is_complete: bool = False
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionListResponse(BaseModel):
    """List of analysis sessions."""
    sessions: List[Dict[str, Any]]
    total: int


class ChatRequest(BaseModel):
    """Request for conversational chat."""
    message: str = Field(
        ...,
        description="User's message",
        example="Tell me more about the business segments",
        min_length=1,
        max_length=1000
    )
    session_id: Optional[str] = Field(
        None,
        description="Session ID for conversation continuity"
    )


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    session_id: str = Field(..., description="Session identifier")
    message: str = Field(..., description="AI response")
    action_taken: str = Field(
        ...,
        description="Action: 'chat' or 'analysis_triggered'"
    )
    intent: str = Field(..., description="Classified intent")
    analysis_result: Optional[EarningsResponse] = Field(
        None,
        description="Analysis result if new analysis was triggered"
    )


class SearchHistoryEntry(BaseModel):
    """A single entry in the search history."""
    id: str = Field(..., description="Unique entry identifier")
    timestamp: str = Field(..., description="ISO timestamp when search was performed")
    ticker_symbol: Optional[str] = Field(None, description="Company ticker symbol")
    company_name: Optional[str] = Field(None, description="Company name")
    fiscal_year: Optional[str] = Field(None, description="Fiscal year (e.g., '2024')")
    fiscal_quarter: Optional[str] = Field(None, description="Fiscal quarter (e.g., '1', '2', '3', '4')")
    query: str = Field(..., description="Original search query")
    action: str = Field(..., description="Action type: 'analysis' or 'chat'")
    message_count: Optional[int] = Field(None, description="Number of messages at this point")
    messages: Optional[List[Dict[str, Any]]] = Field(None, description="Snapshot of messages at this point")
    stage_reasoning: Optional[Dict[str, str]] = Field(None, description="Reasoning for each stage: {stage_id: reasoning}")


class SearchHistoryResponse(BaseModel):
    """Response containing session search history."""
    session_id: str = Field(..., description="Session identifier")
    searches: List[SearchHistoryEntry] = Field(..., description="List of previous searches")
    total: int = Field(..., description="Total number of searches")
