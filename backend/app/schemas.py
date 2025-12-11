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

