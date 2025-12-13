"""
Session Manager - Manages conversation state and history.

This module provides in-memory session management for conversations.
Could be extended to use Redis or database for persistence.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
from collections import defaultdict
import asyncio


class SessionData:
    """Data structure for a conversation session."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.last_accessed = datetime.utcnow()
        self.conversation_history: List[Dict[str, Any]] = []
        self.last_analysis: Optional[Dict[str, Any]] = None
        self.metadata: Dict[str, Any] = {}
    
    def add_message(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.last_accessed = datetime.utcnow()
    
    def set_analysis(self, analysis: Dict[str, Any]):
        """Store the latest analysis result."""
        self.last_analysis = analysis
        self.last_accessed = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "conversation_history": self.conversation_history,
            "last_analysis": self.last_analysis,
            "metadata": self.metadata
        }


class SessionManager:
    """Manages conversation sessions."""
    
    def __init__(self, session_timeout_minutes: int = 60):
        self._sessions: Dict[str, SessionData] = {}
        self._session_timeout = timedelta(minutes=session_timeout_minutes)
        self._cleanup_task = None
    
    async def start_cleanup_task(self):
        """Start background task to clean up expired sessions."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def _cleanup_loop(self):
        """Background loop to remove expired sessions."""
        while True:
            await asyncio.sleep(300)  # Run every 5 minutes
            await self.cleanup_expired_sessions()
    
    async def cleanup_expired_sessions(self):
        """Remove sessions that haven't been accessed recently."""
        now = datetime.utcnow()
        expired_sessions = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_accessed > self._session_timeout
        ]
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
        
        if expired_sessions:
            print(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def create_session(self) -> str:
        """Create a new session and return its ID."""
        session_id = str(uuid.uuid4())
        self._sessions[session_id] = SessionData(session_id)
        return session_id
    
    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session by ID."""
        session = self._sessions.get(session_id)
        if session:
            session.last_accessed = datetime.utcnow()
        return session
    
    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionData:
        """Get existing session or create a new one."""
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]
            session.last_accessed = datetime.utcnow()
            return session
        
        # Create new session
        new_session_id = str(uuid.uuid4())
        session = SessionData(new_session_id)
        self._sessions[new_session_id] = session
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions."""
        return [session.to_dict() for session in self._sessions.values()]
    
    def get_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._sessions)


# Global session manager instance
_session_manager: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


async def initialize_session_manager():
    """Initialize and start the session manager."""
    manager = get_session_manager()
    await manager.start_cleanup_task()




