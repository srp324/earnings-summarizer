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
        self.search_history: List[Dict[str, Any]] = []  # Track previous searches
    
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
        
        # Add to search history when analysis is completed
        # Store a snapshot of the conversation at this point
        search_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "ticker_symbol": analysis.get("ticker_symbol"),
            "company_name": analysis.get("company_name"),
            "fiscal_year": analysis.get("requested_fiscal_year"),
            "fiscal_quarter": analysis.get("requested_quarter"),
            "query": analysis.get("company_query", ""),
            "action": "analysis",
            "message_count": len(self.conversation_history),  # Store how many messages at this point
            "messages": self.conversation_history.copy()  # Store snapshot of messages
        }
        self.search_history.append(search_entry)
        # Keep only last 50 searches to avoid memory issues
        if len(self.search_history) > 50:
            self.search_history = self.search_history[-50:]
    
    def add_search_entry(self, query: str, action: str = "chat", ticker_symbol: Optional[str] = None, 
                        fiscal_year: Optional[str] = None, fiscal_quarter: Optional[str] = None):
        """Add a search entry for general chat messages (first message in conversation)."""
        # Only add if it's the first message or if it's a chat without analysis
        if action == "chat" and not self.search_history:
            search_entry = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.utcnow().isoformat(),
                "ticker_symbol": ticker_symbol,
                "company_name": None,
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter,
                "query": query[:100] if len(query) > 100 else query,  # Truncate long queries
                "action": action,
                "message_count": len(self.conversation_history),  # Store how many messages at this point
                "messages": self.conversation_history.copy()  # Store snapshot of messages
            }
            self.search_history.append(search_entry)
    
    def get_search_entry_messages(self, search_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get the messages for a specific search entry."""
        for entry in self.search_history:
            if entry.get("id") == search_id:
                return entry.get("messages", [])
        return None
    
    def get_search_history(self) -> List[Dict[str, Any]]:
        """Get search history sorted by most recent first."""
        return sorted(self.search_history, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "conversation_history": self.conversation_history,
            "last_analysis": self.last_analysis,
            "metadata": self.metadata,
            "search_history": self.get_search_history()
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




