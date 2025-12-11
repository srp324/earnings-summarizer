from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.dialects.postgresql import ARRAY
from pgvector.sqlalchemy import Vector
from datetime import datetime

from app.config import get_settings

settings = get_settings()

# Create async engine
engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_pre_ping=True,
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()


class EarningsDocument(Base):
    """Store earnings documents and their embeddings for RAG."""
    __tablename__ = "earnings_documents"

    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), nullable=False, index=True)
    ticker_symbol = Column(String(10), nullable=False, index=True)
    document_type = Column(String(50))  # 10-K, 10-Q, earnings call, etc.
    fiscal_period = Column(String(50))  # Q1 2024, FY 2023, etc.
    source_url = Column(Text)
    content = Column(Text)
    chunk_index = Column(Integer, default=0)
    embedding = Column(Vector(1536))  # OpenAI ada-002 embedding size
    created_at = Column(DateTime, default=datetime.utcnow)


class SearchSession(Base):
    """Track user search sessions and results."""
    __tablename__ = "search_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(100), unique=True, index=True)
    company_query = Column(String(255))
    ticker_symbol = Column(String(10))
    investor_relations_url = Column(Text)
    documents_found = Column(Integer, default=0)
    summary = Column(Text)
    status = Column(String(50), default="pending")  # pending, searching, parsing, summarizing, complete, error
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        # Create pgvector extension
        from sqlalchemy import text
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        yield session

