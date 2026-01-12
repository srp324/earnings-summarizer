from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, Float, Index
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


class FinancialMetrics(Base):
    """Store extracted financial metrics from earnings reports."""
    __tablename__ = "financial_metrics"

    id = Column(Integer, primary_key=True, index=True)
    ticker_symbol = Column(String(10), nullable=False, index=True)
    company_name = Column(String(255), nullable=False)
    fiscal_year = Column(String(4), nullable=False, index=True)
    fiscal_quarter = Column(String(1), nullable=False, index=True)  # 1, 2, 3, 4
    report_date = Column(DateTime)  # When the earnings report was released
    
    # Revenue metrics
    revenue = Column(Float)  # Total revenue
    revenue_qoq_change = Column(Float)  # Quarter-over-quarter change (%)
    revenue_yoy_change = Column(Float)  # Year-over-year change (%)
    revenue_growth = Column(Float)  # Growth rate
    
    # Earnings metrics
    eps = Column(Float)  # Earnings per share
    eps_actual = Column(Float)  # Actual EPS
    eps_estimate = Column(Float)  # Estimated EPS (if mentioned)
    eps_beat_miss = Column(Float)  # Beat/miss amount
    
    # Profitability metrics
    net_income = Column(Float)
    gross_margin = Column(Float)  # Gross margin %
    operating_margin = Column(Float)  # Operating margin %
    net_margin = Column(Float)  # Net margin %
    
    # Cash flow
    free_cash_flow = Column(Float)
    operating_cash_flow = Column(Float)
    
    # Balance sheet metrics
    total_assets = Column(Float)
    total_liabilities = Column(Float)
    total_equity = Column(Float)
    current_assets = Column(Float)
    current_liabilities = Column(Float)
    
    # Guidance (forward-looking)
    revenue_guidance = Column(Float)
    eps_guidance = Column(Float)
    guidance_range_low = Column(Float)
    guidance_range_high = Column(Float)
    
    # Segment data (stored as JSON)
    segment_data = Column(Text)  # JSON string of segment breakdown
    
    # Raw scraped data (stored as JSON for flexibility)
    raw_income_statement = Column(Text)  # JSON string
    raw_balance_sheet = Column(Text)  # JSON string
    raw_cash_flow = Column(Text)  # JSON string
    
    # Metadata
    source_url = Column(Text)
    session_id = Column(String(100), index=True)  # Link to search session
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Unique constraint to prevent duplicates
    __table_args__ = (
        Index('idx_metrics_ticker_fy_fq', 'ticker_symbol', 'fiscal_year', 'fiscal_quarter', unique=True),
    )


async def init_db():
    """Initialize database tables."""
    async with engine.begin() as conn:
        # Create pgvector extension and ensure schema is up to date
        from sqlalchemy import text
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        # Ensure financial_metrics has all expected columns even on existing databases
        # (CREATE TABLE via metadata.create_all() will not add new columns)
        await conn.execute(text("""
            ALTER TABLE IF EXISTS financial_metrics
            ADD COLUMN IF NOT EXISTS report_date timestamp,
            ADD COLUMN IF NOT EXISTS revenue double precision,
            ADD COLUMN IF NOT EXISTS revenue_qoq_change double precision,
            ADD COLUMN IF NOT EXISTS revenue_yoy_change double precision,
            ADD COLUMN IF NOT EXISTS revenue_growth double precision,
            ADD COLUMN IF NOT EXISTS eps double precision,
            ADD COLUMN IF NOT EXISTS eps_actual double precision,
            ADD COLUMN IF NOT EXISTS eps_estimate double precision,
            ADD COLUMN IF NOT EXISTS eps_beat_miss double precision,
            ADD COLUMN IF NOT EXISTS net_income double precision,
            ADD COLUMN IF NOT EXISTS gross_margin double precision,
            ADD COLUMN IF NOT EXISTS operating_margin double precision,
            ADD COLUMN IF NOT EXISTS net_margin double precision,
            ADD COLUMN IF NOT EXISTS free_cash_flow double precision,
            ADD COLUMN IF NOT EXISTS operating_cash_flow double precision,
            ADD COLUMN IF NOT EXISTS total_assets double precision,
            ADD COLUMN IF NOT EXISTS total_liabilities double precision,
            ADD COLUMN IF NOT EXISTS total_equity double precision,
            ADD COLUMN IF NOT EXISTS current_assets double precision,
            ADD COLUMN IF NOT EXISTS current_liabilities double precision,
            ADD COLUMN IF NOT EXISTS revenue_guidance double precision,
            ADD COLUMN IF NOT EXISTS eps_guidance double precision,
            ADD COLUMN IF NOT EXISTS guidance_range_low double precision,
            ADD COLUMN IF NOT EXISTS guidance_range_high double precision,
            ADD COLUMN IF NOT EXISTS segment_data text,
            ADD COLUMN IF NOT EXISTS raw_income_statement text,
            ADD COLUMN IF NOT EXISTS raw_balance_sheet text,
            ADD COLUMN IF NOT EXISTS raw_cash_flow text,
            ADD COLUMN IF NOT EXISTS source_url text,
            ADD COLUMN IF NOT EXISTS session_id varchar(100),
            ADD COLUMN IF NOT EXISTS created_at timestamp,
            ADD COLUMN IF NOT EXISTS updated_at timestamp
        """))

        await conn.run_sync(Base.metadata.create_all)


async def get_db() -> AsyncSession:
    """Get database session."""
    async with AsyncSessionLocal() as session:
        yield session

