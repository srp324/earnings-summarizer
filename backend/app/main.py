"""
Earnings Report Summarizer - FastAPI Backend

A multi-agent application using LangChain and LangGraph to analyze
and summarize stock earnings reports.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import get_settings
from app.database import init_db
from app.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Earnings Summarizer API...")
    
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Earnings Summarizer API...")


# Create FastAPI application
app = FastAPI(
    title="Earnings Report Summarizer",
    description="""
    A multi-agent AI system that analyzes and summarizes stock earnings reports.
    
    ## Features
    
    - **Company Detection**: Automatically identifies companies from names or ticker symbols
    - **IR Site Discovery**: Finds official investor relations websites
    - **Document Parsing**: Extracts content from PDF earnings reports
    - **AI Summarization**: Generates comprehensive earnings summaries
    
    ## How It Works
    
    1. Submit a company name or ticker symbol
    2. The system searches for the investor relations site
    3. Earnings reports are automatically discovered and parsed
    4. An AI agent generates a detailed summary covering:
       - Financial highlights
       - Business segment performance
       - Key metrics and KPIs
       - Management outlook
       - Risks and challenges
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["Earnings Analysis"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Earnings Report Summarizer API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )

