"""RAG (Retrieval-Augmented Generation) utilities for earnings transcripts.

This module handles:
- Text chunking for long transcripts
- Embedding generation using OpenAI
- Vector storage in PostgreSQL with pgvector
- Semantic search for retrieving relevant chunks
"""

from typing import List, Dict, Optional, Tuple, Any
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, text
from sqlalchemy.sql import func
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.database import EarningsDocument, AsyncSessionLocal

logger = logging.getLogger(__name__)

# Default configuration (can be overridden by settings)
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model (1536 dimensions)


class RAGService:
    """Service for RAG operations: chunking, embedding, and retrieval."""
    
    def __init__(self):
        settings = get_settings()
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=settings.openai_api_key,
        )
        chunk_size = getattr(settings, "rag_chunk_size", DEFAULT_CHUNK_SIZE)
        chunk_overlap = getattr(settings, "rag_chunk_overlap", DEFAULT_CHUNK_OVERLAP)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],  # Try to split on paragraphs first
        )
        self.top_k = getattr(settings, "rag_top_k", 10)
    
    async def chunk_and_store_transcript(
        self,
        transcript_content: str,
        ticker_symbol: str,
        company_name: str,
        fiscal_year: str,
        quarter: str,
        source_url: str,
    ) -> int:
        """
        Chunk a transcript, generate embeddings, and store in database.
        
        Args:
            transcript_content: Full transcript text
            ticker_symbol: Stock ticker (e.g., "NVDA")
            company_name: Company name (e.g., "NVIDIA")
            fiscal_year: Fiscal year (e.g., "2025")
            quarter: Quarter number (e.g., "2")
            source_url: URL where transcript was retrieved from
        
        Returns:
            Number of chunks stored
        """
        # Delete existing chunks for this transcript to avoid duplicates
        await self._delete_existing_chunks(ticker_symbol, fiscal_year, quarter)
        
        # Chunk the transcript
        chunks = self.text_splitter.split_text(transcript_content)
        logger.info(f"Split transcript into {len(chunks)} chunks for {ticker_symbol} FY{fiscal_year} Q{quarter}")
        
        if not chunks:
            logger.warning("No chunks created from transcript")
            return 0
        
        # Generate embeddings for all chunks
        logger.info("Generating embeddings for chunks...")
        try:
            embeddings_list = await self.embeddings.aembed_documents(chunks)
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
        
        # Store chunks and embeddings in database
        async with AsyncSessionLocal() as session:
            fiscal_period = f"FY{fiscal_year} Q{quarter}"
            
            for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings_list)):
                doc = EarningsDocument(
                    ticker_symbol=ticker_symbol.upper(),
                    company_name=company_name,
                    document_type="earnings_call_transcript",
                    fiscal_period=fiscal_period,
                    source_url=source_url,
                    content=chunk,
                    chunk_index=idx,
                    embedding=embedding,  # pgvector will handle the conversion
                )
                session.add(doc)
            
            await session.commit()
            logger.info(f"Stored {len(chunks)} chunks with embeddings in database")
            return len(chunks)
    
    async def retrieve_relevant_chunks(
        self,
        query: str,
        ticker_symbol: Optional[str] = None,
        fiscal_year: Optional[str] = None,
        quarter: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks using semantic search.
        
        Args:
            query: Search query (will be embedded and used for similarity search)
            ticker_symbol: Optional ticker to filter by
            fiscal_year: Optional fiscal year to filter by
            quarter: Optional quarter to filter by
            top_k: Number of top chunks to retrieve
        
        Returns:
            List of dicts with keys: content, ticker_symbol, fiscal_period, chunk_index, similarity_score
        """
        # Generate embedding for the query
        try:
            query_embedding = await self.embeddings.aembed_query(query)
        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            raise
        
        # Build query
        async with AsyncSessionLocal() as session:
            # Start with base query
            base_query = select(EarningsDocument)
            
            # Add filters if provided
            if ticker_symbol:
                base_query = base_query.where(
                    EarningsDocument.ticker_symbol == ticker_symbol.upper()
                )
            if fiscal_year and quarter:
                fiscal_period = f"FY{fiscal_year} Q{quarter}"
                base_query = base_query.where(
                    EarningsDocument.fiscal_period == fiscal_period
                )
            
            # Perform vector similarity search using pgvector
            
            # Use raw SQL for vector similarity search since SQLAlchemy ORM doesn't have direct support
            # pgvector supports <-> operator for L2 distance and <#> for inner product
            # We'll use 1 - inner product (since embeddings are normalized, inner product = cosine similarity)
            # Actually, for cosine similarity, we need to use the cosine distance operator
            
            # Build the WHERE clause for filters
            where_clauses = []
            params = {}
            
            # Convert embedding list to PostgreSQL array format string
            # pgvector expects format like '[0.1,0.2,0.3]'
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            if ticker_symbol:
                where_clauses.append("ticker_symbol = :ticker_symbol")
                params["ticker_symbol"] = ticker_symbol.upper()
            if fiscal_year and quarter:
                fiscal_period = f"FY{fiscal_year} Q{quarter}"
                where_clauses.append("fiscal_period = :fiscal_period")
                params["fiscal_period"] = fiscal_period
            
            where_sql = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            # Use cosine distance operator <=> (lower is better, so we order ASC)
            # cosine_distance = 1 - cosine_similarity, so similarity = 1 - distance
            sql = text(f"""
                SELECT 
                    id, content, ticker_symbol, fiscal_period, chunk_index, source_url,
                    1 - (embedding <=> CAST(:query_embedding AS vector)) AS similarity
                FROM earnings_documents
                WHERE {where_sql}
                ORDER BY embedding <=> CAST(:query_embedding AS vector)
                LIMIT :top_k
            """)
            params["query_embedding"] = embedding_str
            params["top_k"] = top_k
            
            result = await session.execute(sql, params)
            rows = result.fetchall()
            
            # Convert to dicts
            chunks = []
            for row in rows:
                chunks.append({
                    "content": row.content,
                    "ticker_symbol": row.ticker_symbol,
                    "fiscal_period": row.fiscal_period,
                    "chunk_index": row.chunk_index,
                    "similarity_score": float(row.similarity) if row.similarity else 0.0,
                    "source_url": row.source_url,
                })
            
            logger.info(f"Retrieved {len(chunks)} relevant chunks for query: {query[:50]}...")
            return chunks
    
    async def retrieve_all_chunks_for_transcript(
        self,
        ticker_symbol: str,
        fiscal_year: str,
        quarter: str,
    ) -> List[str]:
        """
        Retrieve all chunks for a specific transcript (ordered by chunk_index).
        Useful as a fallback when we want the full transcript but it's already chunked.
        
        Args:
            ticker_symbol: Stock ticker
            fiscal_year: Fiscal year
            quarter: Quarter number
        
        Returns:
            List of chunk contents in order
        """
        fiscal_period = f"FY{fiscal_year} Q{quarter}"
        
        async with AsyncSessionLocal() as session:
            query = select(EarningsDocument).where(
                EarningsDocument.ticker_symbol == ticker_symbol.upper(),
                EarningsDocument.fiscal_period == fiscal_period,
            ).order_by(EarningsDocument.chunk_index)
            
            result = await session.execute(query)
            docs = result.scalars().all()
            
            return [doc.content for doc in docs]
    
    async def _delete_existing_chunks(
        self,
        ticker_symbol: str,
        fiscal_year: str,
        quarter: str,
    ):
        """Delete existing chunks for a transcript before storing new ones."""
        fiscal_period = f"FY{fiscal_year} Q{quarter}"
        
        async with AsyncSessionLocal() as session:
            delete_query = delete(EarningsDocument).where(
                EarningsDocument.ticker_symbol == ticker_symbol.upper(),
                EarningsDocument.fiscal_period == fiscal_period,
            )
            result = await session.execute(delete_query)
            await session.commit()
            
            if result.rowcount > 0:
                logger.info(f"Deleted {result.rowcount} existing chunks for {ticker_symbol} {fiscal_period}")
    
    async def transcript_is_chunked(
        self,
        ticker_symbol: str,
        fiscal_year: str,
        quarter: str,
    ) -> bool:
        """Check if a transcript is already chunked and stored."""
        fiscal_period = f"FY{fiscal_year} Q{quarter}"
        
        async with AsyncSessionLocal() as session:
            query = select(func.count(EarningsDocument.id)).where(
                EarningsDocument.ticker_symbol == ticker_symbol.upper(),
                EarningsDocument.fiscal_period == fiscal_period,
            )
            result = await session.execute(query)
            count = result.scalar() or 0
            
            return count > 0


# Global instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create the global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service

