# 📊 Earnings Report Summarizer

A multi-agent AI application that automatically analyzes and summarizes stock earnings reports using **LangChain**, **LangGraph**, **FastAPI**, and **React**.

![Architecture](https://img.shields.io/badge/LangGraph-Multi--Agent-blue)
![Backend](https://img.shields.io/badge/FastAPI-Python-green)
![Frontend](https://img.shields.io/badge/React-TypeScript-blue)
![Database](https://img.shields.io/badge/PostgreSQL-pgvector-orange)

## ✨ Features

### 🤖 Conversational AI Interface
- **Natural Conversations**: Ask follow-up questions and have multi-turn conversations
- **Intelligent Intent Recognition**: Automatically distinguishes between new analysis requests and follow-up questions
- **Session-Based Context**: Maintains conversation history and analysis results across interactions
- **Smart Routing**: Seamlessly switches between analysis mode and chat mode based on your intent

### 📈 Earnings Analysis
- **Natural Language Queries**: Enter a company name or ticker symbol (e.g., "Apple" or "AAPL")
- **Web Scraping Integration**: Retrieves earnings call transcripts by scraping discountingcashflows.com
- **Historical Data**: Access to earnings transcripts with comprehensive historical coverage
- **No API Keys Required**: Works directly by scraping publicly available transcript pages
- **RAG-Powered Processing**: Uses pgvector embeddings for fast, efficient transcript analysis
- **Comprehensive Summaries**: Generates detailed summaries covering:
  - Financial highlights (revenue, EPS, margins)
  - Business segment performance
  - Key metrics and KPIs
  - Management outlook and guidance
  - Risks and challenges

## 🏗️ Architecture

The application uses a **conversational AI architecture** with intelligent routing:

```
┌─────────────────┐     ┌─────────────────────────────┐     ┌─────────────────┐
│   React UI      │────▶│   FastAPI Backend           │────▶│   PostgreSQL    │
│   (Chat)        │◀────│   /api/v1/chat             │◀────│   + pgvector    │
└─────────────────┘     └────────────┬────────────────┘     └─────────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │  Conversation Router     │
                        │  (Intent Classification) │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────┴────────────┐
                        │                         │
                  ┌─────▼──────┐          ┌──────▼─────┐
                  │  Analysis  │          │    Chat    │
                  │   Agent    │          │   Agent    │
                  │(Web Scrape)│          │ (Context)  │
                  └─────┬──────┘          └────────────┘
                        │
         ┌──────────────┼──────────────┐
         │              │              │
   ┌─────▼────┐   ┌─────▼────────┐  ┌─────▼────┐
   │  Query   │   │  Transcript  │  │Summarizer│
   │ Analyzer │   │  Retriever   │  └──────────┘
   └──────────┘   └──────────────┘
                        │
            (discountingcashflows.com)
```

### Conversational Flow

The system intelligently routes user input:

1. **Intent Classification**: Determines if input is a new analysis or follow-up question
2. **Route Decision**:
   - **New Analysis** → Triggers multi-agent earnings pipeline (see below)
   - **Follow-Up Question** → Answers from existing analysis using chat agent
   - **General Chat** → Conversational responses about system capabilities
3. **Session Management**: Maintains conversation history and context

### Multi-Agent Flow (LangGraph)

When a new analysis is triggered:

1. **Query Analyzer Agent**: Identifies ticker symbol and lists available transcripts by scraping discountingcashflows.com
2. **Transcript Retriever Agent**: Retrieves the full earnings call transcript by scraping the transcript page
3. **Embedding Storage**: Transcript is chunked and embedded using OpenAI embeddings, stored in PostgreSQL with pgvector
4. **Summarizer Agent**: Uses RAG to retrieve relevant chunks and generates comprehensive financial summaries

### RAG (Retrieval-Augmented Generation) System

The system uses RAG with pgvector for efficient transcript processing:

1. **Chunking**: Long transcripts are split into manageable chunks (1000 chars with 200 overlap)
2. **Embedding**: Each chunk is embedded using OpenAI's `text-embedding-3-small` model
3. **Storage**: Chunks and embeddings stored in PostgreSQL with pgvector extension
4. **Retrieval**: Semantic search retrieves relevant chunks based on query similarity
5. **Summarization**: Only relevant chunks are processed, reducing LLM context usage and improving speed

**Benefits**:
- 30-50% faster summarization
- Handles very long transcripts without context limits
- Reusable embeddings for follow-up questions
- More efficient LLM usage

For detailed architecture documentation, see [ARCHITECTURE.md](./ARCHITECTURE.md).

## 🎯 Recent Updates

### RAG with pgvector Embeddings
- ✅ **Faster Processing**: Transcripts are chunked and embedded for efficient semantic search
- ✅ **Speed Improvements**: 30-50% faster summarization using relevant chunks
- ✅ **Scalability**: Handles very long transcripts without context limits
- ✅ **Reusability**: Once embedded, transcripts can be quickly retrieved for follow-ups

### Streaming Support
- ✅ **Real-time Updates**: `/api/v1/chat/stream` endpoint provides live progress updates
- ✅ **Better UX**: Users see immediate feedback during analysis
- ✅ **Server-Sent Events**: Uses SSE for efficient real-time communication

### Enhanced Conversation Handling
- ✅ **Context-Aware**: Handles incomplete inputs (e.g., "Q1" after "NVDA 2022")
- ✅ **Fiscal Year Handling**: Intelligently prompts for quarter when only year is provided
- ✅ **Conversation History**: Maintains full context across interactions

### Web Scraping Integration
- ✅ No API keys required
- ✅ Direct access to publicly available transcripts
- ✅ Comprehensive historical transcript coverage
- ✅ BeautifulSoup-based HTML parsing for reliable extraction

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Node.js 18+
- Docker & Docker Compose (recommended)
- OpenAI API key (required)

### Option 1: Docker (Recommended)

1. **Clone and configure:**
   ```bash
   cd earnings-summarizer
   
   # Create environment file
   echo "OPENAI_API_KEY=your-openai-key-here" > .env
   ```

2. **Start all services:**
   ```bash
   docker-compose up --build
   # Or with Make:
   make docker-up
   ```

3. **Access the application:**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Option 2: Manual Setup

**Quick Start (using Make):**
```bash
# Install all dependencies
make install

# Start development servers
make dev
```

#### Backend Setup

**Quick Setup (using setup script):**
```bash
cd backend

# macOS/Linux:
chmod +x setup.sh && ./setup.sh

# Windows PowerShell:
# .\setup.ps1
```

**Manual Setup:**
```bash
cd backend

# Install uv (if not already installed)
# macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Create virtual environment and install dependencies with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies (much faster than pip!)
uv pip install -r requirements.txt
# Or use: uv sync (if using pyproject.toml)

# Create .env file
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Start the server
python run.py
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

#### Database Setup

```bash
# Using Docker for PostgreSQL with pgvector
docker run -d \
  --name earnings-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=earnings_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

## 📖 API Reference

### Chat (Recommended - Conversational Interface)

```http
POST /api/v1/chat
Content-Type: application/json

{
  "message": "Apple",
  "session_id": null  // or existing session_id for continuity
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "message": "## Financial Highlights\n\n...",
  "action_taken": "analysis_triggered",  // or "chat"
  "intent": "new_analysis",
  "analysis_result": {
    "session_id": "uuid",
    "company_query": "Apple",
    "status": "complete",
    "summary": "...",
    "messages": [...]
  }
}
```

**Example Conversation:**
```bash
# First request - triggers analysis
POST /api/v1/chat
{ "message": "NVDA", "session_id": null }
→ Returns analysis + session_id

# Follow-up question - uses existing analysis
POST /api/v1/chat
{ "message": "Tell me about business segments", "session_id": "abc-123" }
→ Returns chat response based on previous analysis

# Another follow-up
POST /api/v1/chat
{ "message": "yes", "session_id": "abc-123" }
→ Intelligently handles affirmative responses
```

### Analyze Earnings (Legacy - Direct Analysis)

```http
POST /api/v1/analyze
Content-Type: application/json

{
  "company_query": "Apple"
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "company_query": "Apple",
  "status": "complete",
  "summary": "## Financial Highlights\n\n...",
  "messages": [...],
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Stream Analysis (SSE)

```http
POST /api/v1/analyze/stream
Content-Type: application/json

{
  "company_query": "NVDA"
}
```

Returns Server-Sent Events with real-time progress updates.

### Stream Chat (SSE)

```http
POST /api/v1/chat/stream
Content-Type: application/json

{
  "message": "NVDA",
  "session_id": null
}
```

Returns Server-Sent Events with real-time updates during analysis or chat. Provides stage updates and streaming messages.

### List Sessions

```http
GET /api/v1/sessions?limit=20&offset=0
```

### Get Session

```http
GET /api/v1/sessions/{session_id}
```

## 🛠️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql+asyncpg://postgres:postgres@localhost:5432/earnings_db` |
| `LLM_MODEL` | OpenAI model to use | `gpt-4.1-mini` |
| `LLM_TEMPERATURE` | LLM temperature | `0.1` |
| `HOST` | Backend host | `0.0.0.0` |
| `PORT` | Backend port | `8000` |
| `RAG_ENABLED` | Enable RAG with pgvector | `true` |
| `RAG_CHUNK_SIZE` | Characters per chunk | `1000` |
| `RAG_CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `RAG_TOP_K` | Number of chunks to retrieve | `10` |

## ⚠️ Known Limitations & Complications

### 1. Web Scraping Considerations
- Website structure changes may break the scraper
- Rate limiting: Be respectful of discountingcashflows.com - implement delays between requests for production use
- Network dependency: Requires stable internet connection to access transcript pages

### 2. Transcript Availability
- Not all companies may have transcripts available on discountingcashflows.com
- Some transcripts may be delayed or unavailable for certain periods
- Try alternative ticker symbols if a company isn't found
- Transcript page structure may vary, requiring selector updates

### 3. LLM Context Limits
- Very long transcripts are automatically chunked and embedded using RAG
- RAG system retrieves relevant chunks instead of processing full transcript
- First-time analysis includes embedding generation (one-time cost)

## 📁 Project Structure

```
earnings-summarizer/
├── backend/
│   ├── app/
│   │   ├── agents/           # LangGraph multi-agent system
│   │   │   ├── earnings_agent.py      # Earnings analysis pipeline
│   │   │   └── conversation_router.py # Intent classification & routing
│   │   ├── api/              # FastAPI routes
│   │   │   └── routes.py     # /chat, /analyze endpoints
│   │   ├── tools/            # LangChain tools
│   │   │   └── investor_relations.py  # Web scraping for earnings transcripts from discountingcashflows.com
│   │   ├── config.py         # Settings
│   │   ├── database.py       # SQLAlchemy + pgvector
│   │   ├── rag.py            # RAG service with embeddings
│   │   ├── session_manager.py # Conversation session management
│   │   ├── schemas.py        # Pydantic models
│   │   └── main.py           # FastAPI app
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main React component (conversational UI)
│   │   ├── main.tsx
│   │   └── index.css         # Tailwind styles
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── ARCHITECTURE.md           # Detailed architecture documentation
└── README.md
```

## 🧪 Development

### Running Tests

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

### Code Quality

```bash
# Backend linting
cd backend
ruff check .
black --check .

# Frontend linting
cd frontend
npm run lint
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Install dependencies with `uv sync`
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Multi-agent orchestration
- [FastAPI](https://fastapi.tiangolo.com/) - Backend framework
- [React](https://react.dev/) - Frontend framework
- [pgvector](https://github.com/pgvector/pgvector) - Vector similarity search
- [uv](https://github.com/astral-sh/uv) - Fast Python package manager

