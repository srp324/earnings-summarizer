# 📊 Earnings Report Summarizer

A multi-agent AI application that automatically analyzes and summarizes stock earnings reports using **LangChain**, **LangGraph**, **FastAPI**, and **React**.

![Architecture](https://img.shields.io/badge/LangGraph-Multi--Agent-blue)
![Backend](https://img.shields.io/badge/FastAPI-Python-green)
![Frontend](https://img.shields.io/badge/React-TypeScript-blue)
![Database](https://img.shields.io/badge/PostgreSQL-pgvector-orange)

## ✨ Features

- **Natural Language Queries**: Enter a company name or ticker symbol (e.g., "Apple" or "AAPL")
- **Automatic IR Discovery**: AI agents find the company's investor relations website
- **Document Extraction**: Automatically extracts links to earnings reports (10-K, 10-Q, etc.)
- **PDF Parsing**: Parses PDF earnings reports with table extraction
- **Comprehensive Summaries**: Generates detailed summaries covering:
  - Financial highlights (revenue, EPS, margins)
  - Business segment performance
  - Key metrics and KPIs
  - Management outlook and guidance
  - Risks and challenges

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   React UI      │────▶│   FastAPI       │────▶│   PostgreSQL    │
│   (Chat)        │◀────│   Backend       │◀────│   + pgvector    │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
              ┌─────▼────┐ ┌─────▼────┐ ┌─────▼────┐
              │  Query   │ │   IR     │ │ Document │
              │ Analyzer │ │  Finder  │ │  Parser  │
              │  Agent   │ │  Agent   │ │  Agent   │
              └──────────┘ └──────────┘ └──────────┘
                                │
                          ┌─────▼────┐
                          │Summarizer│
                          │  Agent   │
                          └──────────┘
```

### Multi-Agent Flow (LangGraph)

1. **Query Analyzer Agent**: Understands user input, identifies company/ticker
2. **IR Finder Agent**: Searches for and validates investor relations websites
3. **Document Extractor Agent**: Finds links to earnings reports on IR pages
4. **Document Parser Agent**: Downloads and parses PDF/HTML earnings documents
5. **Summarizer Agent**: Generates comprehensive financial summaries

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (Python package manager)
- Node.js 18+
- Docker & Docker Compose (recommended)
- OpenAI API key

### Option 1: Docker (Recommended)

1. **Clone and configure:**
   ```bash
   cd earnings-summarizer
   
   # Create environment file
   echo "OPENAI_API_KEY=your-api-key-here" > .env
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

# Install Playwright browsers (for web scraping)
playwright install chromium

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

### Analyze Earnings

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
| `LLM_MODEL` | OpenAI model to use | `gpt-4o` |
| `LLM_TEMPERATURE` | LLM temperature | `0.1` |
| `HOST` | Backend host | `0.0.0.0` |
| `PORT` | Backend port | `8000` |

## ⚠️ Known Limitations & Complications

### 1. PDF Parsing Challenges
- Some PDFs contain scanned images instead of text
- Complex table layouts may not parse correctly
- Very large documents (100+ pages) are truncated

### 2. Dynamic Website Content
- Some IR sites load content via JavaScript
- May require Playwright for full rendering (currently uses HTTP requests)

### 3. Rate Limiting
- Search APIs and IR sites may rate-limit requests
- Implement delays between requests for production use

### 4. Document Discovery
- Companies use different naming conventions (10-K, Annual Report, etc.)
- Some sites require authentication or are behind paywalls

### 5. LLM Context Limits
- Very long documents are chunked/truncated
- Consider implementing RAG with pgvector for large documents

## 📁 Project Structure

```
earnings-summarizer/
├── backend/
│   ├── app/
│   │   ├── agents/           # LangGraph multi-agent system
│   │   │   └── earnings_agent.py
│   │   ├── api/              # FastAPI routes
│   │   │   └── routes.py
│   │   ├── tools/            # LangChain tools
│   │   │   ├── web_search.py
│   │   │   ├── document_parser.py
│   │   │   └── investor_relations.py
│   │   ├── config.py         # Settings
│   │   ├── database.py       # SQLAlchemy + pgvector
│   │   ├── schemas.py        # Pydantic models
│   │   └── main.py           # FastAPI app
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Main React component
│   │   ├── main.tsx
│   │   └── index.css         # Tailwind styles
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
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

