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

**Note**: The database is optional for basic debugging. The app will start even if the database connection fails (it will show a warning but continue). However, for full functionality (RAG embeddings, session management), you'll need PostgreSQL.

**Quick Setup (Recommended):**

```bash
# Option 1: Using Docker Compose (easiest)
docker-compose up postgres -d

# Option 2: Using Docker directly
docker run -d \
  --name earnings-postgres \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=earnings_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

**Verify database is running:**
```bash
# Check if PostgreSQL is running
docker ps | grep earnings-postgres
# Or check port
netstat -ano | findstr :5432  # Windows
lsof -i :5432  # Mac/Linux
```

**For debugging without database:**
- The app will start and work for basic API calls
- You'll see a warning: "Database initialization skipped"
- RAG features and session persistence won't work
- This is fine for frontend/backend integration debugging

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
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | `true` (optional) |
| `LANGCHAIN_API_KEY` | LangSmith API key for tracing | - |
| `LANGCHAIN_PROJECT` | LangSmith project name | `earnings-summarizer` (optional) |

### LangSmith Tracing (Optional)

To visualize LangGraph graph traversals in LangSmith:

1. **Get a LangSmith API key:**
   - Sign up at https://smith.langchain.com
   - Go to Settings → API Keys
   - Create a new API key

2. **Add to your `.env` file:**
   ```bash
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_API_KEY=your-api-key-here
   LANGCHAIN_PROJECT=earnings-summarizer
   ```

3. **Run your application** - LangGraph automatically sends traces when these env vars are set

4. **View traces in LangSmith:**
   - Open https://smith.langchain.com
   - Navigate to your project
   - Click on any trace to see the graph traversal visualization

**Note:** LangSmith is a cloud service (not a local CLI tool). The graph visualization happens in the web UI. You can also use `langsmith-fetch` CLI tool to fetch traces programmatically if needed.

### LangGraph Studio (Local Development)

LangGraph Studio provides a local IDE for visualizing and debugging your LangGraph agents. To use it:

1. **Install dependencies** (if not already installed):
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Start LangGraph Studio:**
   ```bash
   cd backend
   langgraph dev
   ```

3. **Access the Studio UI:**
   - The command will output a URL like: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`
   - Open this URL in your browser
   - You'll see your graph visualization and can interact with it

4. **Optional: Use tunnel mode (for Safari or secure connections):**
   ```bash
   langgraph dev --tunnel
   ```

The Studio will:
- Visualize your graph structure
- Show graph traversal in real-time
- Allow you to test your agent interactively
- Debug node executions step-by-step

**Note:** Make sure your `.env` file is configured with `OPENAI_API_KEY` before running.

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

### Debugging in Windsurf/VS Code

The project includes debug configurations for both backend and frontend:

#### Available Debug Configurations

1. **Python: FastAPI Backend** - Debug the Python backend server
   - Sets breakpoints in Python files
   - Hot reload enabled
   - Runs on `http://localhost:8000`

2. **Python: FastAPI Backend (via run.py)** - Alternative backend debugger using `run.py`
   - Same as above but uses the `run.py` entry point

3. **Chrome: Frontend** - Debug the React frontend in Chrome
   - Sets breakpoints in TypeScript/React files
   - Automatically starts the Vite dev server
   - Opens Chrome at `http://localhost:5173`

4. **Debug Full Stack** - Debug both backend and frontend simultaneously
   - Launches both servers
   - Allows debugging across the full stack
   - **Note**: If you see an error about "compound" type, you can run the configurations separately:
     1. Start "Python: FastAPI Backend (Debug Mode)" first
     2. Then start "Chrome: Frontend (server already running)" once backend is ready

#### How to Use

**For Full Stack Debugging (Recommended Approach):**

Since compound configurations may not work reliably in all environments, use this two-step process:

1. **Start the Backend First:**
   - Select "Python: FastAPI Backend (Debug Mode)" from the debug dropdown
   - Press `F5` to start debugging
   - **Wait** until you see "Uvicorn running on http://0.0.0.0:8000" in the terminal
   - Verify it's running: Open http://localhost:8000/docs in your browser

2. **Then Start the Frontend:**
   - Select "Chrome: Frontend (server already running)" from the debug dropdown
   - Press `F5` to start debugging
   - Chrome will open automatically with the frontend

**Alternative: Individual Debugging**

1. **Set breakpoints** in your code by clicking in the gutter next to line numbers

2. **Start debugging:**
   - Press `F5` or go to Run and Debug panel (Ctrl+Shift+D)
   - Select a debug configuration from the dropdown
   - Click the green play button or press `F5`

3. **Debug controls:**
   - `F5` - Continue
   - `F10` - Step Over
   - `F11` - Step Into
   - `Shift+F11` - Step Out
   - `Shift+F5` - Stop

#### Prerequisites

- **Python Extension**: Install the Python extension for VS Code/Windsurf
- **Chrome Debugger Extension**: Install the "Debugger for Chrome" extension (if not already included)
- **Virtual Environment**: Ensure your Python virtual environment is activated (`.venv` in `backend/`)

#### Troubleshooting

- **Python path issues**: The configuration uses `${command:python.interpreterPath}`. Make sure you've selected the correct Python interpreter (Ctrl+Shift+P → "Python: Select Interpreter")
- **Backend not starting in debugger**: 
  - **CRITICAL: Verify Python interpreter is selected:**
    1. Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
    2. Type "Python: Select Interpreter"
    3. **Choose the interpreter from `backend/.venv/Scripts/python.exe`** (Windows) or `backend/.venv/bin/python` (Mac/Linux)
    4. It should show something like: "Python 3.11.x ('venv': venv)"
  - **Test backend manually first** (this verifies everything works):
    ```bash
    cd backend
    
    # On Windows (PowerShell/CMD):
    .venv\Scripts\activate
    python run_debug.py
    
    # On Windows (Git Bash):
    .venv/Scripts/python.exe run_debug.py
    
    # On Mac/Linux:
    source .venv/bin/activate
    python run_debug.py
    ```
    If this works, the backend is fine - the issue is with the debugger configuration.
  - **Check Debug Console**: When debugging, look for:
    - A separate terminal/console showing Python output
    - You should see: `Starting Earnings Summarizer API...` and `INFO: Uvicorn running on http://0.0.0.0:8000`
    - If you don't see this, the debugger isn't starting the backend
  - **Verify virtual environment exists:**
    - Windows: Check for `backend/.venv/Scripts/python.exe`
    - Mac/Linux: Check for `backend/.venv/bin/python`
    - If missing, run: `cd backend && uv venv && uv pip install -r requirements.txt`
  - **Ensure `.env` file exists** in `backend/` directory with `OPENAI_API_KEY` set
  - **If backend debugger still doesn't start**: Use the manual approach - start backend manually, then debug frontend
- **Database authentication errors**: 
  - **This is OK for debugging!** The app will start even if the database connection fails
  - You'll see: "Database initialization skipped: [error]" - this is expected
  - To fix: Start PostgreSQL with `docker-compose up postgres -d` or see Database Setup section
  - For basic debugging, you can ignore database errors - the API will still work
- **Frontend proxy errors (ECONNREFUSED)**: 
  - This is normal if the backend hasn't started yet - the frontend will retry when you make a request
  - Verify the backend is running by checking for a terminal with "Uvicorn running on http://0.0.0.0:8000"
  - If backend is running but still getting errors, check that it's listening on `localhost:8000` (not just `0.0.0.0`)
- **Frontend not starting / "vite is not recognized"**: 
  - First, ensure Node.js dependencies are installed:
    ```bash
    cd frontend
    npm install
    ```
  - If the error persists, manually start the frontend dev server in a terminal:
    ```bash
    cd frontend
    npm run dev
    ```
  - Then use the "Chrome: Frontend (server already running)" debug configuration instead
- **Port conflicts**: If ports 8000 or 5173 are in use, stop those services first
- **Windows-specific issues**: If npm commands fail, try using `npm.cmd` explicitly or ensure npm is in your PATH

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

