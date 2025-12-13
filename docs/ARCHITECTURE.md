# Earnings Summarizer - Conversational Architecture

## Overview

The Earnings Summarizer now features an **intelligent conversational architecture** that seamlessly handles both earnings analysis and follow-up conversations. This addresses the critical issue where users couldn't respond to AI questions—every input would trigger a new analysis.

## Problem Statement

### Before
- Every user input triggered the full earnings analysis pipeline
- No way to have follow-up conversations with the AI
- Questions from the AI (e.g., "Would you like a deeper dive...?") couldn't be answered
- No session or conversation history
- User responses like "yes" would be treated as a new company ticker

### After
- **Intelligent intent classification** determines if input is a new analysis or a follow-up
- **Session-based conversations** maintain context across interactions
- **Dual-mode operation**: Analysis mode vs Chat mode
- Users can now respond to AI questions naturally
- Conversation history is preserved within sessions

## Architecture Components

### 1. Conversation Router (`conversation_router.py`)

The heart of the new architecture. It classifies user intent and routes to appropriate handlers.

#### Intent Classification
The router classifies every user input into one of four categories:

1. **new_analysis**: User wants to analyze a NEW company
   - Examples: "Apple", "Analyze Tesla", "NVDA"
   - Triggers: Full earnings analysis pipeline
   - Action: Routes to earnings agent with tools

2. **follow_up_question**: User wants MORE DETAIL about current analysis
   - Examples: "Tell me about business segments", "What were key metrics?", "yes"
   - Requires: Existing analysis in session
   - Action: Chat response based on existing analysis

3. **clarification**: User wants to REFINE current analysis
   - Examples: "Actually, I meant Q2", "Can you include more detail?"
   - Action: Chat response with clarification handling

4. **general_chat**: General conversation
   - Examples: "How are you?", "What can you do?", "Help"
   - Action: General chat response

#### Context-Aware Classification
The router is **contextually aware**:
- Considers previous assistant message
- Checks if there's an active analysis in the session
- Handles short affirmatives intelligently (e.g., "yes" after "deeper dive?" = follow_up_question)
- **Maintains conversation history** to combine incomplete inputs (e.g., "Q1" after "NVDA 2022" becomes "NVDA 2022 Q1")
- Handles fiscal year-only inputs (e.g., "2025") by prompting for quarter specification

#### Key Methods

```python
async def classify_intent(
    user_input: str,
    conversation_history: List[Dict[str, str]],
    has_active_analysis: bool,
    previous_message: Optional[str]
) -> IntentClassification
```

Classifies intent using structured LLM output with confidence scoring.

```python
async def route_conversation(
    user_input: str,
    session_data: Dict[str, Any]
) -> Dict[str, Any]
```

Routes conversation to either "analyze" or "chat" action with appropriate context.

### 2. Session Manager (`session_manager.py`)

Manages conversation state and history across interactions.

#### SessionData
Each session stores:
- **conversation_history**: Full chat history (user + assistant messages)
- **last_analysis**: Most recent earnings analysis result
- **metadata**: Additional session information
- **timestamps**: Created and last accessed times

#### Features
- **In-memory storage** (can be extended to Redis/database)
- **Automatic cleanup** of expired sessions (default: 60 minutes)
- **Session persistence** across requests using session_id
- **Background cleanup task** runs every 5 minutes

#### Key Methods

```python
def get_or_create_session(session_id: Optional[str]) -> SessionData
```

Gets existing session or creates new one—ensures continuity.

```python
def add_message(role: str, content: str)
```

Adds message to conversation history with timestamp.

```python
def set_analysis(analysis: Dict[str, Any])
```

Stores analysis result for follow-up questions.

### 3. Chat Endpoint (`/api/v1/chat`)

The unified endpoint that handles all user interactions.

#### Flow

1. **Session Management**
   - Gets or creates session using session_id
   - Adds user message to conversation history

2. **Intent Classification**
   - Routes conversation using ConversationRouter
   - Classifies intent with confidence scoring
   - Determines action: "analyze" or "chat"

3. **Action Execution**

   **If action = "analyze":**
   - Triggers full earnings analysis pipeline
   - Stores result in database AND session
   - Returns analysis with full stages
   - Client shows stage progression UI

   **If action = "chat":**
   - Generates contextual chat response
   - Uses existing analysis if available
   - Returns conversational response
   - Client shows simple "Thinking..." indicator

4. **Response**
   - Returns `ChatResponse` with:
     - `session_id`: For continuity
     - `message`: AI response
     - `action_taken`: "analysis_triggered" or "chat"
     - `intent`: Classified intent type
     - `analysis_result`: Full analysis (if triggered)

#### Streaming Endpoint (`/api/v1/chat/stream`)

A streaming version that provides real-time updates via Server-Sent Events (SSE):

- **Real-time stage updates**: Shows progress through analysis stages
- **Streaming messages**: Messages are streamed as they're generated
- **Better UX**: Users see progress immediately instead of waiting for complete response
- **Stage mapping**: Maps internal agent stages to user-friendly labels

### 4. RAG (Retrieval-Augmented Generation) System (`rag.py`)

The system now uses **RAG with pgvector embeddings** for faster and more efficient transcript processing.

#### How It Works

1. **Chunking**: Long transcripts are split into chunks (default: 1000 characters with 200 overlap)
2. **Embedding**: Each chunk is embedded using OpenAI's `text-embedding-3-small` model (1536 dimensions)
3. **Storage**: Chunks and embeddings are stored in PostgreSQL with pgvector extension
4. **Retrieval**: During summarization, relevant chunks are retrieved using semantic similarity search
5. **Fallback**: If RAG is unavailable, system falls back to full transcript processing

#### Benefits

- **Speed**: Only processes relevant chunks instead of entire transcript
- **Efficiency**: Reduces LLM context usage and processing time
- **Scalability**: Can handle very long transcripts without context limits
- **Reusability**: Once chunked, transcripts can be quickly retrieved for follow-up questions

#### Configuration

```python
# In config.py
rag_enabled: bool = True
rag_chunk_size: int = 1000
rag_chunk_overlap: int = 200
rag_top_k: int = 10  # Number of chunks to retrieve
```

### 5. Streaming Support

The system now supports **streaming responses** for real-time user feedback:

#### Streaming Endpoints

- `/api/v1/chat/stream`: Streaming version of chat endpoint
- `/api/v1/analyze/stream`: Streaming version of analyze endpoint

#### Implementation

- Uses Server-Sent Events (SSE) for real-time updates
- Streams stage updates as analysis progresses
- Provides immediate feedback to users
- Maps internal agent stages to user-friendly labels

#### Stage Updates

The streaming system provides updates at each stage:
- `analyzing_query`: Analyzing user input
- `retrieving_transcript`: Fetching earnings transcript
- `storing_embeddings`: Storing transcript chunks with embeddings
- `complete`: Analysis complete with summary

### 6. Frontend Updates (`App.tsx`)

#### New State Management

```typescript
const [sessionId, setSessionId] = useState<string | null>(null)
const [isAnalyzing, setIsAnalyzing] = useState(false)
```

- **sessionId**: Persists across requests for conversation continuity
- **isAnalyzing**: Distinguishes between analysis and chat loading states

#### Unified Submission Handler

All user input now goes through `/api/v1/chat` or `/api/v1/chat/stream`:

```typescript
const response = await fetch('/api/v1/chat', {
  method: 'POST',
  body: JSON.stringify({ 
    message: currentInput,
    session_id: sessionId 
  }),
})
```

#### Dynamic UI Based on Action

- **Analysis mode**: Shows full stage progression (Analyzing → Retrieving Report → Storing Embeddings → Summarizing)
- **Chat mode**: Shows simple "Thinking..." indicator
- Determined by `data.action_taken` from response

## Data Flow Example

### Scenario 1: New Analysis Request

```
User: "NVDA"
  ↓
Frontend: POST /api/v1/chat { message: "NVDA", session_id: null }
  ↓
Backend: ConversationRouter classifies as "new_analysis"
  ↓
Backend: Runs earnings_agent with tools
  ↓
Backend: Stores analysis in session
  ↓
Frontend: Receives { action_taken: "analysis_triggered", ... }
  ↓
Frontend: Shows stage progression
  ↓
User sees: Full earnings analysis summary
```

### Scenario 2: Follow-up Question

```
User: "yes" (responding to "Want deeper dive?")
  ↓
Frontend: POST /api/v1/chat { message: "yes", session_id: "abc-123" }
  ↓
Backend: Gets session "abc-123" with history
  ↓
Backend: Sees previous message: "Would you like a deeper dive..."
  ↓
Backend: ConversationRouter classifies as "follow_up_question"
  ↓
Backend: Generates chat response using existing analysis
  ↓
Frontend: Receives { action_taken: "chat", ... }
  ↓
Frontend: Shows simple "Thinking..." indicator
  ↓
User sees: Detailed answer about specific section
```

## Benefits

### 1. **Natural Conversations**
Users can now have multi-turn conversations with the AI, responding to questions and asking follow-ups naturally.

### 2. **Context Preservation**
Session-based architecture maintains conversation context, allowing the AI to reference previous analyses and discussions.

### 3. **Intelligent Routing**
Automatic classification means users don't need to think about "modes"—the system intelligently determines what they want.

### 4. **Better UX**
- Follow-up questions are fast (no re-analysis)
- Clear distinction between analysis (with stages) and chat
- Session continuity across multiple interactions

### 5. **Scalability**
The architecture supports:
- Multiple concurrent users (session isolation)
- Easy extension to Redis/database for persistence
- Background cleanup of stale sessions

## Configuration

### Session Timeout
Default: 60 minutes of inactivity

```python
session_manager = SessionManager(session_timeout_minutes=60)
```

### LLM Settings
Intent classification uses low temperature for consistency:

```python
llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.1,  # Consistent classification
)
```

Chat responses use higher temperature for natural conversation:

```python
llm = ChatOpenAI(
    model=settings.llm_model,
    temperature=0.7,  # More creative responses
)
```

## Recent Enhancements

### 1. RAG with pgvector (Implemented)
- ✅ Transcript chunking and embedding storage
- ✅ Semantic search for relevant chunks
- ✅ Faster summarization using retrieved chunks
- ✅ Fallback to full transcript if RAG unavailable

### 2. Streaming Support (Implemented)
- ✅ Real-time stage updates via SSE
- ✅ Streaming chat endpoint (`/api/v1/chat/stream`)
- ✅ Better user experience with immediate feedback

### 3. Improved Conversation History (Implemented)
- ✅ Context-aware intent classification
- ✅ Handles incomplete inputs (e.g., "Q1" after "NVDA 2022")
- ✅ Fiscal year-only input handling
- ✅ Better conversation continuity

## Future Enhancements

### 1. Persistent Sessions
Currently in-memory. Could add:
- Redis for distributed sessions
- Database for long-term persistence
- Session export/import

### 2. Enhanced Intent Classification
- Fine-tune classification with user feedback
- Support more intent types (e.g., "compare", "historical")
- Multi-company analysis

### 3. Conversational Analysis Refinement
- Allow users to refine analysis parameters through conversation
- "Show me only the risk factors"
- "Compare to previous quarter"

### 4. Memory Across Sessions
- Remember user preferences
- Reference previous analyses from different sessions
- User-specific context

### 5. Advanced RAG Features
- Multi-company semantic search
- Historical comparison using embeddings
- Cross-quarter analysis

## Migration Guide

### For Existing Integrations

The old `/api/v1/analyze` endpoint still works for backwards compatibility. To use the new conversational features:

1. **Switch to `/api/v1/chat`**
   ```typescript
   // Old
   POST /api/v1/analyze
   { company_query: "NVDA" }
   
   // New
   POST /api/v1/chat
   { message: "NVDA", session_id: null }
   ```

2. **Store session_id**
   ```typescript
   const [sessionId, setSessionId] = useState<string | null>(null)
   
   // After first request
   setSessionId(response.session_id)
   ```

3. **Pass session_id on subsequent requests**
   ```typescript
   POST /api/v1/chat
   { message: "Tell me about segments", session_id: savedSessionId }
   ```

4. **Handle action_taken**
   ```typescript
   if (data.action_taken === 'analysis_triggered') {
     // Show analysis stages
   } else {
     // Show chat indicator
   }
   ```

## Testing the Conversational Flow

### Test Scenario 1: Basic Conversation

```
User: "Apple"
AI: [Full earnings analysis]
    "Would you like a deeper dive into any specific sections...?"

User: "yes"
AI: [Chat response]
    "I'd be happy to provide more details. Which aspect interests you most:
     - Business segment performance
     - Key metrics..."

User: "business segments"
AI: [Chat response with segment details from analysis]
```

### Test Scenario 2: Intent Classification

```
User: "NVDA"
→ Classified as: new_analysis
→ Action: Full analysis triggered

User: "What were the revenue numbers?"
→ Classified as: follow_up_question
→ Action: Chat response from existing analysis

User: "Actually analyze Microsoft instead"
→ Classified as: new_analysis
→ Action: New analysis triggered
```

## Technical Details

### Structured Output
Intent classification uses Pydantic models for structured LLM output:

```python
class IntentClassification(BaseModel):
    intent: str  # One of: new_analysis, follow_up_question, etc.
    confidence: float  # 0-1
    reasoning: str
    extracted_company: Optional[str]
```

### Error Handling
- Graceful fallback if classification fails
- Error messages added to conversation history
- Session state preserved on errors

### Performance
- Intent classification: ~500ms (LLM call)
- Session lookup: <1ms (in-memory)
- Chat response: ~1-2s (LLM call)
- Full analysis: 10-30s (tool use + LLM)
- **RAG retrieval**: ~200-500ms (vector similarity search)
- **Embedding generation**: ~1-2s per transcript (one-time cost)
- **Streaming updates**: Real-time (no additional latency)

## Conclusion

This conversational architecture transforms the Earnings Summarizer from a single-purpose analysis tool into an interactive AI assistant. Users can now engage in natural multi-turn conversations, ask follow-up questions, and receive contextual responses—all while maintaining the powerful earnings analysis capabilities.

The modular design makes it easy to extend with new features like multi-company comparisons, historical analysis, and personalized preferences.




