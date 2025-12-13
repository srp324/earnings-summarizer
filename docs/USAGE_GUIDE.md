# Usage Guide - Conversational Earnings Summarizer

## Quick Start

### Starting a New Analysis

Simply type a company name or ticker symbol:

```
You: "Apple"
```

The system will:
1. Recognize this as a new analysis request
2. Find Apple's investor relations site
3. Extract and parse earnings reports
4. Generate a comprehensive summary

### Having a Conversation

After receiving an analysis, you can ask follow-up questions naturally:

```
You: "Tell me more about the business segments"
AI: [Detailed response about Apple's segments from the analysis]

You: "What were the key metrics?"
AI: [Response with metrics from the analysis]

You: "How did they perform compared to expectations?"
AI: [Contextual response based on the analysis]
```

## Example Conversations

### Example 1: Basic Analysis with Follow-ups

```
You: "NVDA"

AI: [Full earnings analysis for NVIDIA's most recent earnings report]
    
    The system automatically retrieved NVIDIA's latest available earnings 
    transcript and generated a comprehensive summary covering financial 
    highlights, business segments, key metrics, and management outlook.
    
    Would you like a deeper dive into any specific sections of NVIDIA's 
    earnings report, such as business segment performance, key metrics, 
    or management's outlook?

You: "yes"

AI: I'd be happy to provide more details. Which aspect interests you most:
    - Business segment performance (Gaming, Data Center, Professional 
      Visualization, Automotive)
    - Key financial metrics (revenue growth, margins, EPS)
    - Management's outlook and guidance
    - Risks and challenges mentioned
    
    Or I can provide a comprehensive deep dive covering all of these areas.

You: "data center performance"

AI: [Detailed analysis of NVIDIA's Data Center segment from the earnings report]
```

**Note**: When you provide only a company name or ticker symbol (like "NVDA"), the system automatically retrieves and analyzes the **most recent** earnings transcript available. You don't need to specify a year or quarter - the system will use the latest report by default.

### Example 2: Multiple Companies

```
You: "Microsoft"

AI: [Full earnings analysis for Microsoft]

You: "Now analyze Apple"

AI: [Full earnings analysis for Apple - recognized as new analysis request]

You: "Compare their cloud revenues"

AI: Based on the analyses:
    - Microsoft Azure: [details from analysis]
    - Apple Services: [details from analysis]
    [Comparison and insights]
```

### Example 3: Clarifications

```
You: "Tesla earnings"

AI: [Full earnings analysis for Tesla]

You: "Actually, I meant the Q2 report specifically"

AI: I've analyzed Tesla's most recent earnings report. If you need a 
    specific quarter's report, you can:
    1. Specify "Tesla Q2 2024" for a targeted search
    2. Ask me to focus on specific sections of the current analysis
    
    Would you like me to help you find the Q2 report?
```

## Understanding System Behavior

### When Analysis is Triggered

The system triggers a **new analysis** when you:
- Mention a company name or ticker: "Apple", "MSFT", "Tesla"
- Use analysis keywords: "Analyze Google", "Tell me about Amazon's earnings"
- Request a different company: "Now do Microsoft"

**UI Behavior**: Shows full stage progression (Analyzing → Retrieving Report → Parsing → Summarizing)

### When Chat Mode is Used

The system uses **chat mode** when you:
- Ask follow-up questions: "Tell me more about...", "What were the..."
- Respond to AI questions: "yes", "sure", "tell me about segments"
- Request clarification: "Can you explain that?", "What does that mean?"
- Have general conversation: "What can you do?", "Help"

**UI Behavior**: Shows simple "Thinking..." indicator

## Tips for Best Results

### 1. Be Specific with the Fiscal Year and Quarter

✅ Good:
- "Apple FY2025 Q3"
- "NVDA Q2 2021"
- "Microsoft Corporation 2024Q2"

❌ Less Effective:
- "that tech company" (too vague)
- "the iPhone maker" (requires inference)

### 2. Ask Focused Follow-up Questions

✅ Good:
- "What were the revenue numbers?"
- "Tell me about the Data Center segment"
- "What did management say about guidance?"

❌ Less Effective:
- "Tell me everything again" (redundant)
- "What about other stuff?" (too vague)

### 3. Use Natural Language

You don't need to use special commands or formats. Just talk naturally:

✅ Good:
- "yes"
- "Tell me more about business segments"
- "How did they perform?"
- "What were the risks mentioned?"

### 4. Session Continuity

Your conversation is maintained within a session:
- Follow-up questions reference the current analysis
- Context is preserved across multiple exchanges
- Sessions expire after 60 minutes of inactivity

To start fresh:
- Refresh the page
- Request a different company (triggers new analysis)

## Advanced Usage

### Asking for Specific Sections

You can ask about specific parts of earnings reports:

```
"What did management say about outlook?"
"Tell me about the risks and challenges"
"What were the key metrics?"
"How did different business segments perform?"
"What was the revenue breakdown?"
```

### Comparative Questions

After analyzing a company, you can ask comparative questions:

```
"How does this compare to last quarter?"
"Is this better than expected?"
"What's different from previous reports?"
```

*Note: Comparisons work best when information is available in the current analysis*

### Clarifying Responses

If you get a response that doesn't match your intent:

```
You: "Apple"
AI: [Starts analysis]

You: "Wait, I meant to ask about Apple's supply chain, not analyze earnings"
AI: [Clarifies and adjusts response]
```

## Troubleshooting

### "I can't find information about that"

This means the requested information wasn't in the analyzed earnings report. Try:
- Asking a different question about available information
- Requesting a new analysis if you need different data

### Analysis Takes Too Long

Earnings analysis involves:
- Searching for investor relations sites
- Downloading and parsing PDFs
- LLM processing of large documents

Typical time: 10-30 seconds

If it takes longer:
- Check your internet connection
- Verify the backend is running
- Check backend logs for errors

### Session Lost

If you see "No analysis available" when asking follow-up questions:
- Your session may have expired (60 min timeout)
- Refresh the page and start a new analysis

## API Usage

### Using the Chat Endpoint Programmatically

```python
import requests

# First request - triggers analysis
response = requests.post('http://localhost:8000/api/v1/chat', json={
    'message': 'Apple',
    'session_id': None
})

data = response.json()
session_id = data['session_id']
print(data['message'])  # Earnings summary

# Follow-up question
response = requests.post('http://localhost:8000/api/v1/chat', json={
    'message': 'Tell me about business segments',
    'session_id': session_id
})

data = response.json()
print(data['message'])  # Detailed segment information
```

### Using the Streaming Chat Endpoint

For real-time progress updates, use the streaming endpoint:

```python
import requests
import json

# Streaming chat endpoint
response = requests.post(
    'http://localhost:8000/api/v1/chat/stream',
    json={'message': 'NVDA', 'session_id': None},
    stream=True
)

session_id = None
for line in response.iter_lines():
    if line:
        # Parse SSE format: "data: {...}"
        if line.startswith(b'data: '):
            data = json.loads(line[6:])  # Skip "data: " prefix
            
            if data.get('type') == 'stage_update':
                print(f"Stage: {data.get('stage_label')} - {data.get('status')}")
            elif data.get('type') == 'complete':
                session_id = data.get('session_id')
                print(f"Summary: {data.get('message')}")
```

### Handling Incomplete Inputs

The system now intelligently handles incomplete inputs by combining them with conversation history:

```python
# First message
response = requests.post('http://localhost:8000/api/v1/chat', json={
    'message': 'NVDA 2022',
    'session_id': None
})
data = response.json()
session_id = data['session_id']

# Follow-up with just quarter - system combines with previous context
response = requests.post('http://localhost:8000/api/v1/chat', json={
    'message': 'Q1',  # System understands this as "NVDA 2022 Q1"
    'session_id': session_id
})
```

### Handling Fiscal Year Only

If you provide just a fiscal year, the system will prompt for a quarter:

```python
# Fiscal year only
response = requests.post('http://localhost:8000/api/v1/chat', json={
    'message': 'NVDA 2025',
    'session_id': None
})
# System will ask which quarter you want
```

### Checking Action Taken

```python
if data['action_taken'] == 'analysis_triggered':
    print("New analysis was performed")
    analysis = data['analysis_result']
    print(f"Company: {analysis['company_query']}")
    print(f"Status: {analysis['status']}")
elif data['action_taken'] == 'chat':
    print("Chat response based on existing context")
```

## Keyboard Shortcuts

- **Enter**: Send message
- **Shift+Enter**: New line in input (if multiline input is enabled)

## Privacy & Data

- **Sessions**: Stored in-memory, expire after 60 minutes
- **Conversations**: Not persisted to database (only analysis results are)
- **API Keys**: Your OpenAI API key is used server-side only
- **Company Data**: Fetched from public investor relations websites

## Getting Help

If you're unsure what to do, just ask:

```
"What can you do?"
"Help"
"How does this work?"
```

The AI will explain its capabilities and guide you.

## Best Practices

1. **Start with a company name or ticker** for analysis
2. **Ask follow-up questions** to dive deeper into specific areas
3. **Be specific** in your questions for better responses
4. **Use natural language** - no special syntax needed
5. **Refresh if needed** to start a new session

## Limitations

- Analysis is based on publicly available earnings reports
- Some companies may have restricted or hard-to-parse reports
- Very recent earnings (< 24 hours) may not be indexed yet
- Follow-up questions are limited to information in the analyzed report
- Sessions expire after 60 minutes of inactivity
- RAG embeddings are generated on first analysis (one-time cost)
- Streaming requires SSE support in client

---

**Need more details?** Check out [ARCHITECTURE.md](./ARCHITECTURE.md) for technical documentation.




