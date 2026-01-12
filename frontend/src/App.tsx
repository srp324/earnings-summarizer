import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, TrendingUp, Loader2, Sparkles, BarChart3, Search, CheckCircle, AlertCircle, ChevronDown, ChevronUp } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { MetricsDashboard } from './components/MetricsDashboard'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  stages?: AnalysisStage[] // Optional stage flow data for analysis messages
  tickerSymbol?: string // Ticker symbol for financial dashboard
  companyName?: string // Company name for financial dashboard
}

interface AnalysisStage {
  id: string
  label: string
  status: 'pending' | 'active' | 'complete' | 'error'
  icon: React.ReactNode
  reasoning?: string
}

const stages: AnalysisStage[] = [
  { id: 'analyzing', label: 'Analyzing Query', status: 'pending', icon: <Search className="w-4 h-4" /> },
  { id: 'searching', label: 'Retrieving Reports', status: 'pending', icon: <TrendingUp className="w-4 h-4" /> },
  { id: 'summarizing', label: 'Generating Summary', status: 'pending', icon: <Sparkles className="w-4 h-4" /> },
]

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentStages, setCurrentStages] = useState<AnalysisStage[]>(stages)
  const [showWelcome, setShowWelcome] = useState(true)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [expandedStages, setExpandedStages] = useState<Set<string>>(new Set())
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const stagesRef = useRef<AnalysisStage[]>(stages)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const resetStages = () => {
    const reset = stages.map(s => ({ ...s, status: 'pending' as const, reasoning: undefined }))
    stagesRef.current = reset
    setCurrentStages(reset)
    setExpandedStages(new Set())
  }
  
  const toggleStageReasoning = (stageId: string) => {
    setExpandedStages(prev => {
      const newSet = new Set(prev)
      if (newSet.has(stageId)) {
        newSet.delete(stageId)
      } else {
        newSet.add(stageId)
      }
      return newSet
    })
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim(),
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    const currentInput = input.trim()
    setInput('')
    setIsLoading(true)
    setShowWelcome(false)

    try {
      // Use the streaming chat endpoint for real-time updates
      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          message: currentInput,
          session_id: sessionId 
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      // Check if response is streaming (SSE)
      const contentType = response.headers.get('content-type')
      if (contentType?.includes('text/event-stream')) {
        // Handle Server-Sent Events
        const reader = response.body?.getReader()
        const decoder = new TextDecoder()
        
        if (!reader) {
          throw new Error('Response body is not readable')
        }

        let buffer = ''
        let assistantContent = ''
        let receivedSessionId = sessionId
        let actionTaken = 'chat'
        // Hold pending metrics dashboard info so we can render it AFTER the summary
        let pendingMetricsTicker: string | undefined
        let pendingMetricsCompany: string | undefined

        while (true) {
          const { done, value } = await reader.read()
          
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n\n')
          buffer = lines.pop() || '' // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                console.log('[SSE Event]', data.type, data)  // Log all SSE events
                
                // Handle different event types
                if (data.type === 'status') {
                  // Status update - show thinking indicator
                  if (data.stage === 'thinking') {
                    setIsAnalyzing(false)
                  }
                } else if (data.type === 'reasoning') {
                  // Reasoning event - show AI reasoning as a visible step
                  // Also mark the stage as active when reasoning is received
                  const stageId = data.stage_id
                  const reasoning = data.reasoning

                  console.log(`[Reasoning Event] ${stageId}:`, {
                    stage_label: data.stage_label,
                    node: data.node,
                    reasoning_length: reasoning ? reasoning.length : 0,
                    reasoning_preview: reasoning ? reasoning.substring(0, 100) : null
                  })
                  
                  // Mark as analyzing when reasoning is received (stage is active)
                  setIsAnalyzing(true)
                  
                  // Update the stage with reasoning and mark as active
                  if (reasoning && reasoning.trim()) {
                    setCurrentStages(prev => {
                      const updated = prev.map(s => {
                        if (s.id === stageId) {
                          const hasNewReasoning = !s.reasoning
                          // Auto-expand reasoning when it first appears
                          if (hasNewReasoning) {
                            setExpandedStages(prevExpanded => new Set([...prevExpanded, stageId]))
                          }
                          return {
                            ...s,
                            label: data.stage_label || s.label,
                            status: 'active' as const, // Mark as active when reasoning is received
                            reasoning: reasoning
                          }
                        }
                        return s
                      })
                      stagesRef.current = updated
                      return updated
                    })
                  }
                } else if (data.type === 'stage_update') {
                  // Stage update - update UI in real-time
                  setIsAnalyzing(true)
                  const stageId = data.stage_id
                  const status: 'pending' | 'active' | 'complete' | 'error' = 
                    data.status === 'active' ? 'active' : 
                    data.status === 'complete' ? 'complete' : 
                    data.status === 'error' ? 'error' : 'pending'
                  const reasoning = data.reasoning  // Capture reasoning from backend
                  
                  // Debug logging - log ALL stage updates
                  console.log(`[Stage Update] ${stageId}: status=${status}`, {
                    stage_label: data.stage_label,
                    node: data.node,
                    has_reasoning: !!reasoning,
                    reasoning_length: reasoning ? reasoning.length : 0,
                    reasoning_preview: reasoning ? reasoning.substring(0, 100) : null,
                    full_data: data
                  })
                  
                  // Update the specific stage with label and reasoning if provided by backend
                  setCurrentStages(prev => {
                    const updated = prev.map((s) => {
                      if (s.id === stageId) {
                        // Update the target stage
                        const hasNewReasoning = reasoning !== undefined && reasoning !== null && reasoning.trim().length > 0
                        const newReasoning = hasNewReasoning ? reasoning : s.reasoning
                        
                        // Auto-expand reasoning when it first appears
                        if (hasNewReasoning && !s.reasoning) {
                          setExpandedStages(prevExpanded => new Set([...prevExpanded, stageId]))
                        }
                        
                        return {
                          ...s, 
                          label: data.stage_label || s.label, 
                          status,
                          reasoning: newReasoning
                        }
                      }
                      
                      // Leave other stages unchanged - backend will explicitly send 'complete' status
                      // for previous stages when transitioning to new stages
                      return s
                    })
                    stagesRef.current = updated
                    return updated
                  })
                } else if (data.type === 'complete') {
                  // Final result received
                  assistantContent = data.message || assistantContent
                  receivedSessionId = data.session_id || receivedSessionId
                  actionTaken = data.action_taken || 'chat'
                  
                  // If this was an analysis, create a message with the stage flow before hiding it
                  if (actionTaken === 'analysis_triggered') {
                    // Capture current stages from ref
                    const currentStagesSnapshot = stagesRef.current
                    
                    // Create completed stages for the message
                    const completedStages: AnalysisStage[] = currentStagesSnapshot.map(s => ({
                      ...s,
                      status: 'complete' as const,
                      icon: s.icon // Keep icon reference
                    }))
                    
                    // Create a message with the stage flow
                    const stageFlowMessage: Message = {
                      id: `stage-flow-${Date.now()}`,
                      role: 'assistant',
                      content: '', // Empty content - we'll render stages instead
                      timestamp: new Date(),
                      stages: completedStages
                    }
                    
                    // Add the stage flow message
                    setMessages(prevMessages => [...prevMessages, stageFlowMessage])
                    
                    // Mark stages as complete
                    setCurrentStages(completedStages)
                    stagesRef.current = completedStages
                    setIsAnalyzing(false)
                  }
                } else if (data.type === 'metrics_dashboard') {
                  // Remember metrics dashboard data; we'll render it after the summary message
                  pendingMetricsTicker = data.ticker_symbol
                  pendingMetricsCompany = data.company_name
                } else if (data.type === 'error') {
                  // Error occurred
                  assistantContent = data.message || data.error || 'An error occurred'
                  
                  // Update stages to show error
                  if (isAnalyzing) {
                    setCurrentStages(prev => {
                      const updated = prev.map(s => 
                        s.status === 'active' ? { ...s, status: 'error' as const } : s
                      )
                      stagesRef.current = updated
                      return updated
                    })
                    setIsAnalyzing(false)
                  }
                }
              } catch (parseError) {
                console.error('Error parsing SSE data:', parseError)
              }
            }
          }
        }

        // Store session ID
        if (receivedSessionId) {
          setSessionId(receivedSessionId)
        }

        // Add assistant message (summary)
        if (assistantContent) {
          const summaryMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: assistantContent,
            timestamp: new Date(),
          }
          
          setMessages(prev => {
            const updated = [...prev, summaryMessage]
            // Immediately follow summary with metrics dashboard, if available
            if (pendingMetricsTicker) {
              updated.push({
                id: `metrics-${Date.now() + 2}`,
                role: 'assistant',
                content: '',
                timestamp: new Date(),
                tickerSymbol: pendingMetricsTicker,
                companyName: pendingMetricsCompany,
              })
            }
            return updated
          })
        }
      } else {
        // Fallback to non-streaming endpoint
        const data = await response.json()
        
        if (data.session_id) {
          setSessionId(data.session_id)
        }

        if (data.action_taken === 'analysis_triggered') {
          setIsAnalyzing(true)
          resetStages()
        }

        const assistantContent = data.message || data.error || 'Unable to generate response. Please try again.'
        
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: 'assistant',
          content: assistantContent,
          timestamp: new Date(),
        }

        setMessages(prev => [...prev, assistantMessage])
        
        if (data.action_taken === 'analysis_triggered') {
          setIsAnalyzing(false)
        }
      }
    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please ensure the backend server is running and try again.',
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errorMessage])
      
      // Update stages to show error if we were analyzing
      if (isAnalyzing) {
        setCurrentStages(prev => {
          const updated = prev.map(s => 
            s.status === 'active' ? { ...s, status: 'error' as const } : s
          )
          stagesRef.current = updated
          return updated
        })
        setIsAnalyzing(false)
      }
    } finally {
      setIsLoading(false)
    }
  }

  const exampleQueries = [
    'Apple',
    'NVDA',
    'Microsoft',
    'Tesla',
  ]

  return (
    <div className="min-h-screen bg-slate-950 relative overflow-hidden">
      {/* Animated background */}
      <div className="fixed inset-0 pointer-events-none">
        <div className="glow-orb w-96 h-96 bg-ocean-600 top-[-10%] left-[-10%]" />
        <div className="glow-orb w-80 h-80 bg-coral-500 bottom-[-5%] right-[-5%]" style={{ animationDelay: '-3s' }} />
        <div className="glow-orb w-64 h-64 bg-ocean-400 top-[40%] right-[20%]" style={{ animationDelay: '-5s' }} />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto px-4 py-8 min-h-screen flex flex-col">
        {/* Header */}
        <motion.header 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-8"
        >
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="p-3 rounded-2xl bg-gradient-to-br from-ocean-500 to-ocean-700 shadow-lg shadow-ocean-500/25">
              <BarChart3 className="w-8 h-8 text-white" />
            </div>
            <h1 className="font-display text-4xl font-bold gradient-text">
              Earnings Summarizer
            </h1>
          </div>
          <p className="text-slate-400 text-lg">
            AI-powered analysis of stock earnings reports
          </p>
        </motion.header>

        {/* Main chat area */}
        <div className="flex-1 flex flex-col glass rounded-3xl overflow-hidden shadow-2xl shadow-black/50">
          {/* Messages container */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <AnimatePresence mode="popLayout">
              {showWelcome && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="flex flex-col items-center justify-center py-12"
                >
                  <div className="p-4 rounded-full bg-ocean-500/10 mb-6">
                    <Sparkles className="w-12 h-12 text-ocean-400" />
                  </div>
                  <h2 className="font-display text-2xl font-semibold text-white mb-3">
                    What company reports would you like to analyze?
                  </h2>
                  <p className="text-slate-400 mb-8 text-center max-w-md">
                    Enter a company name or stock ticker symbol, and I'll fetch and summarize their latest earnings report.
                  </p>
                  <div className="flex flex-wrap gap-3 justify-center">
                    {exampleQueries.map((query) => (
                      <button
                        key={query}
                        onClick={() => setInput(query)}
                        className="px-4 py-2 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 
                                   border border-slate-700/50 hover:border-ocean-500/50
                                   text-slate-300 hover:text-white transition-all duration-300
                                   text-sm font-medium"
                      >
                        {query}
                      </button>
                    ))}
                  </div>
                </motion.div>
              )}

              {messages.map((message) => {
                // Check if this message has stage flow data
                const hasStageFlow = message.stages && message.stages.length > 0
                
                return (
                  <motion.div
                    key={message.id}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-2xl px-5 py-4 ${
                        message.role === 'user'
                          ? 'bg-gradient-to-br from-ocean-600 to-ocean-700 text-white'
                          : 'bg-slate-800/80 text-slate-100'
                      } ${
                        // Make metrics dashboard messages expand to full available width
                        message.role === 'assistant' && message.tickerSymbol ? 'w-full' : ''
                      }`}
                    >
                      {message.role === 'assistant' && hasStageFlow ? (
                        // Render stage flow message
                        <div className="space-y-3">
                          {message.stages!.map((stage, index) => {
                            const stageKey = `${message.id}-${stage.id}`
                            const isExpanded = expandedStages.has(stageKey)
                            const hasReasoning = stage.reasoning && stage.reasoning.trim().length > 0
                            
                            const toggleStageReasoning = () => {
                              setExpandedStages(prev => {
                                const newSet = new Set(prev)
                                if (newSet.has(stageKey)) {
                                  newSet.delete(stageKey)
                                } else {
                                  newSet.add(stageKey)
                                }
                                return newSet
                              })
                            }
                            
                            return (
                              <motion.div
                                key={stage.id}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className="space-y-2"
                              >
                                <div className={`flex items-center gap-3 ${
                                  stage.status === 'active' ? 'text-ocean-400' :
                                  stage.status === 'complete' ? 'text-emerald-400' :
                                  stage.status === 'error' ? 'text-coral-500' :
                                  'text-slate-500'
                                }`}>
                                  <div className="relative">
                                    {stage.status === 'active' ? (
                                      <Loader2 className="w-4 h-4 animate-spin" />
                                    ) : stage.status === 'complete' ? (
                                      <CheckCircle className="w-4 h-4" />
                                    ) : stage.status === 'error' ? (
                                      <AlertCircle className="w-4 h-4" />
                                    ) : (
                                      stage.icon
                                    )}
                                  </div>
                                  <span className="text-sm font-medium flex-1">{stage.label}</span>
                                  {hasReasoning ? (
                                    <button
                                      onClick={toggleStageReasoning}
                                      className={`ml-2 p-1.5 rounded transition-colors flex items-center gap-1 ${
                                        isExpanded 
                                          ? 'bg-slate-700/70 hover:bg-slate-600/70 text-ocean-300' 
                                          : 'hover:bg-slate-700/50 text-slate-400 hover:text-ocean-400'
                                      }`}
                                      title={isExpanded ? "Hide reasoning" : "Show AI reasoning"}
                                    >
                                      <span className="text-xs font-medium">
                                        {isExpanded ? 'Hide' : 'Show'} Reasoning
                                      </span>
                                      {isExpanded ? (
                                        <ChevronUp className="w-3.5 h-3.5" />
                                      ) : (
                                        <ChevronDown className="w-3.5 h-3.5" />
                                      )}
                                    </button>
                                  ) : null}
                                </div>
                                
                                {/* Reasoning section - expandable */}
                                {hasReasoning && isExpanded && (
                                  <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="ml-7 mt-2 p-3 rounded-lg bg-slate-900/50 border border-slate-700/50"
                                  >
                                    <div className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wide">
                                      AI Reasoning
                                    </div>
                                    <div className="text-sm text-slate-300 whitespace-pre-wrap break-words">
                                      {stage.reasoning}
                                    </div>
                                  </motion.div>
                                )}
                              </motion.div>
                            )
                          })}
                        </div>
                      ) : message.role === 'assistant' ? (
                        <div className="w-full">
                          {message.content ? (
                            <div className="markdown-content">
                              <ReactMarkdown>{message.content}</ReactMarkdown>
                            </div>
                          ) : null}
                          {message.tickerSymbol && (
                            <div className="mt-4 w-full">
                              <MetricsDashboard 
                                tickerSymbol={message.tickerSymbol}
                              />
                            </div>
                          )}
                        </div>
                      ) : (
                        <p className="text-lg">{message.content}</p>
                      )}
                    </div>
                  </motion.div>
                )
              })}

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-start"
                >
                  <div className="bg-slate-800/80 rounded-2xl px-5 py-4 max-w-md">
                    {isAnalyzing ? (
                      <>
                        {/* Stage indicators for analysis */}
                        <div className="space-y-3 mb-4">
                          {currentStages.map((stage, index) => {
                            const isExpanded = expandedStages.has(stage.id)
                            const hasReasoning = stage.reasoning && stage.reasoning.trim().length > 0
                            
                            return (
                              <motion.div
                                key={stage.id}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                transition={{ delay: index * 0.1 }}
                                className="space-y-2"
                              >
                                <div className={`flex items-center gap-3 ${
                                  stage.status === 'active' ? 'text-ocean-400' :
                                  stage.status === 'complete' ? 'text-emerald-400' :
                                  stage.status === 'error' ? 'text-coral-500' :
                                  'text-slate-500'
                                }`}>
                                  <div className="relative">
                                    {stage.status === 'active' ? (
                                      <Loader2 className="w-4 h-4 animate-spin" />
                                    ) : stage.status === 'complete' ? (
                                      <CheckCircle className="w-4 h-4" />
                                    ) : stage.status === 'error' ? (
                                      <AlertCircle className="w-4 h-4" />
                                    ) : (
                                      stage.icon
                                    )}
                                  </div>
                                  <span className="text-sm font-medium flex-1">{stage.label}</span>
                                  {hasReasoning ? (
                                    <button
                                      onClick={() => toggleStageReasoning(stage.id)}
                                      className={`ml-2 p-1.5 rounded transition-colors flex items-center gap-1 ${
                                        isExpanded 
                                          ? 'bg-slate-700/70 hover:bg-slate-600/70 text-ocean-300' 
                                          : 'hover:bg-slate-700/50 text-slate-400 hover:text-ocean-400'
                                      }`}
                                      title={isExpanded ? "Hide reasoning" : "Show AI reasoning"}
                                    >
                                      <span className="text-xs font-medium">
                                        {isExpanded ? 'Hide' : 'Show'} Reasoning
                                      </span>
                                      {isExpanded ? (
                                        <ChevronUp className="w-3.5 h-3.5" />
                                      ) : (
                                        <ChevronDown className="w-3.5 h-3.5" />
                                      )}
                                    </button>
                                  ) : (
                                    stage.status === 'active' && (
                                      <div className="flex gap-1 ml-auto">
                                        <span className="typing-dot w-1.5 h-1.5 bg-ocean-400 rounded-full" />
                                        <span className="typing-dot w-1.5 h-1.5 bg-ocean-400 rounded-full" />
                                        <span className="typing-dot w-1.5 h-1.5 bg-ocean-400 rounded-full" />
                                      </div>
                                    )
                                  )}
                                </div>
                                
                                {/* Reasoning section - expandable */}
                                {hasReasoning && isExpanded && (
                                  <motion.div
                                    initial={{ opacity: 0, height: 0 }}
                                    animate={{ opacity: 1, height: 'auto' }}
                                    exit={{ opacity: 0, height: 0 }}
                                    className="ml-7 mt-2 p-3 rounded-lg bg-slate-900/50 border border-slate-700/50"
                                  >
                                    <div className="text-xs font-semibold text-slate-400 mb-2 uppercase tracking-wide">
                                      AI Reasoning
                                    </div>
                                    <div className="text-sm text-slate-300 whitespace-pre-wrap break-words">
                                      {stage.reasoning}
                                    </div>
                                  </motion.div>
                                )}
                              </motion.div>
                            )
                          })}
                        </div>
                      </>
                    ) : (
                      /* Simple loading indicator for chat */
                      <div className="flex items-center gap-3 text-ocean-400">
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span className="text-sm font-medium">Thinking...</span>
                      </div>
                    )}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>

          {/* Input area */}
          <div className="p-4 border-t border-slate-800/50">
            <form onSubmit={handleSubmit} className="flex gap-3">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Enter a company name or ticker symbol..."
                disabled={isLoading}
                className="flex-1 bg-slate-800/50 border border-slate-700/50 rounded-xl px-5 py-3
                           text-white placeholder-slate-500 focus:outline-none focus:border-ocean-500/50
                           focus:ring-2 focus:ring-ocean-500/20 transition-all duration-300
                           disabled:opacity-50 disabled:cursor-not-allowed"
              />
              <motion.button
                type="submit"
                disabled={isLoading || !input.trim()}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="px-6 py-3 bg-gradient-to-r from-ocean-500 to-ocean-600 
                           hover:from-ocean-400 hover:to-ocean-500
                           text-white font-semibold rounded-xl shadow-lg shadow-ocean-500/25
                           disabled:opacity-50 disabled:cursor-not-allowed disabled:shadow-none
                           transition-all duration-300 flex items-center gap-2"
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    <Send className="w-5 h-5" />
                    Send
                  </>
                )}
              </motion.button>
            </form>
          </div>
        </div>

        {/* Footer */}
        <footer className="text-center mt-6 text-slate-500 text-sm">
          <p>Powered by LangChain & LangGraph • Built with React & FastAPI</p>
        </footer>
      </div>
    </div>
  )
}

export default App

