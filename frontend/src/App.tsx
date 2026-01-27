import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, TrendingUp, Loader2, Sparkles, BarChart3, Search, CheckCircle, AlertCircle, ChevronDown, ChevronUp, History } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { MetricsDashboard } from './components/MetricsDashboard'
import { SearchHistorySidebar } from './components/SearchHistorySidebar'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
  stages?: AnalysisStage[] // Optional stage flow data for analysis messages
  tickerSymbol?: string // Ticker symbol for financial dashboard
  companyName?: string // Company name for financial dashboard
  isMetricsLoading?: boolean // Flag to show loading state for metrics dashboard
}

interface SearchHistoryEntry {
  id: string
  timestamp: string
  ticker_symbol?: string
  company_name?: string
  fiscal_year?: string
  fiscal_quarter?: string
  query: string
  action: string
  message_count?: number
  messages?: Array<{
    role: string
    content: string
    timestamp?: string
  }>
  stage_reasoning?: Record<string, string>  // {stage_id: reasoning}
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
  const sessionIdRef = useRef<string | null>(null) // Keep ref in sync for synchronous access
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [expandedStages, setExpandedStages] = useState<Set<string>>(new Set())
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const stagesRef = useRef<AnalysisStage[]>(stages)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const abortControllerRef = useRef<AbortController | null>(null)
  const isCancelledRef = useRef<boolean>(false)
  
  // Keep ref in sync with state
  useEffect(() => {
    sessionIdRef.current = sessionId
  }, [sessionId])

  // Removed auto-scroll functionality - user controls scrolling manually

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

  const submitQuery = async (query: string) => {
    if (!query.trim() || isLoading) return

    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }

    // Reset cancellation flag
    isCancelledRef.current = false

    // Create new AbortController for this request
    const abortController = new AbortController()
    abortControllerRef.current = abortController

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: query.trim(),
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, userMessage])
    const currentInput = query.trim()
    setInput('')
    setIsLoading(true)
    setShowWelcome(false)
    // Reset stages for new query
    resetStages()

    try {
      // Use the streaming chat endpoint for real-time updates
      // Use ref value to ensure we have the latest session_id synchronously
      const currentSessionId = sessionIdRef.current || sessionId
      const payload = { 
        message: currentInput,
        session_id: currentSessionId 
      }
      console.log('[App] Sending request with payload:', payload, '(sessionId state:', sessionId, ', ref:', sessionIdRef.current, ')')
      
      const response = await fetch('/api/v1/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: abortController.signal,
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
        let metricsLoadingMessageId: string | undefined

        while (true) {
          // Check if request was cancelled
          if (isCancelledRef.current || abortController.signal.aborted) {
            console.log('[App] Request cancelled, stopping stream processing')
            reader.cancel()
            break
          }

          const { done, value } = await reader.read()
          
          if (done) break

          // Check again after read
          if (isCancelledRef.current || abortController.signal.aborted) {
            console.log('[App] Request cancelled after read, stopping stream processing')
            reader.cancel()
            break
          }

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n\n')
          buffer = lines.pop() || '' // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6))
                console.log('[SSE Event]', data.type, data)  // Log all SSE events
                
                // Check if cancelled before processing event
                if (isCancelledRef.current || abortController.signal.aborted) {
                  console.log('[App] Request cancelled, ignoring event:', data.type)
                  break
                }

                // Handle different event types
                if (data.type === 'status') {
                  // Status update - show thinking indicator
                  if (data.stage === 'thinking') {
                    if (!isCancelledRef.current) setIsAnalyzing(false)
                  }
                  // Capture session_id from status events (backend sends it in initial status event)
                  if (data.session_id && !isCancelledRef.current) {
                    console.log('[App] Received session_id from status event:', data.session_id)
                    receivedSessionId = data.session_id
                    // Update both ref (synchronous) and state (async) immediately
                    sessionIdRef.current = data.session_id
                    setSessionId(data.session_id)
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
                  if (!isCancelledRef.current) setIsAnalyzing(true)
                  
                  // Update the stage with reasoning and mark as active
                  if (reasoning && reasoning.trim() && !isCancelledRef.current) {
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
                  if (!isCancelledRef.current) setIsAnalyzing(true)
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
                  if (!isCancelledRef.current) {
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
                  }
                } else if (data.type === 'complete') {
                  // Final result received
                  if (isCancelledRef.current) {
                    console.log('[App] Request cancelled, ignoring complete event')
                    break
                  }
                  assistantContent = data.message || assistantContent
                  if (data.session_id && data.session_id !== receivedSessionId) {
                    receivedSessionId = data.session_id
                    // Update both ref (synchronous) and state (async) immediately
                    sessionIdRef.current = data.session_id
                    setSessionId(data.session_id)
                  }
                  actionTaken = data.action_taken || 'chat'
                  
                  // If this was an analysis, create a message with the stage flow before hiding it
                  if (actionTaken === 'analysis_triggered' && !isCancelledRef.current) {
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
                    
                    // Preserve expanded state when moving stages to message
                    // Convert stage IDs to message-based keys and keep them expanded
                    setExpandedStages(prev => {
                      const newSet = new Set(prev)
                      // Add all stages with reasoning to expanded set using message-based keys
                      completedStages.forEach(stage => {
                        if (stage.reasoning && stage.reasoning.trim().length > 0) {
                          const messageStageKey = `${stageFlowMessage.id}-${stage.id}`
                          newSet.add(messageStageKey)
                        }
                      })
                      return newSet
                    })
                    
                    // Add the stage flow message
                    setMessages(prevMessages => [...prevMessages, stageFlowMessage])
                    
                    // Mark stages as complete
                    setCurrentStages(completedStages)
                    stagesRef.current = completedStages
                    setIsAnalyzing(false)
                  }
                } else if (data.type === 'metrics_dashboard_loading') {
                  if (isCancelledRef.current) break
                  // Store loading message info - we'll add it after the summary
                  // DON'T overwrite assistantContent - that contains the summary!
                  metricsLoadingMessageId = `metrics-loading-${Date.now()}`
                  pendingMetricsTicker = data.ticker_symbol
                  pendingMetricsCompany = data.company_name
                  console.log('[App] Received metrics_dashboard_loading for', data.ticker_symbol, 'messageId:', metricsLoadingMessageId)
                } else if (data.type === 'metrics_dashboard') {
                  if (isCancelledRef.current) break
                  // Replace loading message with actual dashboard
                  pendingMetricsTicker = data.ticker_symbol
                  pendingMetricsCompany = data.company_name
                  console.log('[App] Received metrics_dashboard for', data.ticker_symbol, 'looking for messageId:', metricsLoadingMessageId)
                  
                  // Only replace if loading message already exists in messages
                  // Otherwise, wait for summary to be added first, then add dashboard after summary
                  setMessages(prev => {
                    if (metricsLoadingMessageId) {
                      const loadingIndex = prev.findIndex(m => m.id === metricsLoadingMessageId)
                      console.log('[App] Looking for loading message with id:', metricsLoadingMessageId, 'found at index:', loadingIndex)
                      if (loadingIndex !== -1) {
                        // Replace loading message with actual dashboard (clear isMetricsLoading flag)
                        const updated = [...prev]
                        updated[loadingIndex] = {
                          id: metricsLoadingMessageId,
                          role: 'assistant',
                          content: '',
                          timestamp: new Date(),
                          tickerSymbol: pendingMetricsTicker,
                          companyName: pendingMetricsCompany,
                          // Don't set isMetricsLoading at all (undefined = false)
                        }
                        console.log('[App] ✅ Replaced loading message with dashboard message for', pendingMetricsTicker)
                        return updated
                      } else {
                        // Loading message not added yet - don't add dashboard yet, wait for summary
                        // The summary addition logic will handle adding the dashboard after summary
                        console.log('[App] Loading message not found yet, will add dashboard after summary for', pendingMetricsTicker)
                      }
                    } else {
                      // No loading message ID - don't add dashboard yet, wait for summary
                      console.log('[App] No loading message ID, will add dashboard after summary for', pendingMetricsTicker)
                    }
                    return prev
                  })
                } else if (data.type === 'error') {
                  if (isCancelledRef.current) break
                  // Error occurred
                  assistantContent = data.message || data.error || 'An error occurred'
                  
                  // Capture session_id from error events if present
                  if (data.session_id) {
                    receivedSessionId = data.session_id
                    // Update both ref (synchronous) and state (async) immediately
                    sessionIdRef.current = data.session_id
                    setSessionId(data.session_id)
                  }
                  
                  // Update stages to show error
                  if (isAnalyzing && !isCancelledRef.current) {
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
                
                // Always try to capture session_id from any event that has it
                // Update if we have a session_id and either we don't have one yet, or we got a new one
                if (data.session_id) {
                  if (!receivedSessionId || data.session_id !== receivedSessionId) {
                    console.log('[App] Updating session_id from', receivedSessionId, 'to', data.session_id)
                    receivedSessionId = data.session_id
                    // Update both ref (synchronous) and state (async) immediately
                    sessionIdRef.current = data.session_id
                    setSessionId(data.session_id)
                  }
                }
              } catch (parseError) {
                console.error('Error parsing SSE data:', parseError)
              }
            }
          }
        }

        // Final check - ensure session ID is stored (should already be set above, but this is a fallback)
        if (!isCancelledRef.current) {
          if (receivedSessionId && receivedSessionId !== sessionIdRef.current) {
            console.log('[App] Final check - setting session_id state to:', receivedSessionId)
            sessionIdRef.current = receivedSessionId
            setSessionId(receivedSessionId)
          } else if (!receivedSessionId) {
            console.warn('[App] No session_id received from backend! receivedSessionId is:', receivedSessionId, 'current sessionId state:', sessionId, 'ref:', sessionIdRef.current)
          }
        }

        // Add assistant message (summary) - only if not cancelled
        if (assistantContent && !isCancelledRef.current) {
          const summaryMessage: Message = {
            id: (Date.now() + 1).toString(),
            role: 'assistant',
            content: assistantContent,
            timestamp: new Date(),
          }
          
          setMessages(prev => {
            const updated = [...prev, summaryMessage]
            // Add metrics dashboard after summary (only if we haven't already added it)
            if (pendingMetricsTicker && !prev.some(m => m.tickerSymbol === pendingMetricsTicker && m.id !== summaryMessage.id)) {
              // Check if metrics_dashboard event already came (which means data is ready)
              // If a dashboard message already exists (without isMetricsLoading), skip adding loading message
              const hasDashboardMessage = prev.some(m => m.tickerSymbol === pendingMetricsTicker && m.isMetricsLoading === undefined)
              
              console.log('[App] Adding metrics message after summary:', {
                pendingMetricsTicker,
                metricsLoadingMessageId,
                hasDashboardMessage,
                hasLoadingMessage: prev.some(m => m.id === metricsLoadingMessageId)
              })
              
              if (!hasDashboardMessage) {
                // Always add metrics message AFTER summary
                // Use the loading message ID if it exists, otherwise create a new one
                const messageId = metricsLoadingMessageId || `metrics-${Date.now() + 2}`
                
                // Check if metrics_dashboard event already came (by checking if we have pendingMetricsTicker but no loading message in array)
                // If metrics_dashboard came, the loading message would have been replaced, so we add dashboard
                // Otherwise, add loading message
                const loadingMessageExists = prev.some(m => m.id === metricsLoadingMessageId && m.isMetricsLoading === true)
                
                if (!loadingMessageExists && metricsLoadingMessageId) {
                  // Loading message doesn't exist - metrics_dashboard probably came, so add dashboard
                  updated.push({
                    id: messageId,
                    role: 'assistant',
                    content: '',
                    timestamp: new Date(),
                    tickerSymbol: pendingMetricsTicker,
                    companyName: pendingMetricsCompany,
                    // Don't set isMetricsLoading - this is the dashboard
                  })
                  console.log('[App] ✅ Added dashboard message after summary for', pendingMetricsTicker, '(metrics_dashboard already came)')
                } else if (metricsLoadingMessageId && !prev.some(m => m.id === metricsLoadingMessageId)) {
                  // Add loading message (metrics_dashboard hasn't come yet)
                  updated.push({
                    id: messageId,
                    role: 'assistant',
                    content: 'Loading financial metric charts...',
                    timestamp: new Date(),
                    tickerSymbol: pendingMetricsTicker,
                    companyName: pendingMetricsCompany,
                    isMetricsLoading: true
                  })
                  console.log('[App] ✅ Added loading message after summary for', pendingMetricsTicker)
                } else if (!metricsLoadingMessageId) {
                  // No loading message ID - add dashboard directly
                  updated.push({
                    id: messageId,
                    role: 'assistant',
                    content: '',
                    timestamp: new Date(),
                    tickerSymbol: pendingMetricsTicker,
                    companyName: pendingMetricsCompany,
                  })
                  console.log('[App] ✅ Added dashboard message directly after summary for', pendingMetricsTicker)
                } else {
                  console.log('[App] ⚠️ Message already exists for', pendingMetricsTicker)
                }
              } else {
                console.log('[App] ⚠️ Dashboard message already exists for', pendingMetricsTicker, 'skipping')
              }
            } else {
              console.log('[App] ⚠️ Not adding metrics message:', {
                hasPendingTicker: !!pendingMetricsTicker,
                alreadyExists: prev.some(m => m.tickerSymbol === pendingMetricsTicker && m.id !== summaryMessage.id)
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
      // Don't show error if request was cancelled
      if (isCancelledRef.current || (error instanceof Error && error.name === 'AbortError')) {
        console.log('[App] Request was cancelled')
        return
      }
      
      console.error('Chat error:', error)
      if (!isCancelledRef.current) {
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
      }
    } finally {
      if (!isCancelledRef.current) {
        setIsLoading(false)
      }
      // Clean up abort controller
      if (abortControllerRef.current === abortController) {
        abortControllerRef.current = null
      }
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await submitQuery(input)
  }

  const exampleQueries = [
    'Apple',
    'GOOGL',
    'Microsoft',
    'TSLA',
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
          <div className="flex items-center justify-center gap-3 mb-3 relative">
            <button
              onClick={() => setSidebarOpen(true)}
              className="absolute left-0 p-2 rounded-xl bg-slate-800/50 hover:bg-slate-700/50 border border-slate-700/50 hover:border-ocean-500/50 text-slate-400 hover:text-ocean-400 transition-all duration-300"
              title="Search History"
            >
              <History className="w-5 h-5" />
            </button>
            <div className="p-3 rounded-2xl bg-gradient-to-br from-ocean-500 to-ocean-700 shadow-lg shadow-ocean-500/25">
              <BarChart3 className="w-8 h-8 text-white" />
            </div>
            <button
              onClick={async () => {
                // If already on home page with no activity, do nothing (don't scroll)
                const isOnHomePage = messages.length === 0 && !isLoading && !isAnalyzing && showWelcome
                if (isOnHomePage) {
                  return
                }
                
                // Cancel any ongoing request first
                if (abortControllerRef.current) {
                  console.log('[App] Cancelling ongoing request')
                  isCancelledRef.current = true
                  abortControllerRef.current.abort()
                  abortControllerRef.current = null
                }
                
                // Stop any ongoing analysis/stages
                setIsLoading(false)
                setIsAnalyzing(false)
                
                // Save current conversation to search history if there are messages
                if (messages.length > 0 && sessionId && !isCancelledRef.current) {
                  try {
                    await fetch(`/api/v1/sessions/${sessionId}/save-conversation`, {
                      method: 'POST',
                      headers: { 'Content-Type': 'application/json' },
                    })
                    console.log('[App] Conversation saved to search history')
                  } catch (error) {
                    console.error('[App] Error saving conversation:', error)
                    // Continue anyway - don't block navigation
                  }
                }
                
                // Reset UI to home
                setMessages([])
                setShowWelcome(true)
                resetStages()
                // Reset session to start a completely new conversation
                // Setting to null ensures the next query creates a NEW session
                // This prevents overwriting previous session's search history
                setSessionId(null)
                sessionIdRef.current = null
                isCancelledRef.current = false // Reset cancellation flag
                window.scrollTo({ top: 0, behavior: 'smooth' })
              }}
              className="font-display text-4xl font-bold gradient-text cursor-pointer hover:opacity-80 transition-opacity duration-300"
              title="Return to home"
            >
              Earnings Summarizer
            </button>
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
                              {message.isMetricsLoading === true ? (
                                <div className="text-center py-8">
                                  <div className="text-slate-400 mb-2">
                                    {message.content || 'Loading financial metric charts...'}
                                  </div>
                                  <div className="text-slate-500 text-sm mt-2">
                                    This may take 30-60 seconds while we fetch data from financial statements
                                  </div>
                                  <div className="flex justify-center mt-4">
                                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ocean-400"></div>
                                  </div>
                                </div>
                              ) : (
                                <>
                                  {console.log('[App] Rendering MetricsDashboard for', message.tickerSymbol, 'isMetricsLoading:', message.isMetricsLoading)}
                                  <MetricsDashboard 
                                    tickerSymbol={message.tickerSymbol}
                                  />
                                </>
                              )}
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

      {/* Search History Sidebar */}
      <SearchHistorySidebar
        sessionId={sessionId}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSearchClick={async (entry: SearchHistoryEntry) => {
          // Restore conversation state from stored messages
          if (entry.messages && entry.messages.length > 0) {
            console.log('[Restore] Restoring entry:', {
              id: entry.id,
              action: entry.action,
              ticker_symbol: entry.ticker_symbol,
              message_count: entry.messages.length,
              messages: entry.messages.map(m => ({ role: m.role, content_length: m.content?.length || 0, content_preview: m.content?.substring(0, 50) }))
            })
            
            const restoredMessages: Message[] = []
            
            // Process messages and reconstruct full conversation state
            let messageIndex = 0
            let hasAddedStageFlow = false
            let summaryMessageAdded = false
            
            // First, find the summary message (last assistant message with content)
            let summaryMessage: { role: string; content: string; timestamp?: string } | null = null
            for (let i = entry.messages.length - 1; i >= 0; i--) {
              const msg = entry.messages[i]
              if (msg.role === 'assistant' && msg.content && msg.content.trim()) {
                summaryMessage = msg
                console.log('[Restore] Found summary message:', { content_length: msg.content.length, content_preview: msg.content.substring(0, 100) })
                break
              }
            }
            
            // Process all stored messages in order
            entry.messages.forEach((msg) => {
              const timestamp = msg.timestamp ? new Date(msg.timestamp) : new Date(entry.timestamp)
              
              // Add user message
              if (msg.role === 'user') {
                restoredMessages.push({
                  id: `${entry.id}-user-${messageIndex++}`,
                  role: 'user',
                  content: msg.content || '',
                  timestamp,
                })
                
                // For analysis entries, add stage flow message after user message
                if (entry.action === 'analysis' && !hasAddedStageFlow) {
                  // Restore stages with reasoning if available
                  const stageMessageId = `${entry.id}-stages`
                  const completedStages: AnalysisStage[] = stages.map(s => {
                    const stageWithReasoning: AnalysisStage = {
                      ...s,
                      status: 'complete' as const,
                      icon: s.icon,
                    }
                    // Add reasoning if it was stored for this stage
                    if (entry.stage_reasoning && entry.stage_reasoning[s.id]) {
                      stageWithReasoning.reasoning = entry.stage_reasoning[s.id]
                      // Auto-expand reasoning for stages that have it (using correct key format)
                      const stageKey = `${stageMessageId}-${s.id}`
                      setExpandedStages(prev => new Set([...prev, stageKey]))
                    }
                    return stageWithReasoning
                  })
                  
                  restoredMessages.push({
                    id: stageMessageId,
                    role: 'assistant',
                    content: '',
                    timestamp,
                    stages: completedStages
                  })
                  hasAddedStageFlow = true
                }
              }
              
              // Add assistant messages
              if (msg.role === 'assistant') {
                // Check if this is the summary message (last assistant message with content)
                // Use trim for comparison to handle whitespace differences
                const isSummaryMessage = summaryMessage && 
                  msg.content && msg.content.trim() !== '' &&
                  msg.content.trim() === summaryMessage.content.trim()
                
                if (isSummaryMessage) {
                  console.log('[Restore] Processing summary message')
                }
                
                const restoredMessage: Message = {
                  id: `${entry.id}-assistant-${messageIndex++}`,
                  role: 'assistant',
                  content: msg.content || '',
                  timestamp,
                }
                
                // For analysis entries with ticker, treat the summary message as text-only (no ticker),
                // and add a separate metrics dashboard message immediately after it.
                // This mirrors the live streaming behavior: one summary message, one metrics message.
                if (entry.action === 'analysis' && entry.ticker_symbol && isSummaryMessage && !summaryMessageAdded) {
                  console.log('[Restore] Adding summary message for analysis entry with ticker:', entry.ticker_symbol)
                  restoredMessages.push(restoredMessage)
                  summaryMessageAdded = true
                  
                  // Add metrics dashboard message immediately after summary.
                  // This will be the ONLY message that contains the tickerSymbol,
                  // so only one metrics dashboard is rendered on restore.
                  console.log('[Restore] Adding metrics dashboard message')
                  restoredMessages.push({
                    id: `${entry.id}-metrics`,
                    role: 'assistant',
                    content: '',
                    timestamp: new Date(timestamp.getTime() + 1000), // Slightly after summary
                    tickerSymbol: entry.ticker_symbol,
                    companyName: entry.company_name,
                  })
                } else {
                  // All other assistant messages (including non-summary or already processed summary)
                  // Only skip if this is the summary and we already added it
                  if (!(isSummaryMessage && summaryMessageAdded)) {
                    restoredMessages.push(restoredMessage)
                  }
                }
              }
            })
            
            // Restore messages to state
            setMessages(restoredMessages)
            setShowWelcome(false)
            
            // IMPORTANT: Don't change sessionId when restoring
            // The search history entry belongs to whatever session it was created in
            // We're just viewing/restoring the messages, not switching sessions
            // This ensures that when user resets and starts a new query, it creates a NEW session
            // and doesn't overwrite the previous session's search history
            
            // Don't auto-scroll - let user control their view position
          } else if (entry.query) {
            // Fallback: if no messages stored, restore by re-submitting query
            submitQuery(entry.query)
          }
        }}
      />
    </div>
  )
}

export default App

