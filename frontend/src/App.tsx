import { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Send, TrendingUp, Loader2, Sparkles, BarChart3, FileText, Search, CheckCircle, AlertCircle } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

interface Message {
  id: string
  role: 'user' | 'assistant' | 'system'
  content: string
  timestamp: Date
}

interface AnalysisStage {
  id: string
  label: string
  status: 'pending' | 'active' | 'complete' | 'error'
  icon: React.ReactNode
}

const stages: AnalysisStage[] = [
  { id: 'analyzing', label: 'Analyzing Query', status: 'pending', icon: <Search className="w-4 h-4" /> },
  { id: 'searching', label: 'Finding IR Site', status: 'pending', icon: <TrendingUp className="w-4 h-4" /> },
  { id: 'parsing', label: 'Parsing Reports', status: 'pending', icon: <FileText className="w-4 h-4" /> },
  { id: 'summarizing', label: 'Generating Summary', status: 'pending', icon: <Sparkles className="w-4 h-4" /> },
]

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [currentStages, setCurrentStages] = useState<AnalysisStage[]>(stages)
  const [showWelcome, setShowWelcome] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const updateStage = (stageId: string, status: 'pending' | 'active' | 'complete' | 'error') => {
    setCurrentStages(prev => prev.map(s => 
      s.id === stageId ? { ...s, status } : s
    ))
  }

  const resetStages = () => {
    setCurrentStages(stages.map(s => ({ ...s, status: 'pending' as const })))
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
    setInput('')
    setIsLoading(true)
    setShowWelcome(false)
    resetStages()

    // Simulate stage progression
    const stageProgression = async () => {
      updateStage('analyzing', 'active')
      await new Promise(r => setTimeout(r, 1500))
      updateStage('analyzing', 'complete')
      
      updateStage('searching', 'active')
      await new Promise(r => setTimeout(r, 2000))
      updateStage('searching', 'complete')
      
      updateStage('parsing', 'active')
      await new Promise(r => setTimeout(r, 3000))
      updateStage('parsing', 'complete')
      
      updateStage('summarizing', 'active')
    }

    stageProgression()

    try {
      const response = await fetch('/api/v1/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company_query: input.trim() }),
      })

      const data = await response.json()
      
      updateStage('summarizing', 'complete')

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.summary || data.error || 'Unable to generate summary. Please try again.',
        timestamp: new Date(),
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, there was an error processing your request. Please ensure the backend server is running and try again.',
        timestamp: new Date(),
      }
      setMessages(prev => [...prev, errorMessage])
      
      // Update stages to show error
      setCurrentStages(prev => prev.map(s => 
        s.status === 'active' ? { ...s, status: 'error' as const } : s
      ))
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
                    What company would you like to analyze?
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

              {messages.map((message) => (
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
                    }`}
                  >
                    {message.role === 'assistant' ? (
                      <div className="markdown-content">
                        <ReactMarkdown>{message.content}</ReactMarkdown>
                      </div>
                    ) : (
                      <p className="text-lg">{message.content}</p>
                    )}
                  </div>
                </motion.div>
              ))}

              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="flex justify-start"
                >
                  <div className="bg-slate-800/80 rounded-2xl px-5 py-4 max-w-md">
                    {/* Stage indicators */}
                    <div className="space-y-3 mb-4">
                      {currentStages.map((stage, index) => (
                        <motion.div
                          key={stage.id}
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.1 }}
                          className={`flex items-center gap-3 ${
                            stage.status === 'active' ? 'text-ocean-400' :
                            stage.status === 'complete' ? 'text-emerald-400' :
                            stage.status === 'error' ? 'text-coral-500' :
                            'text-slate-500'
                          }`}
                        >
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
                          <span className="text-sm font-medium">{stage.label}</span>
                          {stage.status === 'active' && (
                            <div className="flex gap-1 ml-auto">
                              <span className="typing-dot w-1.5 h-1.5 bg-ocean-400 rounded-full" />
                              <span className="typing-dot w-1.5 h-1.5 bg-ocean-400 rounded-full" />
                              <span className="typing-dot w-1.5 h-1.5 bg-ocean-400 rounded-full" />
                            </div>
                          )}
                        </motion.div>
                      ))}
                    </div>
                    <p className="text-slate-400 text-sm">
                      Analyzing earnings reports...
                    </p>
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
                    Analyze
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

