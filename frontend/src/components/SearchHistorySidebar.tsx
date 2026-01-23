import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { History, X, ChevronLeft } from 'lucide-react'

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

interface SearchHistorySidebarProps {
  sessionId: string | null
  isOpen: boolean
  onClose: () => void
  onSearchClick: (entry: SearchHistoryEntry) => void
}

export function SearchHistorySidebar({ 
  sessionId, 
  isOpen, 
  onClose, 
  onSearchClick 
}: SearchHistorySidebarProps) {
  const [history, setHistory] = useState<SearchHistoryEntry[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (isOpen) {
      fetchHistory()
    }
  }, [isOpen, sessionId])

  const fetchHistory = async () => {
    setLoading(true)
    try {
      // Always fetch from all sessions to show complete search history
      // This ensures users can see all their previous searches, not just from the current session
      const sessionIdToUse = sessionId || 'all'
      const response = await fetch(`/api/v1/sessions/${sessionIdToUse}/history?all_sessions=true`)
      if (response.ok) {
        const data = await response.json()
        setHistory(data.searches || [])
      } else if (response.status === 404) {
        // Session doesn't exist (maybe was cleared), show empty history
        setHistory([])
      }
    } catch (error) {
      console.error('Error fetching search history:', error)
      setHistory([])
    } finally {
      setLoading(false)
    }
  }

  const formatCaption = (entry: SearchHistoryEntry): string => {
    // For analysis entries, show fiscal year and quarter if available
    if (entry.action === 'analysis') {
      if (entry.fiscal_year && entry.fiscal_quarter) {
        return `FY${entry.fiscal_year} Q${entry.fiscal_quarter}`
      } else if (entry.fiscal_year) {
        return `FY${entry.fiscal_year}`
      } else if (entry.ticker_symbol) {
        return entry.ticker_symbol
      }
    }
    
    // For chat entries or when no fiscal info, show truncated query
    if (entry.query) {
      const truncated = entry.query.length > 50 
        ? entry.query.substring(0, 50) + '...' 
        : entry.query
      return truncated
    }
    
    return 'Search'
  }

  const getTitle = (entry: SearchHistoryEntry): string => {
    // For analysis entries, prefer ticker symbol, then company name
    if (entry.action === 'analysis') {
      if (entry.ticker_symbol) {
        return entry.ticker_symbol
      } else if (entry.company_name) {
        return entry.company_name
      }
    }
    
    // For chat entries or when no ticker/company, show truncated query as title
    if (entry.query) {
      const truncated = entry.query.length > 30 
        ? entry.query.substring(0, 30) + '...' 
        : entry.query
      return truncated
    }
    
    return 'Chat'
  }

  const formatDate = (timestamp: string): string => {
    try {
      const date = new Date(timestamp)
      const now = new Date()
      const diffMs = now.getTime() - date.getTime()
      const diffMins = Math.floor(diffMs / 60000)
      const diffHours = Math.floor(diffMs / 3600000)
      const diffDays = Math.floor(diffMs / 86400000)

      if (diffMins < 1) return 'Just now'
      if (diffMins < 60) return `${diffMins}m ago`
      if (diffHours < 24) return `${diffHours}h ago`
      if (diffDays < 7) return `${diffDays}d ago`
      
      return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
    } catch {
      return ''
    }
  }

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/50 z-40"
          />
          
          {/* Sidebar */}
          <motion.div
            initial={{ x: '-100%' }}
            animate={{ x: 0 }}
            exit={{ x: '-100%' }}
            transition={{ type: 'spring', damping: 25, stiffness: 200 }}
            className="fixed left-0 top-0 bottom-0 w-80 lg:w-96 bg-slate-900 border-r border-slate-800 z-50 flex flex-col overflow-hidden shadow-2xl"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-slate-800">
              <div className="flex items-center gap-2">
                <History className="w-5 h-5 text-ocean-400" />
                <h2 className="text-lg font-semibold text-white">Search History</h2>
              </div>
              <button
                onClick={onClose}
                className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Content */}
            <div className="flex-1 overflow-y-auto p-4">
              {loading ? (
                <div className="flex items-center justify-center py-8">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-ocean-400"></div>
                </div>
              ) : history.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <History className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p>No search history yet</p>
                  <p className="text-sm mt-1">Your searches will appear here</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {history.map((entry) => (
                    <motion.button
                      key={entry.id}
                      onClick={() => {
                        onSearchClick(entry)
                        onClose()
                      }}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="w-full text-left p-3 rounded-lg bg-slate-800/50 hover:bg-slate-800 border border-slate-700/50 hover:border-ocean-500/50 transition-all duration-200 group"
                    >
                      <div className="flex items-start justify-between gap-2">
                        <div className="flex-1 min-w-0">
                          <div className="text-white font-medium text-sm truncate group-hover:text-ocean-300 transition-colors">
                            {getTitle(entry)}
                          </div>
                          <div className="text-slate-400 text-xs mt-1 truncate">
                            {formatCaption(entry)}
                          </div>
                          <div className="text-slate-500 text-xs mt-1">
                            {formatDate(entry.timestamp)}
                          </div>
                        </div>
                      </div>
                    </motion.button>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
