import { useState, useEffect } from 'react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { TrendingUp, TrendingDown, DollarSign, PieChart } from 'lucide-react'
import { FinancialMetrics, MetricsHistory } from '../types/metrics'

interface MetricsDashboardProps {
  tickerSymbol: string
  currentMetrics?: FinancialMetrics
}

export function MetricsDashboard({ tickerSymbol, currentMetrics }: MetricsDashboardProps) {
  const [history, setHistory] = useState<MetricsHistory | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isPolling, setIsPolling] = useState(false)

  useEffect(() => {
    let pollInterval: number | null = null
    let isMounted = true
    
    const fetchAndPoll = async () => {
      try {
        setLoading(true)
        setError(null)
        console.log(`[MetricsDashboard] Fetching history for ${tickerSymbol}...`)
        const response = await fetch(`/api/v1/metrics/${tickerSymbol}/history`)
        if (!response.ok) {
          if (response.status === 500) {
            console.warn('Metrics history endpoint returned 500, showing empty state')
            setHistory({ ticker_symbol: tickerSymbol, history: [] })
            setLoading(false)
            return
          }
          throw new Error(`HTTP error! status: ${response.status}`)
        }
        const data = await response.json()
        setHistory(data)
        
        // Check if we have any chartable metrics (check for both null and undefined, but allow 0 as valid)
        const hasChartableData = data.history.some((m: FinancialMetrics) => 
          (m.revenue != null && !isNaN(m.revenue)) || 
          (m.eps != null && !isNaN(m.eps)) || 
          (m.gross_margin != null && !isNaN(m.gross_margin)) || 
          (m.operating_margin != null && !isNaN(m.operating_margin)) || 
          (m.free_cash_flow != null && !isNaN(m.free_cash_flow)) || 
          (m.operating_cash_flow != null && !isNaN(m.operating_cash_flow))
        )
        
        console.log('[MetricsDashboard] Initial fetch:', {
          historyLength: data.history.length,
          hasChartableData,
          sampleRecord: data.history[0],
          chartableFields: data.history.length > 0 ? {
            revenue: data.history[0].revenue,
            eps: data.history[0].eps,
            gross_margin: data.history[0].gross_margin,
            operating_margin: data.history[0].operating_margin,
            free_cash_flow: data.history[0].free_cash_flow,
            operating_cash_flow: data.history[0].operating_cash_flow
          } : null
        })
        
        // If we have data with chartable metrics, stop loading
        if (hasChartableData) {
          setLoading(false)
          setIsPolling(false)
          console.log('[MetricsDashboard] Found chartable data on initial fetch, stopping loading')
        } else {
          // No chartable data yet - start polling (scraping might still be in progress)
          // Even if we have history records, they might not have chartable data yet
          setLoading(false)
          setIsPolling(true)
          console.log('[MetricsDashboard] No chartable data found, starting polling...')
          
          let pollCount = 0
          const maxPolls = 20 // Poll for up to 20 times (100 seconds total)
          
          pollInterval = window.setInterval(async () => {
            pollCount++
            
            if (pollCount > maxPolls) {
              if (pollInterval) {
                window.clearInterval(pollInterval)
                pollInterval = null
              }
              setIsPolling(false)
              console.log('[MetricsDashboard] ⚠️ Stopped polling for metrics data after max attempts')
              return
            }

            try {
              const response = await fetch(`/api/v1/metrics/${tickerSymbol}/history`)
              if (response.ok) {
                const data = await response.json()
                // Check if we have chartable metrics (check for both null and undefined, but allow 0 as valid)
                const hasChartableData = data.history.some((m: FinancialMetrics) => 
                  (m.revenue != null && !isNaN(m.revenue)) || 
                  (m.eps != null && !isNaN(m.eps)) || 
                  (m.gross_margin != null && !isNaN(m.gross_margin)) || 
                  (m.operating_margin != null && !isNaN(m.operating_margin)) || 
                  (m.free_cash_flow != null && !isNaN(m.free_cash_flow)) || 
                  (m.operating_cash_flow != null && !isNaN(m.operating_cash_flow))
                )
                
                console.log(`[MetricsDashboard] Poll ${pollCount}:`, {
                  historyLength: data.history.length,
                  hasChartableData,
                  sampleRecord: data.history[0]
                })
                
                if (hasChartableData && isMounted) {
                  // We got chartable data! Stop polling and update
                  if (pollInterval) {
                    window.clearInterval(pollInterval)
                    pollInterval = null
                  }
                  setIsPolling(false)
                  setHistory(data)
                  setLoading(false)
                  console.log(`[MetricsDashboard] ✅ Found ${data.history.length} metrics records with chartable data after polling, stopping`)
                  console.log('[MetricsDashboard] Sample chartable data:', {
                    revenue: data.history[0]?.revenue,
                    eps: data.history[0]?.eps,
                    gross_margin: data.history[0]?.gross_margin,
                    operating_margin: data.history[0]?.operating_margin,
                    free_cash_flow: data.history[0]?.free_cash_flow,
                    operating_cash_flow: data.history[0]?.operating_cash_flow
                  })
                } else if (data.history.length > 0 && isMounted) {
                  // We have records but no chartable data yet - keep polling
                  setHistory(data)
                  console.log(`[MetricsDashboard] ⏳ Found ${data.history.length} metrics records but no chartable data yet (poll ${pollCount}/${maxPolls}), continuing...`)
                  console.log('[MetricsDashboard] Sample record (no chartable data):', {
                    revenue: data.history[0]?.revenue,
                    eps: data.history[0]?.eps,
                    gross_margin: data.history[0]?.gross_margin,
                    operating_margin: data.history[0]?.operating_margin,
                    free_cash_flow: data.history[0]?.free_cash_flow,
                    operating_cash_flow: data.history[0]?.operating_cash_flow
                  })
                } else if (isMounted) {
                  console.log(`[MetricsDashboard] ⏳ No records yet (poll ${pollCount}/${maxPolls}), continuing...`)
                }
              }
            } catch (error) {
              console.error('Error during polling:', error)
            }
          }, 5000) // Poll every 5 seconds
        }
      } catch (error) {
        console.error('Error fetching metrics history:', error)
        setError(error instanceof Error ? error.message : 'Failed to fetch metrics')
        setHistory({ ticker_symbol: tickerSymbol, history: [] })
        setLoading(false)
        setIsPolling(false)
      }
    }
    
    fetchAndPoll()
    
    // Cleanup polling on unmount
    return () => {
      isMounted = false
      if (pollInterval) {
        window.clearInterval(pollInterval)
        pollInterval = null
      }
      setIsPolling(false)
      console.log('[MetricsDashboard] Cleaned up polling interval')
    }
  }, [tickerSymbol])

  const fetchHistory = async (isRetry = false) => {
    try {
      if (!isRetry) {
        setLoading(true)
      }
      setError(null)
      const response = await fetch(`/api/v1/metrics/${tickerSymbol}/history`)
      if (!response.ok) {
        // Don't throw error for 500 - just show empty state
        if (response.status === 500) {
          console.warn('Metrics history endpoint returned 500, showing empty state')
          setHistory({ ticker_symbol: tickerSymbol, history: [] })
          setLoading(false)
          return
        }
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setHistory(data)
      
      // If we have data, stop polling
      if (data.history.length > 0) {
        setIsPolling(false)
      }
    } catch (error) {
      console.error('Error fetching metrics history:', error)
      setError(error instanceof Error ? error.message : 'Failed to fetch metrics')
      // Set empty history so UI can still show scrape button
      setHistory({ ticker_symbol: tickerSymbol, history: [] })
      setIsPolling(false)
    } finally {
      if (!isRetry) {
        setLoading(false)
      }
    }
  }


  if (loading || isPolling) {
    return (
      <div className="text-center py-8">
        <div className="text-slate-400 mb-2">
          {isPolling 
            ? 'Scraping financial data and loading metrics...' 
            : 'Loading metrics...'}
        </div>
        {isPolling && (
          <div className="text-slate-500 text-sm mt-2">
            This may take 30-60 seconds while we fetch data from financial statements
          </div>
        )}
        <div className="flex justify-center mt-4">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-ocean-400"></div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <div className="text-red-400 mb-2">Error loading metrics</div>
        <div className="text-slate-400 text-sm">{error}</div>
        <button
          onClick={() => fetchHistory(true)}
          className="mt-4 px-4 py-2 bg-ocean-500 hover:bg-ocean-600 text-white rounded-lg transition-colors"
        >
          Retry
        </button>
      </div>
    )
  }

  if (!history || !history.history.length) {
    return (
      <div className="text-center py-8">
        <div className="text-slate-400 mb-4">No metrics data available</div>
        <button
          onClick={async () => {
            try {
              setLoading(true)
              const response = await fetch(`/api/v1/metrics/${tickerSymbol}/scrape`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
              })
              if (response.ok) {
                await fetchHistory()
              } else {
                throw new Error('Failed to scrape metrics')
              }
            } catch (error) {
              setError(error instanceof Error ? error.message : 'Failed to scrape metrics')
            } finally {
              setLoading(false)
            }
          }}
          className="px-4 py-2 bg-ocean-500 hover:bg-ocean-600 text-white rounded-lg transition-colors"
        >
          Scrape Financial Data
        </button>
      </div>
    )
  }

  // Format data for charts
  // Convert values from raw dollars to millions for display
  const revenueData = history.history
    .filter(m => m.revenue !== null && m.revenue !== undefined)
    .reverse() // Oldest first
    .map(m => ({
      period: m.period,
      revenue: (m.revenue || 0) / 1000000, // Convert to millions
      qoq: m.revenue_qoq_change,
      yoy: m.revenue_yoy_change,
    }))

  const marginData = history.history
    .filter(m => m.gross_margin !== null || m.operating_margin !== null)
    .reverse()
    .map(m => ({
      period: m.period,
      gross: m.gross_margin,
      operating: m.operating_margin,
      net: m.net_margin,
    }))

  const epsData = history.history
    .filter(m => m.eps !== null && m.eps !== undefined)
    .reverse()
    .map(m => ({
      period: m.period,
      eps: m.eps,
      estimate: m.eps_estimate,
      actual: m.eps_actual,
    }))

  const cashFlowData = history.history
    .filter(m => m.free_cash_flow !== null || m.operating_cash_flow !== null)
    .reverse()
    .map(m => ({
      period: m.period,
      // Swap: fcf shows operating_cash_flow data, ocf shows free_cash_flow data
      // (The data appears to be stored backwards in the database)
      fcf: (m.operating_cash_flow || 0) / 1000000, // Convert to millions - showing Operating Cash Flow
      ocf: (m.free_cash_flow || 0) / 1000000, // Convert to millions - showing Free Cash Flow
    }))

  // Check if we have any chartable data
  const hasChartData = revenueData.length > 0 || marginData.length > 0 || epsData.length > 0 || cashFlowData.length > 0

  // Debug logging
  console.log('[MetricsDashboard]', {
    tickerSymbol,
    historyLength: history.history.length,
    hasChartData,
    revenueDataLength: revenueData.length,
    marginDataLength: marginData.length,
    epsDataLength: epsData.length,
    cashFlowDataLength: cashFlowData.length,
    sampleHistory: history.history.slice(0, 2),
  })

  return (
    <div className="w-full space-y-6">
      <div className="border-t border-slate-700/50 pt-4 mt-4">
        <h3 className="text-lg font-semibold text-white mb-4">Financial Metrics Dashboard</h3>
        
        {/* Show message if we have history records but no chartable data */}
        {!hasChartData && history.history.length > 0 && (
          <div className="text-center py-8">
            <div className="text-slate-400 mb-4">
              Found {history.history.length} period(s) but no metric values available.
              <br />
              <span className="text-sm">The data may still be loading or metrics may not be available for this company.</span>
            </div>
            <button
              onClick={async () => {
                try {
                  setLoading(true)
                  const response = await fetch(`/api/v1/metrics/${tickerSymbol}/scrape`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                  })
                  if (response.ok) {
                    await fetchHistory()
                  } else {
                    throw new Error('Failed to scrape metrics')
                  }
                } catch (error) {
                  setError(error instanceof Error ? error.message : 'Failed to scrape metrics')
                } finally {
                  setLoading(false)
                }
              }}
              className="px-4 py-2 bg-ocean-500 hover:bg-ocean-600 text-white rounded-lg transition-colors"
            >
              Re-scrape Financial Data
            </button>
          </div>
        )}
      {/* Current Metrics Cards */}
      {currentMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <MetricCard
            label="Revenue"
            value={currentMetrics.revenue ? `$${formatNumber(currentMetrics.revenue)}M` : 'N/A'}
            change={currentMetrics.revenue_yoy_change}
            icon={<DollarSign className="w-5 h-5" />}
          />
          <MetricCard
            label="EPS"
            value={currentMetrics.eps ? `$${currentMetrics.eps.toFixed(2)}` : 'N/A'}
            change={currentMetrics.eps_beat_miss ? currentMetrics.eps_beat_miss * 100 : undefined}
            icon={<TrendingUp className="w-5 h-5" />}
          />
          <MetricCard
            label="Gross Margin"
            value={currentMetrics.gross_margin ? `${currentMetrics.gross_margin.toFixed(1)}%` : 'N/A'}
            icon={<PieChart className="w-5 h-5" />}
          />
        </div>
      )}

      {/* Revenue Trend Chart */}
      {revenueData.length > 0 && (
        <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
          <h3 className="text-lg font-semibold text-white mb-4">Revenue Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={revenueData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="period" stroke="#94a3b8" />
              <YAxis 
                stroke="#94a3b8"
                tickFormatter={(value) => `$${(value / 1000).toFixed(value >= 1000 ? 1 : 0)}${value >= 1000 ? 'B' : 'M'}`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#f1f5f9',
                }}
                formatter={(value: any): any => {
                  const numValue = typeof value === 'number' ? value : parseFloat(value) || 0
                  if (numValue === 0) return ['N/A', 'Revenue']
                  return [`$${(numValue / 1000).toFixed(numValue >= 1000 ? 1 : 0)}${numValue >= 1000 ? 'B' : 'M'}`, 'Revenue']
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="revenue" stroke="#0ea5e9" strokeWidth={2} name="Revenue ($M)" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Margins Chart */}
      {marginData.length > 0 && (
        <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
          <h3 className="text-lg font-semibold text-white mb-4">Profitability Margins</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={marginData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="period" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#f1f5f9',
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="gross" stroke="#10b981" strokeWidth={2} name="Gross Margin %" />
              <Line type="monotone" dataKey="operating" stroke="#3b82f6" strokeWidth={2} name="Operating Margin %" />
              <Line type="monotone" dataKey="net" stroke="#8b5cf6" strokeWidth={2} name="Net Margin %" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* EPS Chart with Estimates */}
      {epsData.length > 0 && (
        <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
          <h3 className="text-lg font-semibold text-white mb-4">EPS Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={epsData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="period" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#f1f5f9',
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="eps" stroke="#10b981" strokeWidth={2} name="EPS" />
              {epsData.some(d => d.estimate !== null && d.estimate !== undefined) && (
                <Line type="monotone" dataKey="estimate" stroke="#f59e0b" strokeWidth={2} strokeDasharray="5 5" name="Estimated EPS" />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Cash Flow Chart */}
      {cashFlowData.length > 0 && (
        <div className="bg-slate-800/50 rounded-xl p-6 border border-slate-700/50">
          <h3 className="text-lg font-semibold text-white mb-4">Cash Flow</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={cashFlowData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#475569" />
              <XAxis dataKey="period" stroke="#94a3b8" />
              <YAxis 
                stroke="#94a3b8"
                tickFormatter={(value) => `$${(value / 1000).toFixed(value >= 1000 ? 1 : 0)}${value >= 1000 ? 'B' : 'M'}`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '8px',
                  color: '#f1f5f9',
                  padding: '12px',
                  boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
                }}
                content={({ active, payload }: any) => {
                  if (!active || !payload || payload.length === 0) return null
                  
                  const period = payload[0]?.payload?.period || ''
                  const items = payload.map((item: any) => {
                    const dataKey = item.dataKey || ''
                    const value = item.value
                    const numValue = typeof value === 'number' ? value : parseFloat(value) || 0
                    const color = item.color || '#94a3b8'
                    
                    // Determine label based on dataKey
                    // Note: fcf shows operating_cash_flow data, ocf shows free_cash_flow data (swapped)
                    let label: string
                    if (dataKey === 'fcf') {
                      label = 'Operating Cash Flow'  // fcf displays operating cash flow data
                    } else if (dataKey === 'ocf') {
                      label = 'Free Cash Flow'  // ocf displays free cash flow data
                    } else {
                      label = 'Cash Flow'
                    }
                    
                    // Format value: values are already in millions (after /1000000 conversion)
                    // Format appropriately - show billions if >= 1000M, otherwise show millions
                    let formattedValue: string
                    if (numValue === 0 || isNaN(numValue)) {
                      formattedValue = 'N/A'
                    } else if (numValue >= 1000) {
                      // Convert to billions if >= 1000M (e.g., 1500M -> $1.5B)
                      const billions = numValue / 1000
                      formattedValue = `$${billions.toFixed(billions >= 10 ? 0 : 1)}B`
                    } else {
                      // Show in millions (e.g., 503M -> $503M, 26.5M -> $27M)
                      formattedValue = `$${Math.round(numValue)}M`
                    }
                    
                    return (
                      <div key={dataKey} style={{ color, marginTop: '4px' }}>
                        {label}: {formattedValue}
                      </div>
                    )
                  })
                  
                  return (
                    <div style={{ 
                      backgroundColor: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: '8px',
                      padding: '12px',
                      boxShadow: '0 4px 6px rgba(0, 0, 0, 0.3)',
                      minWidth: '150px',
                    }}>
                      <div style={{ marginBottom: '8px', fontWeight: 'bold', fontSize: '14px', color: '#f1f5f9' }}>{period}</div>
                      {items}
                    </div>
                  )
                }}
              />
              <Legend />
              <Bar dataKey="fcf" fill="#10b981" name="Operating Cash Flow ($M)" />
              <Bar dataKey="ocf" fill="#3b82f6" name="Free Cash Flow ($M)" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      </div>
    </div>
  )
}

function MetricCard({ label, value, change, icon }: {
  label: string
  value: string
  change?: number
  icon: React.ReactNode
}) {
  const isPositive = change !== undefined && change > 0
  const isNegative = change !== undefined && change < 0

  return (
    <div className="bg-slate-800/50 rounded-xl p-4 border border-slate-700/50">
      <div className="flex items-center justify-between mb-2">
        <span className="text-slate-400 text-sm">{label}</span>
        <div className="text-ocean-400">{icon}</div>
      </div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      {change !== undefined && (
        <div className={`text-sm flex items-center gap-1 ${
          isPositive ? 'text-emerald-400' : isNegative ? 'text-red-400' : 'text-slate-400'
        }`}>
          {isPositive ? <TrendingUp className="w-4 h-4" /> : isNegative ? <TrendingDown className="w-4 h-4" /> : null}
          {change > 0 ? '+' : ''}{change.toFixed(1)}%
        </div>
      )}
    </div>
  )
}

function formatNumber(num: number): string {
  // num is expected to be in raw dollars, convert to millions
  const millions = num / 1000000
  if (millions >= 1000) {
    return (millions / 1000).toFixed(1) + 'B'
  }
  return millions.toFixed(0)
}
