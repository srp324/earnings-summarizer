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

  useEffect(() => {
    fetchHistory()
  }, [tickerSymbol])

  const fetchHistory = async () => {
    try {
      setLoading(true)
      setError(null)
      const response = await fetch(`/api/v1/metrics/${tickerSymbol}/history`)
      if (!response.ok) {
        // Don't throw error for 500 - just show empty state
        if (response.status === 500) {
          console.warn('Metrics history endpoint returned 500, showing empty state')
          setHistory({ ticker_symbol: tickerSymbol, history: [] })
          return
        }
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      const data = await response.json()
      setHistory(data)
    } catch (error) {
      console.error('Error fetching metrics history:', error)
      setError(error instanceof Error ? error.message : 'Failed to fetch metrics')
      // Set empty history so UI can still show scrape button
      setHistory({ ticker_symbol: tickerSymbol, history: [] })
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return <div className="text-center py-8 text-slate-400">Loading metrics...</div>
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <div className="text-red-400 mb-2">Error loading metrics</div>
        <div className="text-slate-400 text-sm">{error}</div>
        <button
          onClick={fetchHistory}
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
  const revenueData = history.history
    .filter(m => m.revenue !== null && m.revenue !== undefined)
    .reverse() // Oldest first
    .map(m => ({
      period: m.period,
      revenue: m.revenue,
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
      fcf: m.free_cash_flow,
      ocf: m.operating_cash_flow,
    }))

  return (
    <div className="w-full space-y-6">
      <div className="border-t border-slate-700/50 pt-4 mt-4">
        <h3 className="text-lg font-semibold text-white mb-4">Financial Metrics Dashboard</h3>
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
              <Bar dataKey="fcf" fill="#10b981" name="Free Cash Flow ($M)" />
              <Bar dataKey="ocf" fill="#3b82f6" name="Operating Cash Flow ($M)" />
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
  if (num >= 1000) {
    return (num / 1000).toFixed(1) + 'B'
  }
  return num.toFixed(0)
}
