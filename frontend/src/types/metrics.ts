export interface FinancialMetrics {
  ticker_symbol: string
  company_name: string
  fiscal_year: string
  fiscal_quarter: string
  period: string
  revenue?: number
  revenue_qoq_change?: number
  revenue_yoy_change?: number
  eps?: number
  eps_actual?: number
  eps_estimate?: number
  eps_beat_miss?: number
  net_income?: number
  gross_margin?: number
  operating_margin?: number
  net_margin?: number
  free_cash_flow?: number
  operating_cash_flow?: number
  total_assets?: number
  total_liabilities?: number
  total_equity?: number
  current_assets?: number
  current_liabilities?: number
  revenue_guidance?: number
  eps_guidance?: number
  segment_data?: Record<string, number>
  report_date?: string
}

export interface MetricsHistory {
  ticker_symbol: string
  history: FinancialMetrics[]
}
