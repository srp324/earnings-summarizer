"""Service for storing and retrieving financial metrics."""
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Optional, Dict, Any, Tuple
import json
import re
from datetime import datetime

from app.database import FinancialMetrics
import logging

logger = logging.getLogger(__name__)


# Mapping from scraped metric IDs to database fields
METRIC_MAPPING = {
    # Income statement metrics
    'revenue': 'revenue',
    'total-revenue': 'revenue',
    'net-revenue': 'revenue',
    'net-income': 'net_income',
    'gross-profit': None,  # Can calculate margin from this
    'operating-income': None,  # Can calculate margin from this
    'eps': 'eps',
    'earnings-per-share': 'eps',
    'diluted-eps': 'eps',
    
    # Balance sheet metrics
    'total-assets': 'total_assets',
    'total-liabilities': 'total_liabilities',
    'total-equity': 'total_equity',
    'current-assets': 'current_assets',
    'current-liabilities': 'current_liabilities',
    
    # Cash flow metrics
    'operating-cash-flow': 'operating_cash_flow',
    'free-cash-flow': 'free_cash_flow',
    'fcf': 'free_cash_flow',
}


def parse_period(period_str: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse period string into (fiscal_year, fiscal_quarter).

    Handles formats like:
    - 'Q1 2024'
    - '2024-Q1'
    - '2024 Q1'
    - '2025 (Q4) 09-27'
    - '2024 (Q2) 03-30'
    """
    if not period_str:
        return None, None

    # Skip non-quarter summary periods like LTM
    if "LTM" in period_str or "Last Twelve Months" in period_str:
        logger.info(f"Skipping non-quarter period: {period_str}")
        return None, None

    # Normalize whitespace
    s = period_str.strip()

    # Pattern: '2025 (Q4) 09-27' or '2024 (Q2)'
    match = re.search(r'(\d{4})\s*\(Q([1-4])\)', s, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    # Pattern: Q1 2024, Q2 2024, etc.
    match = re.search(r'Q([1-4])\s+(\d{4})', s, re.IGNORECASE)
    if match:
        return match.group(2), match.group(1)

    # Pattern: 2024-Q1
    match = re.search(r'(\d{4})-Q([1-4])', s, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    # Pattern: 2024 Q1
    match = re.search(r'(\d{4})\s+Q([1-4])', s, re.IGNORECASE)
    if match:
        return match.group(1), match.group(2)

    logger.warning(f"Could not parse period: {period_str}")
    return None, None


def normalize_metric_id(metric_id: str) -> str:
    """Normalize metric ID to lowercase with hyphens."""
    if not metric_id:
        return ''
    return metric_id.lower().replace('_', '-')


def extract_metric_value(data: Dict[str, Any], metric_ids: List[str]) -> Optional[float]:
    """
    Extract metric value from scraped data by trying multiple metric IDs.
    
    Args:
        data: Dictionary with metric_id -> {value, name}
        metric_ids: List of possible metric IDs to try
    
    Returns:
        First found value or None
    """
    for metric_id in metric_ids:
        normalized_id = normalize_metric_id(metric_id)
        if normalized_id in data:
            value = data[normalized_id].get('value')
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    pass
    return None


def calculate_margins(income_data: Dict[str, Any], revenue: Optional[float]) -> Dict[str, Optional[float]]:
    """Calculate margin percentages from income statement data."""
    margins = {
        'gross_margin': None,
        'operating_margin': None,
        'net_margin': None,
    }
    
    if not revenue or revenue == 0:
        return margins
    
    # Try to find gross profit
    gross_profit = extract_metric_value(income_data, [
        'gross-profit', 'gross_profit', 'grossprofit'
    ])
    if gross_profit:
        margins['gross_margin'] = (gross_profit / revenue) * 100
    
    # Try to find operating income
    operating_income = extract_metric_value(income_data, [
        'operating-income', 'operating_income', 'operatingincome', 'ebit'
    ])
    if operating_income:
        margins['operating_margin'] = (operating_income / revenue) * 100
    
    # Try to find net income
    net_income = extract_metric_value(income_data, [
        'net-income', 'net_income', 'netincome'
    ])
    if net_income:
        margins['net_margin'] = (net_income / revenue) * 100
    
    return margins


async def store_scraped_metrics(
    db: AsyncSession,
    ticker_symbol: str,
    company_name: str,
    scraped_data: Dict[str, Any],
    session_id: Optional[str] = None
) -> List[FinancialMetrics]:
    """
    Store scraped financial metrics in database.
    
    Args:
        db: Database session
        ticker_symbol: Stock ticker symbol
        company_name: Company name
        scraped_data: Dictionary with 'income_statement', 'balance_sheet', 'cash_flow' keys
        session_id: Optional session ID
    
    Returns:
        List of created/updated FinancialMetrics records
    """
    stored_metrics = []
    
    income_data = scraped_data.get('income_statement', {})
    balance_data = scraped_data.get('balance_sheet', {})
    cash_flow_data = scraped_data.get('cash_flow', {})
    
    # Get all unique periods from all statements
    all_periods = set()
    for statement_data in [income_data, balance_data, cash_flow_data]:
        all_periods.update(statement_data.keys())
    
    for period in all_periods:
        fiscal_year, fiscal_quarter = parse_period(period)
        if not fiscal_year or not fiscal_quarter:
            logger.warning(f"Skipping period {period} - could not parse")
            continue
        
        # Get data for this period from each statement
        period_income = income_data.get(period, {})
        period_balance = balance_data.get(period, {})
        period_cash_flow = cash_flow_data.get(period, {})
        
        # Extract revenue
        revenue = extract_metric_value(period_income, [
            'revenue', 'total-revenue', 'net-revenue', 'totalrevenue', 'netrevenue'
        ])
        
        # Extract net income
        net_income = extract_metric_value(period_income, [
            'net-income', 'net_income', 'netincome'
        ])
        
        # Extract EPS
        eps = extract_metric_value(period_income, [
            'eps', 'earnings-per-share', 'diluted-eps', 'dilutedeps'
        ])
        
        # Calculate margins
        margins = calculate_margins(period_income, revenue)
        
        # Extract balance sheet metrics
        total_assets = extract_metric_value(period_balance, [
            'total-assets', 'total_assets', 'totalassets'
        ])
        total_liabilities = extract_metric_value(period_balance, [
            'total-liabilities', 'total_liabilities', 'totalliabilities'
        ])
        total_equity = extract_metric_value(period_balance, [
            'total-equity', 'total_equity', 'totalequity', 'shareholders-equity'
        ])
        current_assets = extract_metric_value(period_balance, [
            'current-assets', 'current_assets', 'currentassets'
        ])
        current_liabilities = extract_metric_value(period_balance, [
            'current-liabilities', 'current_liabilities', 'currentliabilities'
        ])
        
        # Extract cash flow metrics
        operating_cash_flow = extract_metric_value(period_cash_flow, [
            'operating-cash-flow', 'operating_cash_flow', 'operatingcashflow', 'cash-from-operations'
        ])
        free_cash_flow = extract_metric_value(period_cash_flow, [
            'free-cash-flow', 'free_cash_flow', 'freecashflow', 'fcf'
        ])
        
        # Calculate QoQ and YoY changes if we have previous periods
        revenue_qoq_change = None
        revenue_yoy_change = None
        
        # Try to find previous quarter for QoQ
        prev_quarter = str(int(fiscal_quarter) - 1) if int(fiscal_quarter) > 1 else '4'
        prev_year = fiscal_year if int(fiscal_quarter) > 1 else str(int(fiscal_year) - 1)
        prev_period = f"Q{prev_quarter} {prev_year}"
        
        if prev_period in income_data:
            prev_revenue = extract_metric_value(income_data[prev_period], [
                'revenue', 'total-revenue', 'net-revenue'
            ])
            if prev_revenue and revenue and prev_revenue != 0:
                revenue_qoq_change = ((revenue - prev_revenue) / prev_revenue) * 100
        
        # Try to find same quarter previous year for YoY
        prev_year_period = f"Q{fiscal_quarter} {str(int(fiscal_year) - 1)}"
        if prev_year_period in income_data:
            prev_year_revenue = extract_metric_value(income_data[prev_year_period], [
                'revenue', 'total-revenue', 'net-revenue'
            ])
            if prev_year_revenue and revenue and prev_year_revenue != 0:
                revenue_yoy_change = ((revenue - prev_year_revenue) / prev_year_revenue) * 100
        
        # Check if metrics already exist
        result = await db.execute(
            select(FinancialMetrics).where(
                FinancialMetrics.ticker_symbol == ticker_symbol.upper(),
                FinancialMetrics.fiscal_year == fiscal_year,
                FinancialMetrics.fiscal_quarter == fiscal_quarter
            )
        )
        existing = result.scalar_one_or_none()
        
        # Prepare raw data as JSON
        raw_data = {
            'income_statement': period_income,
            'balance_sheet': period_balance,
            'cash_flow': period_cash_flow
        }
        
        if existing:
            # Update existing record
            existing.revenue = revenue
            existing.net_income = net_income
            existing.eps = eps
            existing.gross_margin = margins.get('gross_margin')
            existing.operating_margin = margins.get('operating_margin')
            existing.net_margin = margins.get('net_margin')
            existing.revenue_qoq_change = revenue_qoq_change
            existing.revenue_yoy_change = revenue_yoy_change
            existing.total_assets = total_assets
            existing.total_liabilities = total_liabilities
            existing.total_equity = total_equity
            existing.current_assets = current_assets
            existing.current_liabilities = current_liabilities
            existing.operating_cash_flow = operating_cash_flow
            existing.free_cash_flow = free_cash_flow
            existing.raw_income_statement = json.dumps(period_income)
            existing.raw_balance_sheet = json.dumps(period_balance)
            existing.raw_cash_flow = json.dumps(period_cash_flow)
            existing.updated_at = datetime.utcnow()
            if session_id:
                existing.session_id = session_id
            metrics = existing
        else:
            # Create new record
            metrics = FinancialMetrics(
                ticker_symbol=ticker_symbol.upper(),
                company_name=company_name,
                fiscal_year=fiscal_year,
                fiscal_quarter=fiscal_quarter,
                report_date=datetime.utcnow(),
                revenue=revenue,
                net_income=net_income,
                eps=eps,
                gross_margin=margins.get('gross_margin'),
                operating_margin=margins.get('operating_margin'),
                net_margin=margins.get('net_margin'),
                revenue_qoq_change=revenue_qoq_change,
                revenue_yoy_change=revenue_yoy_change,
                total_assets=total_assets,
                total_liabilities=total_liabilities,
                total_equity=total_equity,
                current_assets=current_assets,
                current_liabilities=current_liabilities,
                operating_cash_flow=operating_cash_flow,
                free_cash_flow=free_cash_flow,
                raw_income_statement=json.dumps(period_income),
                raw_balance_sheet=json.dumps(period_balance),
                raw_cash_flow=json.dumps(period_cash_flow),
                session_id=session_id,
            )
            db.add(metrics)
        
        stored_metrics.append(metrics)
    
    await db.commit()
    
    # Refresh all metrics
    for m in stored_metrics:
        await db.refresh(m)
    
    logger.info(f"Stored {len(stored_metrics)} financial metrics records for {ticker_symbol}")
    return stored_metrics


async def get_metrics(
    db: AsyncSession,
    ticker_symbol: str,
    fiscal_year: Optional[str] = None,
    fiscal_quarter: Optional[str] = None
) -> Optional[FinancialMetrics]:
    """Get financial metrics for a specific quarter."""
    query = select(FinancialMetrics).where(
        FinancialMetrics.ticker_symbol == ticker_symbol.upper()
    )
    
    if fiscal_year:
        query = query.where(FinancialMetrics.fiscal_year == fiscal_year)
    if fiscal_quarter:
        query = query.where(FinancialMetrics.fiscal_quarter == fiscal_quarter)
    
    query = query.order_by(
        FinancialMetrics.fiscal_year.desc(),
        FinancialMetrics.fiscal_quarter.desc()
    )
    
    result = await db.execute(query)
    return result.scalar_one_or_none()


async def get_metrics_history(
    db: AsyncSession,
    ticker_symbol: str,
    limit: int = 8
) -> List[FinancialMetrics]:
    """Get historical financial metrics for a company."""
    result = await db.execute(
        select(FinancialMetrics)
        .where(FinancialMetrics.ticker_symbol == ticker_symbol.upper())
        .order_by(FinancialMetrics.fiscal_year.desc(), FinancialMetrics.fiscal_quarter.desc())
        .limit(limit)
    )
    return list(result.scalars().all())
