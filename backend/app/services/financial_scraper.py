"""Service for scraping financial data from discountingcashflows.com."""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from playwright.async_api import async_playwright, Browser, Page
import re

logger = logging.getLogger(__name__)


class FinancialScraper:
    """Scraper for financial statements from discountingcashflows.com."""
    
    def __init__(self):
        self.browser: Optional[Browser] = None
        self.playwright = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def scrape_financial_statement(
        self, 
        ticker_symbol: str, 
        statement_type: str
    ) -> Dict[str, Any]:
        """
        Scrape a financial statement from discountingcashflows.com.
        
        Args:
            ticker_symbol: Stock ticker symbol (e.g., 'CB', 'AAPL')
            statement_type: One of 'income-statement', 'balance-sheet-statement', 'cash-flow-statement'
        
        Returns:
            Dictionary with metrics organized by period (quarters)
        """
        url = f"https://discountingcashflows.com/company/{ticker_symbol.upper()}/{statement_type}/quarterly/"
        
        logger.info(f"Scraping {statement_type} for {ticker_symbol} from {url}")
        
        if not self.browser:
            raise RuntimeError("Browser not initialized. Use async context manager.")
        
        page = await self.browser.new_page()
        
        try:
            # Navigate to the page
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Wait for the report table to be loaded
            await page.wait_for_selector('table#report-table', timeout=10000)
            
            # Wait a bit more for JavaScript to fully render
            await asyncio.sleep(2)
            
            # Extract table data
            table_data = await page.evaluate("""
                () => {
                    const table = document.querySelector('table#report-table');
                    if (!table) return null;
                    
                    // Get headers from thead - handle the structure with spans and small tags
                    const thead = table.querySelector('thead');
                    const headers = [];
                    if (thead) {
                        const headerRow = thead.querySelector('tr');
                        if (headerRow) {
                            // Skip first column (Period Ending:)
                            const headerCells = headerRow.querySelectorAll('th');
                            headerCells.forEach((cell, index) => {
                                if (index === 0) {
                                    // First column is "Period Ending:" - skip it
                                    return;
                                }
                                // Extract text from span and small tags
                                const span = cell.querySelector('span');
                                const small = cell.querySelector('small');
                                let periodText = '';
                                if (span) {
                                    periodText = span.textContent.trim();
                                }
                                if (small && small.textContent.trim()) {
                                    periodText += ' ' + small.textContent.trim();
                                }
                                // Fallback to full text if no span/small found
                                if (!periodText) {
                                    periodText = cell.textContent.trim();
                                }
                                // Clean up the period text
                                periodText = periodText.replace(/\\s+/g, ' ').trim();
                                if (periodText && periodText !== 'Period Ending:') {
                                    headers.push(periodText);
                                }
                            });
                        }
                    }
                    
                    // Get data from tbody
                    const tbody = table.querySelector('tbody');
                    const rows = [];
                    if (tbody) {
                        // Get all rows, but prioritize parent rows (main financial metrics)
                        // Also include rows with row-description cells that aren't child rows
                        const allRows = tbody.querySelectorAll('tr');
                        const processedIds = new Set(); // Track processed metric IDs to avoid duplicates
                        
                        allRows.forEach(tr => {
                            // Skip child rows (they have class starting with "child-row-")
                            if (tr.classList.contains('child-row') || tr.className.includes('child-row-')) {
                                return;
                            }
                            
                            // Skip if this row is a child of another row
                            if (tr.className && tr.className.includes('child-row')) {
                                return;
                            }
                            // Find the row-description cell which contains the metric information
                            const rowDescriptionCell = tr.querySelector('td.row-description');
                            if (!rowDescriptionCell) return; // Skip if no row-description cell
                            
                            // Find the span with class "row-description-text" which contains the metric identifier
                            const metricSpan = rowDescriptionCell.querySelector('span.row-description-text');
                            
                            let metricId = null;
                            let metricName = null;
                            
                            if (metricSpan) {
                                // Primary: Get the metric ID from the data-dev attribute (camelCase identifier)
                                metricId = metricSpan.getAttribute('data-dev');
                                
                                // Get the metric name from the span's title attribute or text content
                                metricName = metricSpan.getAttribute('title') || metricSpan.textContent.trim();
                            }
                            
                            // Fallback: Use tr id attribute if data-dev is not available
                            if (!metricId) {
                                metricId = tr.getAttribute('id');
                                // Remove "row-" prefix if present
                                if (metricId && metricId.startsWith('row-')) {
                                    metricId = metricId.substring(4);
                                }
                            }
                            
                            // Fallback: Try to get name from first cell text if span not found
                            if (!metricName) {
                                metricName = rowDescriptionCell.textContent.trim();
                            }
                            
                            // If still no metric ID, try to derive from text content
                            // This handles cases where rows don't have data-dev or id attributes
                            if (!metricId && metricName) {
                                // Create a slug-like ID from the name
                                metricId = metricName.toLowerCase()
                                    .replace(/[^a-z0-9]+/g, '-')
                                    .replace(/^-|-$/g, '')
                                    .replace(/\s+/g, '-');
                            }
                            
                            // Skip if we still can't identify the metric
                            if (!metricId) return;
                            
                            // Skip if this looks like a calculated/derived metric and we want main line items
                            // But don't skip - we want all metrics for now
                            
                            // Get all formatted-value cells from this parent row
                            // These contain the actual financial values for each period
                            const formattedValueCells = tr.querySelectorAll('td.formatted-value');
                            const values = [];
                            
                            formattedValueCells.forEach(cell => {
                                // Prefer data-value attribute for raw numeric value (more accurate)
                                let text = cell.getAttribute('data-value');
                                
                                // Fallback to text content if data-value not available
                                if (!text) {
                                    text = cell.textContent.trim();
                                }
                                
                                // Skip empty cells
                                if (!text) {
                                    values.push(null);
                                    return;
                                }
                                
                                // Parse numeric values (remove commas, handle negative in parentheses)
                                let value = text.replace(/,/g, '');
                                if (value.startsWith('(') && value.endsWith(')')) {
                                    value = '-' + value.slice(1, -1);
                                }
                                
                                // Try to parse as float, otherwise keep as string
                                const numValue = parseFloat(value);
                                values.push(isNaN(numValue) ? (text || null) : numValue);
                            });
                            
                            // Only add row if we have at least one value and haven't processed this ID
                            if ((values.length > 0 || metricId) && metricId && !processedIds.has(metricId)) {
                                processedIds.add(metricId);
                                rows.push({
                                    id: metricId,
                                    name: metricName || metricId,
                                    values: values
                                });
                            }
                        });
                    }
                    
                    return {
                        headers: headers,
                        rows: rows
                    };
                }
            """)
            
            if not table_data:
                logger.warning(f"No table data found for {ticker_symbol} {statement_type}")
                return {}
            
            # Organize data by period (columns)
            periods = table_data.get('headers', [])
            rows = table_data.get('rows', [])
            
            logger.debug(f"Extracted {len(periods)} periods and {len(rows)} rows for {ticker_symbol} {statement_type}")
            if periods:
                logger.debug(f"Sample periods: {periods[:3]}")
            if rows:
                logger.debug(f"Sample row: id={rows[0].get('id')}, name={rows[0].get('name')}, values_count={len(rows[0].get('values', []))}")
                # Log all metric IDs found
                all_metric_ids = [row.get('id', '') for row in rows]
                logger.info(f"Found {len(all_metric_ids)} metric IDs for {ticker_symbol} {statement_type}: {all_metric_ids[:15]}")  # Log first 15
            
            # Create a structured format: {period: {metric_id: value}}
            result = {}
            
            for i, period in enumerate(periods):
                if not period or period.strip() == '':
                    continue
                
                period_data = {}
                for row in rows:
                    metric_id = row.get('id', '')
                    metric_name = row.get('name', '')
                    values = row.get('values', [])
                    
                    if i < len(values) and values[i] is not None:
                        value = values[i]
                        # Use metric_id as key, but also store name
                        period_data[metric_id] = {
                            'value': value,
                            'name': metric_name
                        }
                
                if period_data:  # Only add period if it has data
                    result[period] = period_data
            
            logger.info(f"Successfully scraped {len(result)} periods with {len(rows)} metrics for {ticker_symbol} {statement_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error scraping {statement_type} for {ticker_symbol}: {e}", exc_info=True)
            return {}
        finally:
            await page.close()
    
    async def scrape_all_statements(self, ticker_symbol: str) -> Dict[str, Any]:
        """
        Scrape all three financial statements for a ticker.
        
        Returns:
            Dictionary with 'income_statement', 'balance_sheet', 'cash_flow' keys
        """
        results = {}
        
        statements = [
            ('income-statement', 'income_statement'),
            ('balance-sheet-statement', 'balance_sheet'),
            ('cash-flow-statement', 'cash_flow')
        ]
        
        for statement_type, key in statements:
            try:
                data = await self.scrape_financial_statement(ticker_symbol, statement_type)
                results[key] = data
                # Add a small delay between requests to be respectful
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Failed to scrape {statement_type} for {ticker_symbol}: {e}")
                results[key] = {}
        
        return results


async def scrape_company_financials(ticker_symbol: str) -> Dict[str, Any]:
    """
    Convenience function to scrape all financial statements for a company.
    
    Args:
        ticker_symbol: Stock ticker symbol
    
    Returns:
        Dictionary with financial data from all three statements
    """
    async with FinancialScraper() as scraper:
        return await scraper.scrape_all_statements(ticker_symbol)
