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
                        const trElements = tbody.querySelectorAll('tr[id]');
                        trElements.forEach(tr => {
                            const metricId = tr.getAttribute('id');
                            if (!metricId) return;
                            
                            // Get all cells, skip first one (row header)
                            const cells = tr.querySelectorAll('td');
                            const values = [];
                            
                            // Start from index 1 to skip the first column (row header)
                            for (let i = 1; i < cells.length; i++) {
                                const cell = cells[i];
                                const text = cell.textContent.trim();
                                
                                // Skip empty cells
                                if (!text) {
                                    values.push(null);
                                    continue;
                                }
                                
                                // Parse numeric values (remove commas, handle negative in parentheses)
                                let value = text.replace(/,/g, '');
                                if (value.startsWith('(') && value.endsWith(')')) {
                                    value = '-' + value.slice(1, -1);
                                }
                                
                                // Try to parse as float, otherwise keep as string
                                const numValue = parseFloat(value);
                                values.push(isNaN(numValue) ? (text || null) : numValue);
                            }
                            
                            // Get the metric name from the first cell (row header)
                            const firstCell = tr.querySelector('td:first-child');
                            const metricName = firstCell ? firstCell.textContent.trim() : metricId;
                            
                            rows.push({
                                id: metricId,
                                name: metricName,
                                values: values
                            });
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
