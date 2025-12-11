"""Document parser tool for extracting content from earnings reports."""

from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
import httpx
import io
import tempfile
import os


class PDFParseInput(BaseModel):
    """Input schema for PDF parsing."""
    pdf_url: str = Field(description="URL of the PDF document to parse")


class DocumentParserTool(BaseTool):
    """Tool for downloading and parsing PDF earnings reports."""
    
    name: str = "parse_pdf"
    description: str = """
    Download and parse a PDF earnings report from a URL.
    Returns the extracted text content from the PDF.
    Use this when you have found a PDF link to an earnings report.
    """
    args_schema: Type[BaseModel] = PDFParseInput
    
    def _run(self, pdf_url: str) -> str:
        """Download and parse PDF."""
        try:
            # Download PDF
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                response = client.get(pdf_url, headers=headers)
                response.raise_for_status()
            
            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not pdf_url.lower().endswith('.pdf'):
                return f"URL does not appear to be a PDF. Content-Type: {content_type}"
            
            # Try parsing with pdfplumber first (better for tables)
            try:
                import pdfplumber
                
                pdf_bytes = io.BytesIO(response.content)
                extracted_text = []
                
                with pdfplumber.open(pdf_bytes) as pdf:
                    total_pages = len(pdf.pages)
                    
                    # Extract text from each page (limit to first 50 pages for performance)
                    for i, page in enumerate(pdf.pages[:50]):
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            extracted_text.append(f"\n--- Page {i+1} ---\n{page_text}")
                        
                        # Also try to extract tables
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                table_str = "\n[TABLE]\n"
                                for row in table:
                                    row_str = " | ".join(str(cell) if cell else "" for cell in row)
                                    table_str += row_str + "\n"
                                extracted_text.append(table_str)
                    
                    if total_pages > 50:
                        extracted_text.append(f"\n... (Parsed 50 of {total_pages} pages) ...")
                
                full_text = "\n".join(extracted_text)
                
                if not full_text.strip():
                    return "PDF appears to be empty or contains only images."
                
                # Truncate if too long (keep key sections)
                if len(full_text) > 50000:
                    full_text = full_text[:50000] + "\n\n... [Content truncated for processing] ..."
                
                return f"**Successfully parsed PDF from:** {pdf_url}\n\n{full_text}"
                
            except Exception as e:
                # Fallback to pypdf
                from pypdf import PdfReader
                
                pdf_bytes = io.BytesIO(response.content)
                reader = PdfReader(pdf_bytes)
                
                extracted_text = []
                for i, page in enumerate(reader.pages[:50]):
                    text = page.extract_text() or ""
                    if text.strip():
                        extracted_text.append(f"\n--- Page {i+1} ---\n{text}")
                
                full_text = "\n".join(extracted_text)
                
                if not full_text.strip():
                    return "PDF appears to be empty or contains only images."
                
                if len(full_text) > 50000:
                    full_text = full_text[:50000] + "\n\n... [Content truncated for processing] ..."
                
                return f"**Successfully parsed PDF from:** {pdf_url}\n\n{full_text}"
                
        except httpx.TimeoutException:
            return f"Timeout while downloading PDF from {pdf_url}"
        except Exception as e:
            return f"Error parsing PDF: {str(e)}"
    
    async def _arun(self, pdf_url: str) -> str:
        """Async PDF parsing."""
        return self._run(pdf_url)


class HTMLDocumentInput(BaseModel):
    """Input schema for HTML document parsing."""
    url: str = Field(description="URL of the HTML document to parse")


class HTMLDocumentTool(BaseTool):
    """Tool for parsing HTML-based earnings reports."""
    
    name: str = "parse_html_document"
    description: str = """
    Parse an HTML-based earnings report or financial document.
    Some companies publish earnings data as HTML pages rather than PDFs.
    """
    args_schema: Type[BaseModel] = HTMLDocumentInput
    
    def _run(self, url: str) -> str:
        """Parse HTML document."""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove non-content elements
            for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
                element.decompose()
            
            # Try to find main content area
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up excessive whitespace
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            if len(text) > 50000:
                text = text[:50000] + "\n\n... [Content truncated for processing] ..."
            
            return f"**Successfully parsed HTML from:** {url}\n\n{text}"
            
        except Exception as e:
            return f"Error parsing HTML document: {str(e)}"
    
    async def _arun(self, url: str) -> str:
        """Async HTML parsing."""
        return self._run(url)

