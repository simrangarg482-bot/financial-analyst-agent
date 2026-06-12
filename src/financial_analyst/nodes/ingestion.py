import io
import logging
from pathlib import Path

import pdfplumber
import fitz  # PyMuPDF
import requests
from tavily import TavilyClient

from financial_analyst.state import AnalystState
from financial_analyst.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

def _extract_text_pdfplumber(pdf_path: str) -> str:
    """
    Primary PDF extractor.
    pdfplumber is better at tables and structured layouts (10-K, 10-Q).
    """
    text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            # Extract plain text
            page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if page_text:
                text_parts.append(f"--- Page {page_num} ---\n{page_text}")

            # Extract tables and convert to readable text
            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                table_text = _table_to_text(table)
                if table_text:
                    text_parts.append(f"[TABLE on page {page_num}]\n{table_text}")

    return "\n\n".join(text_parts)


def _extract_text_pymupdf(pdf_path: str) -> str:
    """
    Fallback PDF extractor.
    PyMuPDF handles scanned/image-heavy PDFs better.
    """
    text_parts = []

    doc = fitz.open(pdf_path)
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        if page_text.strip():
            text_parts.append(f"--- Page {page_num} ---\n{page_text}")

    doc.close()
    return "\n\n".join(text_parts)


def _table_to_text(table: list[list]) -> str:
    """
    Converts a pdfplumber table (list of lists) into
    a readable plain-text format for the LLM.
    """
    rows = []
    for row in table:
        cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
        rows.append(" | ".join(cleaned))
    return "\n".join(rows)


def _extract_pdf(pdf_path: str) -> tuple[str, list[str]]:
    """
    Tries pdfplumber first, falls back to PyMuPDF.
    Returns (extracted_text, errors).
    """
    errors = []

    try:
        text = _extract_text_pdfplumber(pdf_path)
        if len(text.strip()) > 100:
            logger.info(f"pdfplumber extracted {len(text)} chars from {pdf_path}")
            return text, errors
        else:
            errors.append("pdfplumber returned very little text — trying PyMuPDF fallback.")
    except Exception as e:
        errors.append(f"pdfplumber failed: {e} — trying PyMuPDF fallback.")

    try:
        text = _extract_text_pymupdf(pdf_path)
        logger.info(f"PyMuPDF extracted {len(text)} chars from {pdf_path}")
        return text, errors
    except Exception as e:
        errors.append(f"PyMuPDF also failed: {e}")
        return "", errors


# ---------------------------------------------------------------------------
# Web fetcher helpers
# ---------------------------------------------------------------------------

def _fetch_by_ticker(ticker: str) -> tuple[str, list[str]]:
    """
    Given a ticker like AAPL, fetches:
    1. Recent news via Tavily search
    2. SEC EDGAR latest filing metadata (free, no API key needed)
    """
    errors = []
    content_parts = []

    # --- Tavily search ---
    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        query = f"{ticker} earnings financial results annual report 10-K"
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=5,
            include_raw_content=True,
        )
        for result in response.get("results", []):
            title = result.get("title", "")
            url = result.get("url", "")
            content = result.get("raw_content") or result.get("content", "")
            if content:
                content_parts.append(
                    f"[SOURCE: {title}]\n[URL: {url}]\n{content[:3000]}"
                )
    except Exception as e:
        errors.append(f"Tavily search failed for ticker {ticker}: {e}")

    # --- SEC EDGAR company search ---
    try:
        headers = {"User-Agent": "financial-analyst-agent contact@example.com"}
        search_url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2023-01-01&forms=10-K"
        resp = requests.get(search_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            hits = data.get("hits", {}).get("hits", [])
            if hits:
                filing_info = hits[0].get("_source", {})
                content_parts.append(
                    f"[SEC EDGAR FILING]\n"
                    f"Company: {filing_info.get('entity_name', ticker)}\n"
                    f"Form: {filing_info.get('file_date', '')}\n"
                    f"Period: {filing_info.get('period_of_report', '')}"
                )
    except Exception as e:
        errors.append(f"SEC EDGAR lookup failed: {e}")

    return "\n\n".join(content_parts), errors


def _fetch_by_url(url: str) -> tuple[str, list[str]]:
    """
    Fetches and returns content from a given URL via Tavily.
    Tavily's extract endpoint is cleaner than raw requests for financial pages.
    """
    errors = []

    try:
        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.extract(urls=[url])
        results = response.get("results", [])
        if results:
            raw = results[0].get("raw_content", "")
            return raw[:15000], errors  # cap at 15k chars
        else:
            errors.append(f"Tavily extract returned no content for {url}")
            return "", errors
    except Exception as e:
        errors.append(f"URL fetch failed for {url}: {e}")
        return "", errors


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def doc_ingestion(state: AnalystState) -> dict:
    """
    LangGraph node — handles PDF input type.

    Reads:  state["user_input"]  (path to a PDF file)
    Writes: state["raw_content"], state["errors"]
    """
    pdf_path = state["user_input"].strip()

    if not Path(pdf_path).exists():
        return {
            "raw_content": "",
            "errors": [f"PDF file not found at path: {pdf_path}"],
        }

    if not Path(pdf_path).suffix.lower() == ".pdf":
        return {
            "raw_content": "",
            "errors": [f"File is not a PDF: {pdf_path}"],
        }

    logger.info(f"Ingesting PDF: {pdf_path}")
    text, errors = _extract_pdf(pdf_path)

    if not text:
        return {
            "raw_content": "",
            "errors": errors + ["PDF extraction produced no text. File may be scanned or encrypted."],
        }

    return {
        "raw_content": text,
        "errors": errors,
    }


def web_fetcher(state: AnalystState) -> dict:
    """
    LangGraph node — handles ticker and URL input types.

    Reads:  state["user_input"], state["input_type"]
    Writes: state["raw_content"], state["errors"]
    """
    user_input = state["user_input"].strip()
    input_type = state["input_type"]

    logger.info(f"Web fetcher running for input_type={input_type}, input={user_input}")

    if input_type == "ticker":
        text, errors = _fetch_by_ticker(user_input)
    elif input_type == "url":
        text, errors = _fetch_by_url(user_input)
    else:
        return {
            "raw_content": "",
            "errors": [f"web_fetcher called with unexpected input_type: {input_type}"],
        }

    if not text:
        return {
            "raw_content": "",
            "errors": errors + [f"No content retrieved for input: {user_input}"],
        }

    return {
        "raw_content": text,
        "errors": errors,
    }