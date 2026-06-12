import re
from typing import Literal
from pathlib import Path

from financial_analyst.state import AnalystState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TICKER_PATTERN = re.compile(r"^[A-Z]{1,5}$")


def _detect_input_type(user_input: str) -> Literal["pdf", "ticker", "url", "text"]:
    """
    Pure function — no LLM, no I/O.
    Inspects the raw user string and returns one of four input types.
    """
    cleaned = user_input.strip()

    # 1. PDF — either a file path ending in .pdf or an existing file on disk
    if cleaned.lower().endswith(".pdf") or Path(cleaned).exists():
        return "pdf"

    # 2. URL — starts with http:// or https://
    if cleaned.startswith("http://") or cleaned.startswith("https://"):
        return "url"

    # 3. Ticker — 1-5 uppercase letters, nothing else (e.g. AAPL, MSFT, GOOGL)
    if TICKER_PATTERN.match(cleaned):
        return "ticker"

    # 4. Default — treat as raw text (pasted earnings call, news article, etc.)
    return "text"


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def input_router(state: AnalystState) -> dict:
    """
    LangGraph node — runs first in the graph.

    Reads:  state["user_input"]
    Writes: state["input_type"]

    Does NOT call an LLM. Pure classification logic so it's
    instant and costs zero tokens.
    """
    user_input = state.get("user_input", "").strip()

    if not user_input:
        return {
            "input_type": "text",
            "errors": ["No input provided — please enter a ticker, URL, PDF path, or text."],
        }

    detected = _detect_input_type(user_input)

    return {
        "input_type": detected,
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Routing function (used by add_conditional_edges in graph.py)
# ---------------------------------------------------------------------------

def route_decision(state: AnalystState,) -> Literal["doc_ingestion", "web_fetcher", "data_extractor"]:
    """
    Called by LangGraph after input_router runs.
    Returns the name of the next node to execute.

    pdf    → doc_ingestion   (extract text from PDF file)
    url    → web_fetcher     (fetch page + search context)
    ticker → web_fetcher     (SEC EDGAR + Tavily search)
    text   → data_extractor  (skip ingestion, go straight to extraction)
    """
    routing_map = {
        "pdf":    "doc_ingestion",
        "url":    "web_fetcher",
        "ticker": "web_fetcher",
        "text":   "data_extractor",
    }

    return routing_map[state["input_type"]]