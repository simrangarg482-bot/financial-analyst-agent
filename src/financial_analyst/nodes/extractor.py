import json
import logging
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from financial_analyst.state import AnalystState
from financial_analyst.config import settings

logger = logging.getLogger(__name__)



# Pydantic schema — this is what the LLM must return

class IncomeStatement(BaseModel):
    revenue: Optional[float] = Field(None, description="Total revenue in millions USD")
    revenue_growth_yoy: Optional[float] = Field(None, description="Revenue YoY growth %")
    gross_profit: Optional[float] = Field(None, description="Gross profit in millions USD")
    gross_margin: Optional[float] = Field(None, description="Gross margin %")
    operating_income: Optional[float] = Field(None, description="Operating income in millions USD")
    net_income: Optional[float] = Field(None, description="Net income in millions USD")
    eps_basic: Optional[float] = Field(None, description="Basic EPS in USD")
    eps_diluted: Optional[float] = Field(None, description="Diluted EPS in USD")
    ebitda: Optional[float] = Field(None, description="EBITDA in millions USD")


class BalanceSheet(BaseModel):
    total_assets: Optional[float] = Field(None, description="Total assets in millions USD")
    total_liabilities: Optional[float] = Field(None, description="Total liabilities in millions USD")
    total_equity: Optional[float] = Field(None, description="Shareholders equity in millions USD")
    cash_and_equivalents: Optional[float] = Field(None, description="Cash and equivalents in millions USD")
    total_debt: Optional[float] = Field(None, description="Total debt in millions USD")
    debt_to_equity: Optional[float] = Field(None, description="Debt to equity ratio")


class KeyMetrics(BaseModel):
    pe_ratio: Optional[float] = Field(None, description="Price to earnings ratio")
    ps_ratio: Optional[float] = Field(None, description="Price to sales ratio")
    roe: Optional[float] = Field(None, description="Return on equity %")
    roa: Optional[float] = Field(None, description="Return on assets %")
    current_ratio: Optional[float] = Field(None, description="Current ratio")
    free_cash_flow: Optional[float] = Field(None, description="Free cash flow in millions USD")


class FinancialData(BaseModel):
    """Complete structured financial data extracted from the document."""
    company_name: Optional[str] = Field(None, description="Company name")
    ticker: Optional[str] = Field(None, description="Stock ticker symbol")
    period: Optional[str] = Field(None, description="Reporting period e.g. FY2024, Q3 2024")
    currency: Optional[str] = Field(None, description="Currency e.g. USD, EUR")
    income_statement: IncomeStatement = Field(default_factory=IncomeStatement)
    balance_sheet: BalanceSheet = Field(default_factory=BalanceSheet)
    key_metrics: KeyMetrics = Field(default_factory=KeyMetrics)
    forward_guidance: Optional[str] = Field(None, description="Management forward guidance or outlook")
    key_risks: list[str] = Field(default_factory=list, description="Key risks mentioned")
    highlights: list[str] = Field(default_factory=list, description="Key positive highlights")


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _get_llm():
    return ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=0,  # extraction must be deterministic
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a financial data extraction specialist. Your job is to extract 
structured financial data from documents with perfect accuracy.

Rules:
- Extract ONLY what is explicitly stated in the document
- Use null for any value not found — never guess or estimate  
- All monetary values should be in millions USD unless stated otherwise
- Percentages should be decimal numbers (e.g. 15.3 for 15.3%)
- Be precise — wrong numbers are worse than null values"""
    ),
    (
        "human",
        """Extract all financial data from the following document.

DOCUMENT:
{raw_content}

Return the complete structured financial data."""
    )
])


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def data_extractor(state: AnalystState) -> dict:
    """
    LangGraph node — extracts structured financial data from raw text.

    Reads:  state["raw_content"]
    Writes: state["structured_data"], state["errors"]

    Uses LLM structured output with a Pydantic schema — the LLM is forced
    to return valid JSON matching FinancialData, no parsing needed.
    """
    raw_content = state.get("raw_content", "").strip()

    if not raw_content:
        return {
            "structured_data": {},
            "errors": ["data_extractor: no raw_content to extract from."],
        }

    # Truncate to avoid context limit — 12k chars covers most filings
    if len(raw_content) > 12000:
        logger.warning(f"Truncating raw_content from {len(raw_content)} to 12000 chars")
        raw_content = raw_content[:12000]

    try:
        llm = _get_llm()
        structured_llm = llm.with_structured_output(FinancialData)
        chain = EXTRACTION_PROMPT | structured_llm

        logger.info("Running data extraction...")
        result: FinancialData = chain.invoke({"raw_content": raw_content})

        # Convert Pydantic model to dict for state storage
        structured_data = result.model_dump(exclude_none=False)

        logger.info(f"Extraction complete: {result.company_name} {result.period}")

        return {
            "structured_data": structured_data,
            "errors": [],
        }

    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        return {
            "structured_data": {},
            "errors": [f"data_extractor failed: {str(e)}"],
        }