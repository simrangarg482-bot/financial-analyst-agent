import time

def _invoke_with_retry(chain, inputs: dict, max_retries: int = 3) -> str:
    """Retries LLM calls on rate limit errors with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return chain.invoke(inputs)
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                logger.warning(f"Rate limited — waiting {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                raise

import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from financial_analyst.state import AnalystState
from financial_analyst.config import settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ANALYST_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a senior equity research analyst at a top-tier investment bank 
with 20 years of experience analyzing financial statements.

Your analysis must follow this exact chain-of-thought structure:

1. REVENUE & GROWTH ANALYSIS
   - Assess revenue trajectory and YoY growth
   - Compare gross margin trends
   - Identify revenue quality (recurring vs one-time)

2. PROFITABILITY ANALYSIS  
   - Evaluate operating leverage
   - Assess EBITDA margins
   - Analyze EPS trend and quality of earnings

3. BALANCE SHEET HEALTH
   - Evaluate debt levels and debt-to-equity
   - Assess liquidity (current ratio, cash position)
   - Identify any balance sheet risks

4. FREE CASH FLOW ANALYSIS
   - Compare FCF to net income (cash conversion quality)
   - Assess capital allocation efficiency

5. FORWARD GUIDANCE & OUTLOOK
   - Summarize management guidance
   - Assess credibility of guidance vs historical accuracy

6. KEY RISKS
   - Identify top 3-5 material risks
   - Assess probability and potential impact

7. INVESTMENT THESIS SUMMARY
   - Bull case (2-3 sentences)
   - Bear case (2-3 sentences)
   - Overall sentiment: BULLISH / NEUTRAL / BEARISH with justification

Rules:
- Every claim must be supported by a specific number from the data
- Flag any data gaps explicitly  
- Be direct — avoid hedging language like "may" or "could"
- Write for a sophisticated institutional investor audience"""
    ),
    (
        "human",
        """Analyze the following financial data.

COMPANY: {company_name} ({ticker}) — {period}

STRUCTURED FINANCIAL DATA:
{structured_data}

RAW DOCUMENT EXCERPTS:
{raw_content}

PREVIOUS CRITIQUE (if any):
{critique}

Produce a comprehensive investment analysis following the 7-step framework above.
If there was a previous critique, address every point raised."""
    )
])


# ---------------------------------------------------------------------------
# LLM setup
# ---------------------------------------------------------------------------

def _get_llm():
    return ChatOpenAI(
        model=settings.model_name,
        openai_api_key=settings.openrouter_api_key,
        openai_api_base=settings.openrouter_base_url,
        temperature=settings.temperature,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_structured_data(data: dict) -> str:
    """
    Converts the nested structured_data dict into a
    readable format for the LLM prompt.
    """
    if not data:
        return "No structured data available."

    lines = []

    # Income statement
    inc = data.get("income_statement", {})
    if any(v is not None for v in inc.values()):
        lines.append("INCOME STATEMENT:")
        if inc.get("revenue"):
            lines.append(f"  Revenue: ${inc['revenue']:,.1f}M")
        if inc.get("revenue_growth_yoy"):
            lines.append(f"  Revenue Growth YoY: {inc['revenue_growth_yoy']:.1f}%")
        if inc.get("gross_margin"):
            lines.append(f"  Gross Margin: {inc['gross_margin']:.1f}%")
        if inc.get("operating_income"):
            lines.append(f"  Operating Income: ${inc['operating_income']:,.1f}M")
        if inc.get("net_income"):
            lines.append(f"  Net Income: ${inc['net_income']:,.1f}M")
        if inc.get("eps_diluted"):
            lines.append(f"  Diluted EPS: ${inc['eps_diluted']:.2f}")
        if inc.get("ebitda"):
            lines.append(f"  EBITDA: ${inc['ebitda']:,.1f}M")

    # Balance sheet
    bal = data.get("balance_sheet", {})
    if any(v is not None for v in bal.values()):
        lines.append("\nBALANCE SHEET:")
        if bal.get("cash_and_equivalents"):
            lines.append(f"  Cash: ${bal['cash_and_equivalents']:,.1f}M")
        if bal.get("total_debt"):
            lines.append(f"  Total Debt: ${bal['total_debt']:,.1f}M")
        if bal.get("debt_to_equity"):
            lines.append(f"  Debt/Equity: {bal['debt_to_equity']:.2f}x")
        if bal.get("total_equity"):
            lines.append(f"  Total Equity: ${bal['total_equity']:,.1f}M")

    # Key metrics
    met = data.get("key_metrics", {})
    if any(v is not None for v in met.values()):
        lines.append("\nKEY METRICS:")
        if met.get("pe_ratio"):
            lines.append(f"  P/E Ratio: {met['pe_ratio']:.1f}x")
        if met.get("roe"):
            lines.append(f"  ROE: {met['roe']:.1f}%")
        if met.get("roa"):
            lines.append(f"  ROA: {met['roa']:.1f}%")
        if met.get("free_cash_flow"):
            lines.append(f"  Free Cash Flow: ${met['free_cash_flow']:,.1f}M")

    # Guidance and risks
    if data.get("forward_guidance"):
        lines.append(f"\nFORWARD GUIDANCE:\n  {data['forward_guidance']}")

    if data.get("key_risks"):
        lines.append("\nKEY RISKS:")
        for risk in data["key_risks"]:
            lines.append(f"  - {risk}")

    if data.get("highlights"):
        lines.append("\nHIGHLIGHTS:")
        for h in data["highlights"]:
            lines.append(f"  + {h}")

    return "\n".join(lines) if lines else "No structured data available."


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def financial_analyst(state: AnalystState) -> dict:
    """
    LangGraph node — core reasoning engine.

    Reads:  state["structured_data"], state["raw_content"],
            state["critique"], state["revision_count"]
    Writes: state["analysis"], state["revision_count"]

    On first run: critique is empty, produces initial analysis.
    On revision:  receives critique feedback, addresses every point raised.
    """
    structured_data = state.get("structured_data", {})
    raw_content = state.get("raw_content", "")
    critique = state.get("critique", "")
    revision_count = state.get("revision_count", 0)

    # Format inputs
    company_name = structured_data.get("company_name") or "Unknown Company"
    ticker = structured_data.get("ticker") or "N/A"
    period = structured_data.get("period") or "Unknown Period"
    formatted_data = _format_structured_data(structured_data)

    # Truncate raw content for context — analyst needs less than extractor
    raw_excerpt = raw_content[:6000] if raw_content else "No raw content available."

    logger.info(
        f"Running financial_analyst node "
        f"(revision {revision_count}) for {company_name}"
    )

    try:
        llm = _get_llm()
        chain = ANALYST_PROMPT | llm | StrOutputParser()

        analysis = _invoke_with_retry(chain,{
            "company_name": company_name,
            "ticker": ticker,
            "period": period,
            "structured_data": formatted_data,
            "raw_content": raw_excerpt,
            "critique": critique if critique else "None — this is the initial analysis.",
        })

        logger.info(f"Analysis complete — {len(analysis)} chars generated")

        return {
            "analysis": analysis,
            "revision_count": revision_count + 1,
        }

    except Exception as e:
        logger.error(f"financial_analyst node failed: {e}")
        return {
            "analysis": "",
            "errors": [f"financial_analyst failed: {str(e)}"],
            "revision_count": revision_count,
        }