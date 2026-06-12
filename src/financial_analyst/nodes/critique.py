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

CRITIQUE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a ruthlessly rigorous peer reviewer at an investment bank.
Your job is to find weaknesses in financial analyses before they reach clients.

You evaluate analyses on these 6 dimensions:

1. FACTUAL ACCURACY — Are all numbers correct and sourced from the data?
2. COMPLETENESS — Are any major financial areas missing or underdeveloped?
3. LOGICAL CONSISTENCY — Do conclusions follow from the evidence presented?
4. RISK COVERAGE — Are material risks identified and properly weighted?
5. GUIDANCE ASSESSMENT — Is management guidance critically evaluated?
6. INVESTMENT THESIS — Is the bull/bear case specific and well-supported?

Your response must follow this exact format:

VERDICT: PASS or REVISE

SCORES:
- Factual Accuracy: X/10
- Completeness: X/10  
- Logical Consistency: X/10
- Risk Coverage: X/10
- Guidance Assessment: X/10
- Investment Thesis: X/10
- Overall: X/10

ISSUES FOUND:
(List each issue as: [DIMENSION] specific problem — specific fix required)
If no issues, write: None

REQUIRED CHANGES:
(Numbered list of specific changes the analyst must make)
If no changes needed, write: None

VERDICT RATIONALE:
(2-3 sentences explaining the overall verdict)

Scoring guide:
- PASS if Overall score >= 7/10 AND no score below 5/10
- REVISE if Overall score < 7/10 OR any score below 5/10"""
    ),
    (
        "human",
        """Review the following financial analysis.

FINANCIAL DATA USED:
{structured_data}

ANALYSIS TO REVIEW:
{analysis}

REVISION NUMBER: {revision_count}

Apply your 6-dimension framework and return your verdict."""
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
        temperature=0,  # critique must be consistent
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_verdict(critique_text: str) -> str:
    """
    Extracts PASS or REVISE from the critique output.
    Defaults to PASS after max revisions to prevent infinite loops.
    """
    upper = critique_text.upper()

    # Look for explicit verdict line
    for line in upper.splitlines():
        if line.startswith("VERDICT:"):
            if "REVISE" in line:
                return "REVISE"
            if "PASS" in line:
                return "PASS"

    # Fallback — scan full text
    if "VERDICT: REVISE" in upper:
        return "REVISE"
    if "VERDICT: PASS" in upper:
        return "PASS"

    # Default to PASS if verdict is unclear
    logger.warning("Could not parse verdict from critique — defaulting to PASS")
    return "PASS"


def _format_structured_data_summary(data: dict) -> str:
    """Compact summary of structured data for the critique prompt."""
    if not data:
        return "No structured data available."

    parts = []
    if data.get("company_name"):
        parts.append(f"Company: {data['company_name']} ({data.get('ticker', 'N/A')})")
    if data.get("period"):
        parts.append(f"Period: {data['period']}")

    inc = data.get("income_statement", {})
    if inc.get("revenue"):
        parts.append(f"Revenue: ${inc['revenue']:,.1f}M")
    if inc.get("net_income"):
        parts.append(f"Net Income: ${inc['net_income']:,.1f}M")
    if inc.get("eps_diluted"):
        parts.append(f"EPS: ${inc['eps_diluted']:.2f}")

    bal = data.get("balance_sheet", {})
    if bal.get("total_debt"):
        parts.append(f"Total Debt: ${bal['total_debt']:,.1f}M")
    if bal.get("debt_to_equity"):
        parts.append(f"D/E Ratio: {bal['debt_to_equity']:.2f}x")

    return "\n".join(parts) if parts else "Minimal structured data available."


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def critique(state: AnalystState) -> dict:
    """
    LangGraph node — evaluates the analysis and decides PASS or REVISE.

    Reads:  state["analysis"], state["structured_data"],
            state["revision_count"]
    Writes: state["critique"]

    The routing function should_revise() in graph.py reads state["critique"]
    and routes to either financial_analyst (revise) or report_generator (pass).
    """
    analysis = state.get("analysis", "").strip()
    structured_data = state.get("structured_data", {})
    revision_count = state.get("revision_count", 0)

    if not analysis:
        logger.warning("critique node received empty analysis — forcing PASS")
        return {
            "critique": "VERDICT: PASS\n\nNo analysis provided to critique.",
        }

    logger.info(f"Running critique node (revision {revision_count})")

    try:
        llm = _get_llm()
        chain = CRITIQUE_PROMPT | llm | StrOutputParser()

        critique_text = chain.invoke({
            "structured_data": _format_structured_data_summary(structured_data),
            "analysis": analysis,
            "revision_count": revision_count,
        })

        verdict = _parse_verdict(critique_text)
        logger.info(f"Critique verdict: {verdict} (revision {revision_count})")

        return {
            "critique": critique_text,
        }

    except Exception as e:
        logger.error(f"critique node failed: {e}")
        # On failure, pass through — don't block the pipeline
        return {
            "critique": f"VERDICT: PASS\n\nCritique failed due to error: {str(e)}",
        }


# ---------------------------------------------------------------------------
# Routing function (used by add_conditional_edges in graph.py)
# ---------------------------------------------------------------------------

def should_revise(state: AnalystState) -> str:
    """
    Called by LangGraph after critique node runs.
    Returns the name of the next node.

    Logic:
    - If max revisions reached → force report generation
    - If critique says PASS → generate report
    - If critique says REVISE → send back to analyst
    """
    revision_count = state.get("revision_count", 0)
    critique_text = state.get("critique", "")

    # Hard stop — prevent infinite loops
    if revision_count >= settings.max_revisions:
        logger.info(
            f"Max revisions ({settings.max_revisions}) reached — "
            f"forcing report generation"
        )
        return "report_generator"

    verdict = _parse_verdict(critique_text)

    if verdict == "REVISE":
        logger.info(f"Routing back to analyst for revision {revision_count + 1}")
        return "financial_analyst"

    logger.info("Critique passed — routing to report_generator")
    return "report_generator"