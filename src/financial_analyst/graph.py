import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from financial_analyst.state import AnalystState
from financial_analyst.nodes import (
    input_router,
    route_decision,
    doc_ingestion,
    web_fetcher,
    data_extractor,
    financial_analyst,
    critique,
    should_revise,
    report_generator,
)

logger = logging.getLogger(__name__)


def build_graph(checkpointer=None):
    graph = StateGraph(AnalystState)

    graph.add_node("input_router",      input_router)
    graph.add_node("doc_ingestion",     doc_ingestion)
    graph.add_node("web_fetcher",       web_fetcher)
    graph.add_node("data_extractor",    data_extractor)
    graph.add_node("financial_analyst", financial_analyst)
    graph.add_node("critique",          critique)
    graph.add_node("report_generator",  report_generator)

    graph.set_entry_point("input_router")

    graph.add_conditional_edges(
        "input_router",
        route_decision,
        {
            "doc_ingestion":  "doc_ingestion",
            "web_fetcher":    "web_fetcher",
            "data_extractor": "data_extractor",
        }
    )

    graph.add_edge("doc_ingestion",     "data_extractor")
    graph.add_edge("web_fetcher",       "data_extractor")
    graph.add_edge("data_extractor",    "financial_analyst")
    graph.add_edge("financial_analyst", "critique")

    graph.add_conditional_edges(
        "critique",
        should_revise,
        {
            "financial_analyst": "financial_analyst",
            "report_generator":  "report_generator",
        }
    )

    graph.add_edge("report_generator", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Graph compiled successfully")
    return compiled


def create_app():
    """
    Creates the production app with in-memory checkpointing.
    MemorySaver persists state across nodes within a single run.
    """
    checkpointer = MemorySaver()
    return build_graph(checkpointer=checkpointer)


def run_analysis(
    user_input: str,
    thread_id: str = "default",
) -> dict:
    """
    End-to-end runner — takes user input, returns final state.

    Args:
        user_input: Ticker symbol, URL, PDF path, or raw text
        thread_id:  Unique ID for this analysis session.

    Returns:
        Final state dict with final_report, structured_data,
        analysis, citations, and errors.
    """
    app = create_app()

    initial_state: AnalystState = {
        "user_input":      user_input,
        "input_type":      "",
        "raw_content":     "",
        "structured_data": {},
        "analysis":        "",
        "critique":        "",
        "revision_count":  0,
        "final_report":    "",
        "citations":       [],
        "errors":          [],
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    logger.info(f"Starting analysis: {user_input} (thread: {thread_id})")

    try:
        final_state = app.invoke(initial_state, config=config)
        logger.info("Pipeline completed successfully")
        return final_state

    except Exception as e:
        logger.error(f"Graph execution failed: {e}")
        raise