import pytest
from unittest.mock import patch, MagicMock
from financial_analyst.graph import build_graph
from financial_analyst.state import AnalystState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_initial_state(user_input: str) -> AnalystState:
    return {
        "user_input": user_input,
        "input_type": "",
        "raw_content": "",
        "structured_data": {},
        "analysis": "",
        "critique": "",
        "revision_count": 0,
        "final_report": "",
        "citations": [],
        "errors": [],
    }


# ---------------------------------------------------------------------------
# Graph compilation tests
# ---------------------------------------------------------------------------

class TestGraphCompilation:

    def test_graph_compiles_without_checkpointer(self):
        graph = build_graph(checkpointer=None)
        assert graph is not None

    def test_graph_compiles_with_memory_saver(self):
        from langgraph.checkpoint.memory import MemorySaver
        graph = build_graph(checkpointer=MemorySaver())
        assert graph is not None

    def test_graph_is_valid_compiled_graph(self):
        from langgraph.graph.state import CompiledStateGraph
        graph = build_graph(checkpointer=None)
        assert isinstance(graph, CompiledStateGraph)


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------

class TestGraphExecution:

    def test_text_input_routes_to_extractor(self):
        from financial_analyst.nodes.router import input_router, route_decision
        state = make_initial_state("Apple reported $94B revenue in Q4 2024")
        router_result = input_router(state)
        assert router_result["input_type"] == "text"
        state.update(router_result)
        assert route_decision(state) == "data_extractor"

    def test_ticker_input_routes_to_web_fetcher(self):
        from financial_analyst.nodes.router import input_router, route_decision
        state = make_initial_state("TSLA")
        router_result = input_router(state)
        assert router_result["input_type"] == "ticker"
        state.update(router_result)
        assert route_decision(state) == "web_fetcher"

    def test_url_input_routes_to_web_fetcher(self):
        from financial_analyst.nodes.router import input_router, route_decision
        state = make_initial_state("https://investor.apple.com/earnings")
        router_result = input_router(state)
        assert router_result["input_type"] == "url"
        state.update(router_result)
        assert route_decision(state) == "web_fetcher"

    def test_initial_state_is_valid(self):
        state = make_initial_state("AAPL")
        assert state["user_input"] == "AAPL"
        assert state["revision_count"] == 0
        assert state["citations"] == []
        assert state["errors"] == []