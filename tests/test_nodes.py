import pytest
from unittest.mock import patch, MagicMock
from financial_analyst.state import AnalystState
from financial_analyst.nodes.router import input_router, route_decision, _detect_input_type
from financial_analyst.nodes.analyst import _format_structured_data as fmt_analyst


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_state(**kwargs) -> AnalystState:
    """Creates a minimal valid state for testing."""
    base = {
        "user_input": "",
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
    base.update(kwargs)
    return base


# ---------------------------------------------------------------------------
# Router tests
# ---------------------------------------------------------------------------

class TestInputRouter:

    def test_detects_ticker(self):
        assert _detect_input_type("AAPL") == "ticker"
        assert _detect_input_type("MSFT") == "ticker"
        assert _detect_input_type("GOOGL") == "ticker"

    def test_detects_url(self):
        assert _detect_input_type("https://example.com") == "url"
        assert _detect_input_type("http://sec.gov/filing") == "url"

    def test_detects_pdf(self):
        assert _detect_input_type("report.pdf") == "pdf"
        assert _detect_input_type("annual_report.PDF") == "pdf"

    def test_defaults_to_text(self):
        assert _detect_input_type("Apple reported strong earnings") == "text"
        assert _detect_input_type("some random text here") == "text"

    def test_router_node_sets_input_type(self):
        state = make_state(user_input="AAPL")
        result = input_router(state)
        assert result["input_type"] == "ticker"
        assert result["errors"] == []

    def test_router_node_handles_empty_input(self):
        state = make_state(user_input="")
        result = input_router(state)
        assert result["input_type"] == "text"
        assert len(result["errors"]) > 0

    def test_router_node_handles_url(self):
        state = make_state(user_input="https://investor.apple.com/earnings")
        result = input_router(state)
        assert result["input_type"] == "url"

    def test_ticker_is_case_sensitive(self):
        # Lowercase should not be detected as ticker
        assert _detect_input_type("aapl") == "text"


class TestRouteDecision:

    def test_routes_pdf_to_doc_ingestion(self):
        state = make_state(input_type="pdf")
        assert route_decision(state) == "doc_ingestion"

    def test_routes_ticker_to_web_fetcher(self):
        state = make_state(input_type="ticker")
        assert route_decision(state) == "web_fetcher"

    def test_routes_url_to_web_fetcher(self):
        state = make_state(input_type="url")
        assert route_decision(state) == "web_fetcher"

    def test_routes_text_to_data_extractor(self):
        state = make_state(input_type="text")
        assert route_decision(state) == "data_extractor"


# ---------------------------------------------------------------------------
# Critique routing tests
# ---------------------------------------------------------------------------

class TestShouldRevise:

    def test_passes_when_verdict_is_pass(self):
        from financial_analyst.nodes.critique import should_revise
        state = make_state(
            critique="VERDICT: PASS\n\nAll good.",
            revision_count=0,
        )
        assert should_revise(state) == "report_generator"

    def test_revises_when_verdict_is_revise(self):
        from financial_analyst.nodes.critique import should_revise
        state = make_state(
            critique="VERDICT: REVISE\n\nNeeds work.",
            revision_count=0,
        )
        assert should_revise(state) == "financial_analyst"

    def test_forces_report_at_max_revisions(self):
        from financial_analyst.nodes.critique import should_revise
        state = make_state(
            critique="VERDICT: REVISE\n\nStill needs work.",
            revision_count=2,  # max_revisions = 2
        )
        assert should_revise(state) == "report_generator"

    def test_forces_report_beyond_max_revisions(self):
        from financial_analyst.nodes.critique import should_revise
        state = make_state(
            critique="VERDICT: REVISE",
            revision_count=5,
        )
        assert should_revise(state) == "report_generator"


# ---------------------------------------------------------------------------
# State accumulator tests
# ---------------------------------------------------------------------------

class TestStateAccumulators:

    def test_errors_accumulate(self):
        """Errors from multiple nodes should accumulate, not overwrite."""
        from operator import add
        errors_1 = ["error from node 1"]
        errors_2 = ["error from node 2"]
        combined = add(errors_1, errors_2)
        assert combined == ["error from node 1", "error from node 2"]

    def test_citations_accumulate(self):
        from operator import add
        c1 = ["source 1"]
        c2 = ["source 2"]
        assert add(c1, c2) == ["source 1", "source 2"]