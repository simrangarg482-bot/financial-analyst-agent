from typing import TypedDict, Annotated
from operator import add


class AnalystState(TypedDict):
    # --- Input ---
    user_input: str              # raw string from the user
    input_type: str              # "pdf" | "ticker" | "url" | "text"

    # --- Ingestion ---
    raw_content: str             # full text extracted from source

    # --- Extraction ---
    structured_data: dict        # revenue, EPS, ratios, guidance, etc.

    # --- Analysis loop ---
    analysis: str                # financial analyst node output
    critique: str                # critique node feedback
    revision_count: int          # tracks how many revisions done

    # --- Output ---
    final_report: str            # formatted markdown report
    citations: Annotated[list[str], add]  # accumulates across nodes
    errors: Annotated[list[str], add]     # non-fatal errors accumulate