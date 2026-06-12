# Financial Analyst Agent

A production-grade multi-agent AI system for automated investment research,
built with LangGraph. Input a stock ticker, URL, PDF filing, or raw text —
get a structured investment research report in return.

## Architecture

The agent is built as a stateful LangGraph pipeline with a reflection loop:

User Input

│

▼

Input Router (detects: ticker / url / pdf / text)

│

├──► Doc Ingestion (pdfplumber + PyMuPDF)

├──► Web Fetcher (Tavily + SEC EDGAR)

└──► ↓

│

▼

Data Extractor (Pydantic structured output)

│

▼

Financial Analyst (7-step CoT reasoning)

│

▼

Critique ──► REVISE ──► Financial Analyst (max 2 revisions)

│

PASS

│

▼

Report Generator

│

▼

Final Investment Report (Markdown)

## Key Features

- **Multi-agent orchestration** — 7 specialized nodes with conditional routing
- **Reflection loop** — LLM-as-critic reviews and improves the analysis
- **Structured extraction** — Pydantic schemas force typed financial data
- **Dual ingestion** — PDF parsing with table extraction + web search
- **Persistent memory** — MemorySaver checkpointing across nodes
- **LangSmith tracing** — Full observability on every node
- **Streamlit UI** — 4 input modes, metrics dashboard, downloadable reports
- **25 tests** — Unit and integration tests with pytest

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | LangGraph |
| LLM | OpenRouter (any model) |
| Web search | Tavily API |
| Financial data | SEC EDGAR API |
| PDF parsing | pdfplumber + PyMuPDF |
| Structured output | Pydantic v2 |
| Observability | LangSmith |
| UI | Streamlit |
| Testing | pytest |

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/financial-analyst-agent
cd financial-analyst-agent
```

**2. Create virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate  # Mac/Linux
pip install -e .
```

**3. Configure environment**
```bash
cp .env.example .env
# Fill in your API keys in .env
```

Get your keys:
- OpenRouter: https://openrouter.ai/keys
- Tavily: https://app.tavily.com
- LangSmith: https://smith.langchain.com (optional)

**4. Run the app**
```bash
streamlit run app.py
```

**5. Run tests**
```bash
pytest tests/ -v
```

## Usage

### Ticker Symbol
Enter `AAPL`, `MSFT`, `GOOGL` — fetches latest financials via Tavily + SEC EDGAR

### URL
Paste any earnings release or financial article URL

### PDF
Upload a 10-K, 10-Q, or annual report PDF

### Raw Text
Paste earnings call transcripts or press releases directly

## Project Structure 

src/financial_analyst/

├── config.py          # Pydantic-settings configuration

├── state.py           # AnalystState TypedDict

├── graph.py           # LangGraph StateGraph assembly

└── nodes/

├── router.py      # Input detection + conditional routing

├── ingestion.py   # PDF parsing + web fetching

├── extractor.py   # Structured financial data extraction

├── analyst.py     # 7-step chain-of-thought analysis

├── critique.py    # 6-dimension peer review + reflection loop

└── reporter.py    # Markdown report generation 

## Resume Bullet

> Built a multi-agent financial analyst using LangGraph — 7-node pipeline
> with reflection loop, Pydantic structured extraction, dual ingestion
> (PDF + SEC EDGAR/Tavily), LangSmith observability, and Streamlit UI.
> 25 pytest tests. Generates institutional-grade investment research reports.

## License

MIT