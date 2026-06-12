import uuid
import logging
import streamlit as st
from pathlib import Path

from financial_analyst.graph import run_analysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Financial Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #0066cc;
    }
    .status-box {
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .error-box {
        background: #fff5f5;
        border-left: 4px solid #e53e3e;
        color: #c53030;
    }
    .success-box {
        background: #f0fff4;
        border-left: 4px solid #38a169;
        color: #276749;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #0066cc;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())[:8]
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("### Input Type")
    input_mode = st.radio(
        "Select how to provide financial data",
        ["Ticker Symbol", "URL", "PDF File", "Raw Text"],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Session")
    st.code(f"Thread ID: {st.session_state.thread_id}", language=None)
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())[:8]
        st.session_state.result = None
        st.rerun()

    st.divider()

    st.markdown("### Analysis History")
    if st.session_state.history:
        for i, item in enumerate(reversed(st.session_state.history[-5:])):
            st.markdown(f"**{i+1}.** `{item['input']}` — {item['status']}")
    else:
        st.caption("No analyses yet.")

    st.divider()
    st.markdown(
        "Built with [LangGraph](https://langchain-ai.github.io/langgraph/) "
        "+ [OpenRouter](https://openrouter.ai)"
    )


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.markdown('<p class="main-header">📊 Financial Analyst Agent</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Multi-agent AI system for investment research — '
    'powered by LangGraph</p>',
    unsafe_allow_html=True
)

# --- Input area ---
col1, col2 = st.columns([3, 1])

with col1:
    if input_mode == "Ticker Symbol":
        user_input = st.text_input(
            "Enter ticker symbol",
            placeholder="e.g. AAPL, MSFT, GOOGL, TSLA",
            help="Enter a stock ticker to fetch and analyze the latest financials",
        )

    elif input_mode == "URL":
        user_input = st.text_input(
            "Enter URL",
            placeholder="https://...",
            help="Paste a URL to an earnings release, filing, or financial article",
        )

    elif input_mode == "PDF File":
        uploaded = st.file_uploader(
            "Upload a financial document",
            type=["pdf"],
            help="Upload a 10-K, 10-Q, annual report, or earnings release PDF",
        )
        user_input = ""
        if uploaded:
            # Save uploaded file temporarily
            tmp_path = Path(f"tmp_{uploaded.name}")
            tmp_path.write_bytes(uploaded.read())
            user_input = str(tmp_path)
            st.caption(f"File saved temporarily: `{tmp_path}`")

    else:  # Raw Text
        user_input = st.text_area(
            "Paste financial text",
            placeholder="Paste earnings call transcript, press release, or any financial text...",
            height=150,
            help="Paste raw financial text to analyze directly",
        )

with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button(
        "🔍 Analyze",
        type="primary",
        use_container_width=True,
        disabled=not user_input,
    )


# ---------------------------------------------------------------------------
# Run analysis
# ---------------------------------------------------------------------------
if analyze_btn and user_input:
    with st.status("🤖 Running multi-agent analysis...", expanded=True) as status:

        st.write("📡 **Step 1/5** — Routing input and fetching data...")
        st.write("🔍 **Step 2/5** — Extracting structured financial data...")
        st.write("🧠 **Step 3/5** — Running financial analysis...")
        st.write("🔎 **Step 4/5** — Peer review & critique...")
        st.write("📝 **Step 5/5** — Generating final report...")

        try:
            result = run_analysis(
                user_input=user_input,
                thread_id=st.session_state.thread_id,
            )
            st.session_state.result = result

            # Save to history
            errors = result.get("errors", [])
            st.session_state.history.append({
                "input": user_input[:30],
                "status": "✅ Done" if not errors else "⚠️ Done with warnings",
            })

            status.update(
                label="✅ Analysis complete!",
                state="complete",
                expanded=False,
            )

        except Exception as e:
            status.update(label="❌ Analysis failed", state="error")
            st.error(f"Pipeline error: {str(e)}")
            st.session_state.history.append({
                "input": user_input[:30],
                "status": "❌ Failed",
            })


# ---------------------------------------------------------------------------
# Display results
# ---------------------------------------------------------------------------
if st.session_state.result:
    result = st.session_state.result
    structured = result.get("structured_data", {})
    errors = result.get("errors", [])
    report = result.get("final_report", "")
    revision_count = result.get("revision_count", 0)

    # --- Errors ---
    if errors:
        with st.expander("⚠️ Warnings / Errors", expanded=False):
            for err in errors:
                st.markdown(
                    f'<div class="status-box error-box">⚠️ {err}</div>',
                    unsafe_allow_html=True,
                )

    # --- Metrics row ---
    if structured:
        st.markdown("### 📈 Key Metrics")
        inc = structured.get("income_statement", {})
        bal = structured.get("balance_sheet", {})
        met = structured.get("key_metrics", {})

        m1, m2, m3, m4, m5, m6 = st.columns(6)

        with m1:
            rev = inc.get("revenue")
            st.metric("Revenue", f"${rev:,.0f}M" if rev else "N/A")
        with m2:
            growth = inc.get("revenue_growth_yoy")
            st.metric("Revenue Growth", f"{growth:.1f}%" if growth else "N/A")
        with m3:
            eps = inc.get("eps_diluted")
            st.metric("Diluted EPS", f"${eps:.2f}" if eps else "N/A")
        with m4:
            margin = inc.get("gross_margin")
            st.metric("Gross Margin", f"{margin:.1f}%" if margin else "N/A")
        with m5:
            fcf = met.get("free_cash_flow")
            st.metric("Free Cash Flow", f"${fcf:,.0f}M" if fcf else "N/A")
        with m6:
            de = bal.get("debt_to_equity")
            st.metric("Debt/Equity", f"{de:.2f}x" if de else "N/A")

        st.divider()

    # --- Report tabs ---
    tab1, tab2, tab3 = st.tabs(["📄 Full Report", "🔬 Raw Analysis", "🗂️ Structured Data"])

    with tab1:
        if report:
            st.markdown(report)
            st.divider()
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    label="⬇️ Download Report (Markdown)",
                    data=report,
                    file_name=f"analysis_{user_input[:10]}_{st.session_state.thread_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_b:
                st.caption(f"Revisions: {revision_count} | "
                          f"Thread: {st.session_state.thread_id}")
        else:
            st.warning("No report generated.")

    with tab2:
        raw_analysis = result.get("analysis", "")
        if raw_analysis:
            st.markdown(raw_analysis)
        else:
            st.info("No raw analysis available.")

        critique_text = result.get("critique", "")
        if critique_text:
            with st.expander("🔎 Critique feedback", expanded=False):
                st.text(critique_text)

    with tab3:
        if structured:
            st.json(structured)
        else:
            st.info("No structured data extracted.")