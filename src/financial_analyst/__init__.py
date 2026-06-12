import logging
import os
from financial_analyst.config import settings

# ---------------------------------------------------------------------------
# Logging — structured, level-aware
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# LangSmith tracing — enabled via .env
# ---------------------------------------------------------------------------
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    logging.getLogger(__name__).info(
        f"LangSmith tracing enabled — project: {settings.langsmith_project}"
    )