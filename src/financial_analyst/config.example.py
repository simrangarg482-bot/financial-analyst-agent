from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    openrouter_api_key: str = "your-openrouter-api-key-here"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "meta-llama/llama-3.1-8b-instruct"
    temperature: float = 0.1
    tavily_api_key: str = "your-tavily-api-key-here"
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "financial-analyst-agent"
    max_revisions: int = 2
    extraction_timeout: int = 60


settings = Settings()