from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # APPLICATION

    app_name: str = "Phoenix AI"
    app_env: str = "development"
    debug: bool = True
    log_level: str = "INFO"

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    # OLLAMA
    llm_provider: str = "ollama"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:4b-instruct"

    # GEMINI
    gemini_api_key: str | None = Field(
        default=None,
        alias="GEMINI_API_KEY",
    )

    gemini_model: str = "gemini-3.1-pro-preview"

    # DEFAULT ROUTING
    default_provider: str = "ollama"
    default_model: str = "qwen3:4b-instruct"

    temperature: float = 0.2
    max_tokens: int = 4096

    # RESEARCH
    research_search_endpoint: str = "https://html.duckduckgo.com/html/"
    research_max_sources: int = 6
    research_timeout_seconds: float = 15.0

    # SECURITY
    require_approval_for_tools: bool = True
    max_tool_calls_per_task: int = 20

    # INFRASTRUCTURE
    database_url: str = (
        "postgresql+psycopg://phoenix:phoenix@localhost:5432/phoenix"
    )

    redis_url: str = "redis://localhost:6379/0"

    # PYDANTIC SETTINGS
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()