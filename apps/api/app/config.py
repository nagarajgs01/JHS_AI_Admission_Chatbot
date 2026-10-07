from functools import lru_cache
from os import getenv

from pydantic import BaseModel


class Settings(BaseModel):
    app_env: str = getenv("APP_ENV", "development")
    database_url: str = getenv(
        "DATABASE_URL",
        "postgresql://admissions:admissions@localhost:5432/admissions",
    )
    llm_provider: str = getenv("LLM_PROVIDER", "mock")
    llm_base_url: str = getenv("LLM_BASE_URL", "http://localhost:11434")
    llm_model: str = getenv("LLM_MODEL", "qwen3:4b")
    llm_timeout_seconds: float = float(getenv("LLM_TIMEOUT_SECONDS", "90"))
    knowledge_provider: str = getenv("KNOWLEDGE_PROVIDER", "postgres")
    embedding_provider: str = getenv("EMBEDDING_PROVIDER", "sentence-transformers")
    embedding_model: str = getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    answer_min_score: float = float(getenv("ANSWER_MIN_SCORE", "0.38"))
    allowed_origins: str = getenv("ALLOWED_ORIGINS", "http://localhost:5173")
    admin_api_key: str = getenv("ADMIN_API_KEY", "change-me-local-only")


@lru_cache
def get_settings() -> Settings:
    return Settings()
