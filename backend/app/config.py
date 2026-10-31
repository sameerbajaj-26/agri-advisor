"""Application settings, read from environment / .env (never hardcode secrets)."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # backend/.env when run from backend/, repo-root .env otherwise
        env_file=(".env", "../.env"),
        extra="ignore",
    )

    # App
    app_env: Literal["development", "staging", "production"] = "development"
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000"]

    # Infrastructure (database_url has no default, so a missing .env fails loudly)
    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = "redis://localhost:6379/0"
    cache_dir: str = "./data/cache"

    # HTTP behaviour for live providers
    http_retries: int = 3
    http_timeout_s: float = 15.0
    slow_http_timeout_s: float = 30.0  # data.gov.in and SoilGrids can be slow
    contact_email: str | None = None  # included in the Nominatim User-Agent (recommended)

    # Live data sources (base URLs verified against provider docs in Milestone 1)
    nominatim_base_url: str = "https://nominatim.openstreetmap.org"
    open_meteo_base_url: str = "https://api.open-meteo.com/v1/forecast"
    soilgrids_base_url: str = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    data_gov_base_url: str = "https://api.data.gov.in/resource"
    mandi_resource_id: str = "9ef84268-d588-465a-a308-a864a43d0070"
    data_gov_user_agent: str = "agri-advisor/0.1"

    # Government data
    data_gov_api_key: SecretStr | None = None

    # LLM
    llm_provider: Literal["ollama", "hosted"] = "ollama"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str | None = None
    hosted_llm_base_url: str | None = None
    hosted_llm_api_key: SecretStr | None = None
    hosted_llm_model: str | None = None

    # Translation / speech
    translation_provider: Literal["indictrans2", "bhashini", "sarvam"] = "indictrans2"
    bhashini_user_id: str | None = None
    bhashini_api_key: SecretStr | None = None
    sarvam_api_key: SecretStr | None = None
    whisper_model_size: str = "small"

    # Agent (Milestone 3)
    llm_timeout_s: float = 180.0
    agent_max_iterations: int = 6  # tool-calling rounds before we force the final answer
    agent_json_retries: int = 2  # re-asks when the structured answer is malformed/ungrounded
    tool_max_price_rows: int = 20
    kb_top_k: int = 5

    # Safety guards (deterministic alerts, app/agent/guards.py). NO built-in defaults: the
    # numbers are policy, not code. Unset => /ask answers 503 instead of guessing a threshold.
    guard_window_hours: int = 48
    guard_rain_total_mm_48h: float | None = None
    guard_rain_prob_pct: float | None = None
    guard_heat_max_c: float | None = None
    guard_frost_min_c: float | None = None

    # RAG
    # Relative paths resolve against backend/ so the API and scripts/ingest_docs.py agree
    # no matter which directory they are started from.
    chroma_dir: str = "./data/chroma"
    kb_raw_dir: str = "./data/kb_raw"  # downloaded source documents (kept for re-ingestion)
    kb_collection: str = "crop_knowledge"
    # sentence-transformers model. Changing it requires `--rebuild` (vector sizes differ).
    # e5 models get "query: "/"passage: " prefixes automatically; BAAI/bge-m3 works too.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL

    @field_validator(
        "guard_rain_total_mm_48h",
        "guard_rain_prob_pct",
        "guard_heat_max_c",
        "guard_frost_min_c",
        mode="before",
    )
    @classmethod
    def _blank_guard_means_unset(cls, v: object) -> object:
        return None if v is None or str(v).strip() == "" else v

    @field_validator("embedding_model", mode="before")
    @classmethod
    def _blank_embedding_model_means_default(cls, v: object) -> object:
        return DEFAULT_EMBEDDING_MODEL if v is None or not str(v).strip() else v

    @property
    def chroma_path(self) -> Path:
        return _resolve(self.chroma_dir)

    @property
    def kb_raw_path(self) -> Path:
        return _resolve(self.kb_raw_dir)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (BACKEND_DIR / path).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
