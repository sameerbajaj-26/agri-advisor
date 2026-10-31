import pytest
from pydantic import ValidationError

from app.config import Settings


def _settings() -> Settings:
    return Settings(_env_file=None)  # type: ignore[call-arg]


def test_defaults_when_env_file_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    settings = _settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.llm_provider == "ollama"
    assert settings.data_gov_api_key is None


def test_invalid_llm_provider_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "bogus")
    with pytest.raises(ValidationError):
        _settings(_env_file=None)  # type: ignore[call-arg]


def test_database_url_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValidationError):
        _settings(_env_file=None)  # type: ignore[call-arg]
