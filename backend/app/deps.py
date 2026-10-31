"""FastAPI dependencies: shared HTTP client/cache live on app.state (set in lifespan)."""

from collections.abc import Callable
from datetime import datetime
from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.llm.base import LLMClient, build_llm
from app.providers.base import ProviderContext, utcnow
from app.providers.geocode import GeocodeProvider
from app.providers.http import RetryPolicy
from app.providers.mandi import MandiProvider
from app.providers.soil import SoilProvider
from app.providers.weather import WeatherProvider
from app.rag.retriever import Retriever
from app.rag.store import drop_client_cache, kb_version


def get_app_settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def get_context(request: Request, settings: SettingsDep) -> ProviderContext:
    state = request.app.state
    return ProviderContext(
        client=state.http,
        cache=state.cache,
        retry=RetryPolicy(attempts=settings.http_retries),
    )


CtxDep = Annotated[ProviderContext, Depends(get_context)]


def get_geocoder(request: Request, ctx: CtxDep, settings: SettingsDep) -> GeocodeProvider:
    return GeocodeProvider(ctx, settings, request.app.state.nominatim_limiter)


def get_weather(ctx: CtxDep, settings: SettingsDep) -> WeatherProvider:
    return WeatherProvider(ctx, settings)


def get_mandi(ctx: CtxDep, settings: SettingsDep) -> MandiProvider:
    return MandiProvider(ctx, settings)


def get_soil(ctx: CtxDep, settings: SettingsDep) -> SoilProvider:
    return SoilProvider(ctx, settings)


def get_llm(request: Request, settings: SettingsDep) -> LLMClient:
    return build_llm(settings, request.app.state.http)


def get_clock() -> datetime:
    return utcnow()


_RETRIEVER: tuple[tuple[str, str, str, int], Retriever] | None = None


def get_retriever(settings: SettingsDep) -> Retriever:
    """Opened lazily (API starts without an ingested KB; /kb/search then answers 503) and
    reopened automatically after scripts/ingest_docs.py writes, so no API restart is needed."""
    global _RETRIEVER
    key = (settings.chroma_dir, settings.kb_collection, settings.embedding_model,
           kb_version(settings))  # fmt: skip
    if _RETRIEVER is None or _RETRIEVER[0] != key:
        if _RETRIEVER is not None:
            drop_client_cache()
        _RETRIEVER = (key, Retriever.from_settings(settings))
    return _RETRIEVER[1]


def get_retriever_factory(settings: SettingsDep) -> Callable[[], Retriever]:
    """Lazy: the agent only touches the KB if the LLM calls search_knowledge, and a missing KB
    then becomes a tool error the LLM must report, not a failed request."""
    return lambda: get_retriever(settings)
