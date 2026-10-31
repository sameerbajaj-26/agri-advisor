"""Provenance, provider context and the cache-through helper shared by all providers."""

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.cache import Cache, CacheEntry
from app.errors import SourceUnavailable
from app.providers.http import RetryPolicy, Sleep, default_sleep

Clock = Callable[[], datetime]

# live   = fetched from the upstream source during this request
# cached = real upstream response still inside its TTL (see fetched_at)
# stale  = upstream failed; an OLDER real response is served, labelled as such
CacheStatus = Literal["live", "cached", "stale"]


def utcnow() -> datetime:
    return datetime.now(UTC)


class Provenance(BaseModel):
    source: str
    source_url: str
    fetched_at: datetime
    cache_status: CacheStatus
    stale_reason: str | None = None


class Sourced(BaseModel):
    provenance: Provenance


@dataclass
class ProviderContext:
    client: httpx.AsyncClient
    cache: Cache
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    clock: Clock = utcnow
    sleep: Sleep = default_sleep


T = TypeVar("T", bound=Sourced)


def cache_key(prefix: str, *parts: Any) -> str:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:24]
    return f"{prefix}:{digest}"


def _always(_: Any) -> bool:
    return True


async def _load(ctx: ProviderContext, key: str, model: type[T]) -> tuple[T, datetime] | None:
    entry = await ctx.cache.get(key)
    if entry is None:
        return None
    try:
        return model.model_validate(entry.value), entry.stored_at
    except ValidationError:
        return None


async def cached_fetch(
    ctx: ProviderContext,
    *,
    key: str,
    ttl: timedelta,
    stale_max: timedelta,
    fetch: Callable[[], Awaitable[T]],
    model: type[T],
    cacheable: Callable[[T], bool] = _always,
) -> T:
    """Fresh cache hit -> 'cached'. Else fetch live. If live fails and an older REAL
    response exists (within stale_max) -> 'stale'. Otherwise re-raise. Only successful,
    validated responses are ever written to the cache."""
    now = ctx.clock()
    loaded = await _load(ctx, key, model)
    if loaded is not None and now - loaded[1] < ttl:
        obj = loaded[0]
        obj.provenance.cache_status = "cached"
        return obj

    try:
        result = await fetch()
    except SourceUnavailable as exc:
        if loaded is not None and now - loaded[1] <= stale_max:
            obj = loaded[0]
            obj.provenance.cache_status = "stale"
            obj.provenance.stale_reason = f"{exc.kind}: {exc.reason}"
            return obj
        raise

    if cacheable(result):
        await ctx.cache.set(
            key,
            CacheEntry(
                value=result.model_dump(mode="json"), stored_at=result.provenance.fetched_at
            ),
            stale_max.total_seconds(),
        )
    return result
