"""Cache of REAL upstream responses (parsed + validated) with timestamps.

Entries live physically for the provider's stale window; freshness is decided by the caller.
Redis can be added later behind the same `Cache` protocol.
"""

import asyncio
from datetime import datetime
from typing import Any, Protocol

import diskcache
from pydantic import BaseModel, ValidationError


class CacheEntry(BaseModel):
    value: dict[str, Any]
    stored_at: datetime


class Cache(Protocol):
    async def get(self, key: str) -> CacheEntry | None: ...

    async def set(self, key: str, entry: CacheEntry, expire_s: float) -> None: ...


class DiskCache:
    def __init__(self, directory: str) -> None:
        self._cache = diskcache.Cache(directory)

    async def get(self, key: str) -> CacheEntry | None:
        raw = await asyncio.to_thread(self._cache.get, key)
        if raw is None:
            return None
        try:
            return CacheEntry.model_validate(raw)
        except ValidationError:
            return None

    async def set(self, key: str, entry: CacheEntry, expire_s: float) -> None:
        await asyncio.to_thread(
            self._cache.set, key, entry.model_dump(mode="json"), expire=expire_s
        )

    def close(self) -> None:
        self._cache.close()
