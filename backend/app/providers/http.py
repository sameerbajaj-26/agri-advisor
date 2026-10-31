"""Shared async HTTP: timeouts, retries with exponential backoff + jitter, rate limiting."""

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from app.errors import ErrorKind, MalformedPayload, SourceUnavailable

Sleep = Callable[[float], Awaitable[None]]


async def default_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


@dataclass(frozen=True)
class RetryPolicy:
    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0

    def delay_for(self, attempt: int) -> float:
        raw = min(self.max_delay, self.base_delay * (2.0 ** (attempt - 1)))
        return raw * random.uniform(0.5, 1.0)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


async def request_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    source: str,
    retry: RetryPolicy,
    sleep: Sleep,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float = 15.0,
    before_attempt: Callable[[], Awaitable[None]] | None = None,
) -> Any:
    """GET `url` and return parsed JSON, or raise SourceUnavailable / MalformedPayload.

    Retries timeouts, connection errors, 429 and 5xx. Other 4xx fail immediately.
    Error text never includes the URL (query strings may contain API keys).
    """
    attempts = max(1, retry.attempts)
    http_timeout = httpx.Timeout(timeout, connect=min(5.0, timeout))
    last: SourceUnavailable | None = None

    for attempt in range(1, attempts + 1):
        if before_attempt is not None:
            await before_attempt()
        wait_hint: float | None = None
        try:
            response = await client.get(url, params=params, headers=headers, timeout=http_timeout)
        except httpx.TimeoutException:
            last = SourceUnavailable(source, "timeout", f"no response within {timeout:.0f}s")
        except httpx.TransportError as exc:
            last = SourceUnavailable(source, "network", f"connection failed ({type(exc).__name__})")
        else:
            status = response.status_code
            if status == 429 or status >= 500:
                kind: ErrorKind = "rate_limited" if status == 429 else "http_error"
                last = SourceUnavailable(source, kind, f"HTTP {status}", status_code=status)
                wait_hint = _retry_after(response)
            elif status >= 400:
                hint = " - check the API key" if status in (401, 403) else ""
                raise SourceUnavailable(
                    source, "http_error", f"HTTP {status}{hint}", status_code=status
                )
            else:
                try:
                    return response.json()
                except ValueError as exc:
                    raise MalformedPayload(source, "response was not valid JSON") from exc

        if attempt < attempts:
            delay = retry.delay_for(attempt)
            if wait_hint is not None:
                delay = max(delay, min(wait_hint, 30.0))
            await sleep(delay)

    assert last is not None
    raise last


class MinIntervalLimiter:
    """Ensures at least `min_interval` seconds between calls (single process)."""

    def __init__(
        self,
        min_interval: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Sleep = default_sleep,
    ) -> None:
        self._interval = min_interval
        self._monotonic = monotonic
        self._sleep = sleep
        self._next_allowed = float("-inf")
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            delay = self._next_allowed - self._monotonic()
            if delay > 0:
                await self._sleep(delay)
            self._next_allowed = self._monotonic() + self._interval
