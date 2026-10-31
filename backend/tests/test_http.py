import asyncio

import httpx
import pytest

from app.errors import MalformedPayload, SourceUnavailable
from app.providers.http import MinIntervalLimiter, RetryPolicy, request_json
from tests.helpers import no_sleep


def _call(handler: httpx.MockTransport, attempts: int = 3) -> object:
    async def run() -> object:
        async with httpx.AsyncClient(transport=handler) as client:
            return await request_json(
                client,
                "https://example.test/x",
                source="T",
                retry=RetryPolicy(attempts=attempts, base_delay=0.0),
                sleep=no_sleep,
            )

    return asyncio.run(run())


def test_retries_5xx_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"ok": 1})

    assert _call(httpx.MockTransport(handler)) == {"ok": 1}
    assert calls["n"] == 3


def test_4xx_not_retried_and_hides_url() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(403)

    with pytest.raises(SourceUnavailable) as err:
        _call(httpx.MockTransport(handler))
    assert calls["n"] == 1 and err.value.status_code == 403
    assert "example.test" not in str(err.value)


def test_non_json_is_malformed() -> None:
    with pytest.raises(MalformedPayload):
        _call(httpx.MockTransport(lambda _: httpx.Response(200, text="<html>oops</html>")))


def test_limiter_spaces_calls() -> None:
    sleeps: list[float] = []
    t = [0.0]

    async def fake_sleep(s: float) -> None:
        sleeps.append(s)
        t[0] += s

    limiter = MinIntervalLimiter(1.0, monotonic=lambda: t[0], sleep=fake_sleep)

    async def run() -> None:
        await limiter.wait()
        await limiter.wait()

    asyncio.run(run())
    assert sleeps == [1.0]
