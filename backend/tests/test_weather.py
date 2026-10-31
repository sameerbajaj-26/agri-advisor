import asyncio
from pathlib import Path

import httpx
import pytest

from app.errors import MalformedPayload, SourceUnavailable
from app.providers.weather import WeatherProvider
from tests.helpers import FakeClock, ctx_for, make_settings, weather_payload


def test_success_cached_and_params(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        assert req.url.params["forecast_days"] == "7"
        assert req.url.params["timezone"] == "auto"
        assert req.url.params["latitude"] == "12.97"  # snapped to 0.01
        return httpx.Response(200, json=weather_payload())

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            p = WeatherProvider(ctx, make_settings())
            first = await p.get_forecast(12.9716, 77.5946)
            assert first.provenance.cache_status == "live"
            assert first.provenance.source == "Open-Meteo"
            assert len(first.daily) == 2 and first.daily[1].precipitation_mm == 12.5
            assert first.hourly[2].temperature_c is None  # null preserved, not invented
            second = await p.get_forecast(12.9716, 77.5946)
            assert second.provenance.cache_status == "cached"
            assert second.provenance.fetched_at == first.provenance.fetched_at

    asyncio.run(run())
    assert calls["n"] == 1


def test_timeout_raises_after_retries(tmp_path: Path) -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        raise httpx.ReadTimeout("slow")

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            with pytest.raises(SourceUnavailable) as err:
                await WeatherProvider(ctx, make_settings()).get_forecast(12.97, 77.59)
            assert err.value.kind == "timeout"

    asyncio.run(run())
    assert calls["n"] == 2  # RetryPolicy(attempts=2) in test ctx


@pytest.mark.parametrize("mutate", ["drop_daily", "short_column", "bad_value"])
def test_malformed_payload(tmp_path: Path, mutate: str) -> None:
    body = weather_payload()
    if mutate == "drop_daily":
        del body["daily"]
    elif mutate == "short_column":
        body["daily"]["precipitation_sum"] = [1.0]
    else:
        body["daily"]["precipitation_sum"] = ["lots", "none"]

    async def run() -> None:
        async with ctx_for(lambda _: httpx.Response(200, json=body), tmp_path) as ctx:
            with pytest.raises(MalformedPayload):
                await WeatherProvider(ctx, make_settings()).get_forecast(12.97, 77.59)

    asyncio.run(run())


def test_stale_served_when_source_down_then_expires(tmp_path: Path) -> None:
    state = {"down": False}

    def handler(_: httpx.Request) -> httpx.Response:
        if state["down"]:
            raise httpx.ConnectError("no route")
        return httpx.Response(200, json=weather_payload())

    async def run() -> None:
        clock = FakeClock()
        async with ctx_for(handler, tmp_path, clock) as ctx:
            p = WeatherProvider(ctx, make_settings())
            first = await p.get_forecast(12.97, 77.59)
            state["down"] = True
            clock.advance(hours=2)  # past 30 min TTL, inside 12 h stale window
            stale = await p.get_forecast(12.97, 77.59)
            assert stale.provenance.cache_status == "stale"
            assert stale.provenance.fetched_at == first.provenance.fetched_at
            assert stale.provenance.stale_reason is not None
            assert "network" in stale.provenance.stale_reason
            clock.advance(hours=11)  # now older than 12 h: refuse rather than mislead
            with pytest.raises(SourceUnavailable):
                await p.get_forecast(12.97, 77.59)

    asyncio.run(run())


def test_failures_are_never_cached(tmp_path: Path) -> None:
    state = {"down": True}

    def handler(_: httpx.Request) -> httpx.Response:
        if state["down"]:
            return httpx.Response(500)
        return httpx.Response(200, json=weather_payload())

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            p = WeatherProvider(ctx, make_settings())
            with pytest.raises(SourceUnavailable):
                await p.get_forecast(12.97, 77.59)
            state["down"] = False
            assert (await p.get_forecast(12.97, 77.59)).provenance.cache_status == "live"

    asyncio.run(run())
