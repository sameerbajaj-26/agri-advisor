import asyncio
from pathlib import Path

import httpx
import pytest

from app.errors import MalformedPayload, SourceUnavailable
from app.providers.base import ProviderContext
from app.providers.geocode import GeocodeProvider
from app.providers.http import MinIntervalLimiter
from tests.helpers import FakeClock, ctx_for, geocode_payload, make_settings


def _provider(ctx: ProviderContext, **kw: str) -> GeocodeProvider:
    return GeocodeProvider(ctx, make_settings(**kw), MinIntervalLimiter(0.0))


def test_success_sends_identifying_user_agent(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        ua = req.headers["user-agent"]
        assert ua.startswith("agri-advisor/") and "ops@example.org" in ua
        assert req.url.params["countrycodes"] == "in"
        return httpx.Response(200, json=geocode_payload())

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            r = await _provider(ctx, contact_email="ops@example.org").search("  Mandya ")
            assert r.query == "Mandya" and r.matches[0].state == "Karnataka"
            assert r.matches[0].latitude == pytest.approx(12.5218)
            assert r.provenance.source == "OpenStreetMap Nominatim"
            again = await _provider(ctx).search("mandya")  # same query, any case -> cache
            assert again.provenance.cache_status == "cached"

    asyncio.run(run())


def test_timeout_malformed_stale(tmp_path: Path) -> None:
    async def run() -> None:
        def slow(_: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("slow")

        async with ctx_for(slow, tmp_path / "t") as ctx:
            with pytest.raises(SourceUnavailable) as err:
                await _provider(ctx).search("Mandya")
            assert err.value.kind == "timeout"

        bad = lambda _: httpx.Response(200, json={"error": "rate limited"})  # noqa: E731
        async with ctx_for(bad, tmp_path / "m") as ctx:
            with pytest.raises(MalformedPayload):
                await _provider(ctx).search("Mandya")

        state = {"down": False}

        def handler(_: httpx.Request) -> httpx.Response:
            if state["down"]:
                return httpx.Response(429, headers={"retry-after": "1"})
            return httpx.Response(200, json=geocode_payload())

        clock = FakeClock()
        async with ctx_for(handler, tmp_path / "s", clock) as ctx:
            p = _provider(ctx)
            first = await p.search("Mandya")
            state["down"] = True
            clock.advance(days=45)
            stale = await p.search("Mandya")
            assert stale.provenance.cache_status == "stale"
            assert stale.provenance.fetched_at == first.provenance.fetched_at
            assert stale.provenance.stale_reason is not None
            assert "rate_limited" in stale.provenance.stale_reason

    asyncio.run(run())
