import asyncio
from pathlib import Path

import httpx
import pytest

from app.errors import MalformedPayload, SourceUnavailable
from app.providers.soil import SoilOverride, SoilProvider
from tests.helpers import FakeClock, ctx_for, make_settings, soil_payload


def test_success_scales_units_and_reports_missing(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.params.get_list("property")[0] == "phh2o"
        assert req.url.params["value"] == "mean"
        return httpx.Response(200, json=soil_payload())

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            prof = await SoilProvider(ctx, make_settings()).get_profile(12.52, 76.9)
            by = {p.name: p for p in prof.properties}
            assert by["phh2o"].value == pytest.approx(6.2)
            assert by["clay"].value == pytest.approx(25.3) and by["clay"].unit == "%"
            assert all(p.origin == "soilgrids" for p in prof.properties)
            assert "nitrogen@0-5cm" in prof.missing and "nitrogen" not in by
            assert prof.soilgrids is not None and prof.soilgrids.cache_status == "live"

    asyncio.run(run())


def test_user_override_is_labelled_and_replaces(tmp_path: Path) -> None:
    async def run() -> None:
        async with ctx_for(lambda _: httpx.Response(200, json=soil_payload()), tmp_path) as ctx:
            prof = await SoilProvider(ctx, make_settings()).get_profile(
                12.52, 76.9, SoilOverride(ph=7.4)
            )
            ph = [p for p in prof.properties if p.name == "phh2o"]
            assert len(ph) == 1 and ph[0].origin == "user-provided" and ph[0].value == 7.4
            assert prof.user_override_applied

    asyncio.run(run())


def test_override_survives_outage_but_nothing_else_does(tmp_path: Path) -> None:
    def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    async def run() -> None:
        async with ctx_for(down, tmp_path) as ctx:
            p = SoilProvider(ctx, make_settings())
            prof = await p.get_profile(12.52, 76.9, SoilOverride(ph=6.8))
            assert [x.origin for x in prof.properties] == ["user-provided"]
            assert prof.soilgrids is None and prof.soilgrids_error is not None
            with pytest.raises(SourceUnavailable):  # no override -> explicit error, no numbers
                await p.get_profile(12.52, 76.9)

    asyncio.run(run())


def test_malformed_and_stale(tmp_path: Path) -> None:
    async def run() -> None:
        async with ctx_for(lambda _: httpx.Response(200, json={"oops": 1}), tmp_path / "m") as ctx:
            with pytest.raises(MalformedPayload):
                await SoilProvider(ctx, make_settings()).fetch_soilgrids(12.52, 76.9)

        state = {"down": False}

        def handler(_: httpx.Request) -> httpx.Response:
            if state["down"]:
                return httpx.Response(500)
            return httpx.Response(200, json=soil_payload())

        clock = FakeClock()
        async with ctx_for(handler, tmp_path / "s", clock) as ctx:
            p = SoilProvider(ctx, make_settings())
            first = await p.fetch_soilgrids(12.52, 76.9)
            state["down"] = True
            clock.advance(days=40)  # beyond 30-day TTL, inside 1-year stale window
            stale = await p.fetch_soilgrids(12.52, 76.9)
            assert stale.provenance.cache_status == "stale"
            assert stale.provenance.fetched_at == first.provenance.fetched_at

    asyncio.run(run())
