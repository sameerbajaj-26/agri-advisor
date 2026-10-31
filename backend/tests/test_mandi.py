import asyncio
from pathlib import Path

import httpx
import pytest

from app.errors import MalformedPayload, NotConfigured, SourceUnavailable
from app.providers.mandi import MandiProvider
from tests.helpers import FakeClock, ctx_for, make_settings, mandi_row

KEYED = {"data_gov_api_key": "test-key"}


def test_success_filters_and_missing_days(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        q = req.url.params
        assert q["api-key"] == "test-key"
        assert q["filters[state.keyword]"] == "Karnataka"  # 'karnataka' title-cased
        assert q["filters[commodity]"] == "Tomato"
        body = {"total": 2, "count": 2, "records": [mandi_row(), mandi_row("Maddur", "27/10/2026")]}
        return httpx.Response(200, json=body)

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            p = MandiProvider(ctx, make_settings(**KEYED))
            r = await p.get_prices(state="karnataka", commodity="tomato")
            assert r.provenance.cache_status == "live"
            assert r.records[0].modal_price == 1500.0
            assert (
                r.latest_arrival_date is not None
                and r.latest_arrival_date.isoformat() == "2026-10-28"
            )
            assert r.days_since_latest == 3  # clock is 2026-10-31: three days without a report
            assert len(r.dates_present) == 2

    asyncio.run(run())


def test_legacy_capitalised_field_names(tmp_path: Path) -> None:
    legacy = {
        "State": "Karnataka",
        "District": "Mandya",
        "Market": "Mandya",
        "Commodity": "Onion",
        "Variety": "Red",
        "Grade": "FAQ",
        "Arrival_Date": "30/10/2026",
        "Min_x0020_Price": "900",
        "Max_x0020_Price": "1400",
        "Modal_x0020_Price": "1200",
    }

    async def run() -> None:
        handler = lambda _: httpx.Response(200, json={"total": 1, "records": [legacy]})  # noqa: E731
        async with ctx_for(handler, tmp_path) as ctx:
            r = await MandiProvider(ctx, make_settings(**KEYED)).get_prices(state="Karnataka")
            assert r.records[0].commodity == "Onion" and r.records[0].modal_price == 1200.0

    asyncio.run(run())


def test_pagination_and_skipped_rows(tmp_path: Path) -> None:
    pages = {
        "0": [mandi_row("A"), mandi_row("B")],
        "2": [mandi_row("C", modal_price="NA")],  # unusable row
    }

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"total": 3, "records": pages.get(req.url.params["offset"], [])}
        )

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            p = MandiProvider(ctx, make_settings(**KEYED), page_size=2)
            r = await p.get_prices(state="Karnataka", max_records=10)
            assert len(r.records) == 2 and r.skipped_records == 1 and not r.truncated

    asyncio.run(run())


def test_missing_key_is_explicit_error(tmp_path: Path) -> None:
    async def run() -> None:
        async with ctx_for(lambda _: httpx.Response(200, json={}), tmp_path) as ctx:
            with pytest.raises(NotConfigured):
                await MandiProvider(ctx, make_settings()).get_prices(state="Karnataka")

    asyncio.run(run())


def test_timeout_and_malformed(tmp_path: Path) -> None:
    async def run() -> None:
        def slow(_: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("slow")

        async with ctx_for(slow, tmp_path) as ctx:
            with pytest.raises(SourceUnavailable) as err:
                await MandiProvider(ctx, make_settings(**KEYED)).get_prices(state="Karnataka")
            assert err.value.kind == "timeout"

        bad = lambda _: httpx.Response(200, json={"status": "error", "message": "bad key"})  # noqa: E731
        async with ctx_for(bad, tmp_path / "b") as ctx:
            with pytest.raises(MalformedPayload):
                await MandiProvider(ctx, make_settings(**KEYED)).get_prices(state="Karnataka")

    asyncio.run(run())


def test_stale_and_empty_not_cached(tmp_path: Path) -> None:
    state = {"mode": "ok"}

    def handler(_: httpx.Request) -> httpx.Response:
        if state["mode"] == "down":
            return httpx.Response(502)
        recs = [mandi_row()] if state["mode"] == "ok" else []
        return httpx.Response(200, json={"total": len(recs), "records": recs})

    async def run() -> None:
        clock = FakeClock()
        async with ctx_for(handler, tmp_path, clock) as ctx:
            p = MandiProvider(ctx, make_settings(**KEYED))
            state["mode"] = "empty"
            assert (await p.get_prices(state="Karnataka")).records == []
            state["mode"] = "down"  # empty answer must not have been cached as 'stale'
            with pytest.raises(SourceUnavailable):
                await p.get_prices(state="Karnataka")
            state["mode"] = "ok"
            first = await p.get_prices(state="Karnataka")
            state["mode"] = "down"
            clock.advance(hours=5)
            stale = await p.get_prices(state="Karnataka")
            assert stale.provenance.cache_status == "stale"
            assert stale.provenance.fetched_at == first.provenance.fetched_at

    asyncio.run(run())
