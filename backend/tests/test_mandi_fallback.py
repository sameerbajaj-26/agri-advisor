import asyncio
from pathlib import Path

import httpx

from app.providers.mandi import MandiProvider
from tests.helpers import ctx_for, make_settings, mandi_row

KEYED = {"data_gov_api_key": "test-key"}


def test_empty_district_falls_back_to_state_and_says_so(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if "filters[district]" in req.url.params:  # the district has no prices today
            return httpx.Response(200, json={"total": 0, "records": []})
        return httpx.Response(200, json={"total": 1, "records": [mandi_row("Chintamani")]})

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            p = MandiProvider(ctx, make_settings(**KEYED))
            r = await p.get_prices_with_fallback(
                state="Karnataka", district="Mandya", commodity="Tomato"
            )
            assert r.scope == "state_fallback"
            assert r.scope_note is not None and "Mandya" in r.scope_note
            assert len(r.records) == 1
            assert "district" not in r.filters  # honestly reports the wider filters used

    asyncio.run(run())


def test_district_with_prices_is_not_labelled_fallback(tmp_path: Path) -> None:
    handler = lambda _: httpx.Response(200, json={"total": 1, "records": [mandi_row()]})  # noqa: E731

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            r = await MandiProvider(ctx, make_settings(**KEYED)).get_prices_with_fallback(
                state="Karnataka", district="Mandya", commodity="Tomato"
            )
            assert r.scope == "as_requested" and r.scope_note is None

    asyncio.run(run())


def test_nothing_anywhere_stays_empty_no_invented_prices(tmp_path: Path) -> None:
    handler = lambda _: httpx.Response(200, json={"total": 0, "records": []})  # noqa: E731

    async def run() -> None:
        async with ctx_for(handler, tmp_path) as ctx:
            r = await MandiProvider(ctx, make_settings(**KEYED)).get_prices_with_fallback(
                state="Karnataka", district="Mandya", commodity="Tomato"
            )
            assert r.records == [] and r.scope == "as_requested"

    asyncio.run(run())
