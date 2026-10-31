"""Test helpers: mock HTTP transport, fake clock, payload builders. MOCKS LIVE ONLY HERE."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from app.cache import DiskCache
from app.config import Settings
from app.providers.base import ProviderContext
from app.providers.http import RetryPolicy

Handler = Callable[[httpx.Request], httpx.Response]


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime(2026, 10, 31, 6, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs: float) -> None:
        self.now += timedelta(**kwargs)


async def no_sleep(_: float) -> None:
    return None


def make_settings(**overrides: Any) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[call-arg]


@asynccontextmanager
async def ctx_for(
    handler: Handler, tmp_path: Path, clock: FakeClock | None = None
) -> AsyncIterator[ProviderContext]:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cache = DiskCache(str(tmp_path / "cache"))
    try:
        yield ProviderContext(
            client=client,
            cache=cache,
            retry=RetryPolicy(attempts=2, base_delay=0.0),
            clock=clock or FakeClock(),
            sleep=no_sleep,
        )
    finally:
        await client.aclose()
        cache.close()


def weather_payload() -> dict[str, Any]:
    return {
        "latitude": 12.97,
        "longitude": 77.59,
        "timezone": "Asia/Kolkata",
        "daily_units": {"temperature_2m_max": "°C", "precipitation_sum": "mm"},
        "hourly_units": {"temperature_2m": "°C", "precipitation": "mm"},
        "daily": {
            "time": ["2026-10-31", "2026-11-01"],
            "temperature_2m_max": [29.1, 28.4],
            "temperature_2m_min": [20.2, 19.9],
            "precipitation_sum": [0.0, 12.5],
            "precipitation_probability_max": [10, 80],
            "wind_speed_10m_max": [14.0, 18.5],
        },
        "hourly": {
            "time": ["2026-10-31T00:00", "2026-10-31T01:00", "2026-10-31T02:00"],
            "temperature_2m": [21.0, 20.5, None],
            "precipitation": [0.0, 0.0, 0.1],
            "precipitation_probability": [5, 5, 10],
        },
    }


def mandi_row(market: str = "Mandya", date: str = "28/10/2026", **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "state": "Karnataka",
        "district": "Mandya",
        "market": market,
        "commodity": "Tomato",
        "variety": "Local",
        "grade": "FAQ",
        "arrival_date": date,
        "min_price": "1000",
        "max_price": "1800",
        "modal_price": "1500",
    }
    row.update(kw)
    return row


def geocode_payload() -> list[dict[str, Any]]:
    return [
        {
            "lat": "12.5218",
            "lon": "76.8951",
            "importance": 0.6,
            "display_name": "Mandya, Mandya District, Karnataka, India",
            "address": {"state_district": "Mandya District", "state": "Karnataka"},
        }
    ]


def soil_payload() -> dict[str, Any]:
    def layer(name: str, mapped: str, target: str, factor: int, mean: int | None) -> dict[str, Any]:
        return {
            "name": name,
            "unit_measure": {"d_factor": factor, "mapped_units": mapped, "target_units": target},
            "depths": [{"label": "0-5cm", "values": {"mean": mean}}],
        }

    return {
        "properties": {
            "layers": [
                layer("phh2o", "pH*10", "-", 10, 62),
                layer("clay", "g/kg", "%", 10, 253),
                layer("nitrogen", "cg/kg", "g/kg", 100, None),
            ]
        }
    }
