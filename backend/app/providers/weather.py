"""7-day daily + hourly forecast from Open-Meteo (free for non-commercial use, no key)."""

import datetime as dt
from datetime import timedelta
from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.errors import MalformedPayload
from app.providers.base import (
    Provenance,
    ProviderContext,
    Sourced,
    cache_key,
    cached_fetch,
)
from app.providers.http import request_json

SOURCE = "Open-Meteo"
TTL = timedelta(minutes=30)
STALE_MAX = timedelta(hours=12)
DAILY_VARS = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_speed_10m_max",
)
HOURLY_VARS = ("temperature_2m", "precipitation", "precipitation_probability")


class DailyForecast(BaseModel):
    day: dt.date
    temp_max_c: float | None
    temp_min_c: float | None
    precipitation_mm: float | None
    precipitation_probability_pct: float | None
    wind_speed_max_kmh: float | None


class HourlyForecast(BaseModel):
    time: dt.datetime  # local time at the location (timezone=auto)
    temperature_c: float | None
    precipitation_mm: float | None
    precipitation_probability_pct: float | None


class WeatherForecast(Sourced):
    latitude: float
    longitude: float
    timezone: str
    units: dict[str, str]  # as reported by the source
    daily: list[DailyForecast]
    hourly: list[HourlyForecast]


def _num(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise MalformedPayload(SOURCE, "non-numeric value in forecast series")
    return float(value)


def _block(payload: dict[str, Any], name: str, variables: tuple[str, ...]) -> dict[str, list[Any]]:
    block = payload.get(name)
    if not isinstance(block, dict):
        raise MalformedPayload(SOURCE, f"missing '{name}' block")
    times = block.get("time")
    if not isinstance(times, list) or not times:
        raise MalformedPayload(SOURCE, f"'{name}.time' missing or empty")
    for var in variables:
        col = block.get(var)
        if not isinstance(col, list) or len(col) != len(times):
            raise MalformedPayload(SOURCE, f"'{name}.{var}' missing or wrong length")
    return block


def parse_forecast(
    payload: Any,
) -> tuple[str, dict[str, str], list[DailyForecast], list[HourlyForecast]]:
    if not isinstance(payload, dict):
        raise MalformedPayload(SOURCE, "expected a JSON object")
    tz = payload.get("timezone")
    if not isinstance(tz, str):
        raise MalformedPayload(SOURCE, "missing timezone")
    d = _block(payload, "daily", DAILY_VARS)
    h = _block(payload, "hourly", HOURLY_VARS)
    try:
        daily = [
            DailyForecast(
                day=dt.date.fromisoformat(d["time"][i]),
                temp_max_c=_num(d["temperature_2m_max"][i]),
                temp_min_c=_num(d["temperature_2m_min"][i]),
                precipitation_mm=_num(d["precipitation_sum"][i]),
                precipitation_probability_pct=_num(d["precipitation_probability_max"][i]),
                wind_speed_max_kmh=_num(d["wind_speed_10m_max"][i]),
            )
            for i in range(len(d["time"]))
        ]
        hourly = [
            HourlyForecast(
                time=dt.datetime.fromisoformat(h["time"][i]),
                temperature_c=_num(h["temperature_2m"][i]),
                precipitation_mm=_num(h["precipitation"][i]),
                precipitation_probability_pct=_num(h["precipitation_probability"][i]),
            )
            for i in range(len(h["time"]))
        ]
    except (TypeError, ValueError) as exc:
        raise MalformedPayload(SOURCE, "bad date/time in forecast") from exc
    units: dict[str, str] = {}
    for unit_block in ("daily_units", "hourly_units"):
        raw = payload.get(unit_block)
        if isinstance(raw, dict):
            units.update({k: v for k, v in raw.items() if isinstance(v, str)})
    return tz, units, daily, hourly


class WeatherProvider:
    def __init__(self, ctx: ProviderContext, settings: Settings) -> None:
        self._ctx = ctx
        self._url = settings.open_meteo_base_url
        self._timeout = settings.http_timeout_s

    async def get_forecast(self, lat: float, lon: float) -> WeatherForecast:
        # Snap to 0.01 deg (~1 km): the same snapped point is requested and used as cache key.
        lat, lon = round(lat, 2), round(lon, 2)
        return await cached_fetch(
            self._ctx,
            key=cache_key("weather", f"{lat:.2f}", f"{lon:.2f}"),
            ttl=TTL,
            stale_max=STALE_MAX,
            fetch=lambda: self._fetch(lat, lon),
            model=WeatherForecast,
        )

    async def _fetch(self, lat: float, lon: float) -> WeatherForecast:
        payload = await request_json(
            self._ctx.client,
            self._url,
            source=SOURCE,
            retry=self._ctx.retry,
            sleep=self._ctx.sleep,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": ",".join(DAILY_VARS),
                "hourly": ",".join(HOURLY_VARS),
                "forecast_days": 7,
                "timezone": "auto",
            },
            timeout=self._timeout,
        )
        tz, units, daily, hourly = parse_forecast(payload)
        return WeatherForecast(
            provenance=Provenance(
                source=SOURCE,
                source_url=self._url,
                fetched_at=self._ctx.clock(),
                cache_status="live",
            ),
            latitude=lat,
            longitude=lon,
            timezone=tz,
            units=units,
            daily=daily,
            hourly=hourly,
        )
