"""Fakes for agent tests. All data here is synthetic test input, never advice."""

import datetime as dt
import json
from collections.abc import Sequence
from typing import Any

from app.agent.guards import Thresholds
from app.errors import SourceUnavailable
from app.llm.base import ChatMessage, LLMResponse, ToolCall, ToolSpec
from app.providers.base import Provenance
from app.providers.geocode import GeocodeResult, GeoMatch
from app.providers.mandi import MandiPrices, MandiRecord
from app.providers.soil import SoilProfile, SoilProperty
from app.providers.weather import DailyForecast, HourlyForecast, WeatherForecast
from app.rag.retriever import RetrievedChunk

UTC = dt.UTC
NOW = dt.datetime(2026, 10, 31, 5, 30, tzinfo=UTC)  # 11:00 in Asia/Kolkata
FETCHED = dt.datetime(2026, 10, 31, 5, 9, tzinfo=UTC)
T = Thresholds(window_hours=48, rain_total_mm=10, rain_prob_pct=60, heat_max_c=40, frost_min_c=2)
LAT, LON = 12.52, 76.9


def prov(source: str = "Open-Meteo", status: str = "live", at: dt.datetime = FETCHED) -> Provenance:
    return Provenance(
        source=source, source_url=f"https://example.test/{source}", fetched_at=at,
        cache_status=status,  # type: ignore[arg-type]
    )  # fmt: skip


def make_forecast(
    rain: dict[int, float] | None = None,  # hour offset from 2026-10-31T00:00 local -> mm
    prob: dict[int, float] | None = None,
    temp: dict[int, float] | None = None,
    base_temp: float = 25.0,
    hours: int = 168,
    status: str = "live",
    at: dt.datetime = FETCHED,
    lat: float = LAT,
    lon: float = LON,
) -> WeatherForecast:
    start = dt.datetime(2026, 10, 31, 0, 0)
    hourly = [
        HourlyForecast(
            time=start + dt.timedelta(hours=i),
            temperature_c=(temp or {}).get(i, base_temp),
            precipitation_mm=(rain or {}).get(i, 0.0),
            precipitation_probability_pct=(prob or {}).get(i, 0.0),
        )
        for i in range(hours)
    ]
    daily = [
        DailyForecast(
            day=start.date() + dt.timedelta(days=d), temp_max_c=28.0 + d, temp_min_c=19.0,
            precipitation_mm=float(sum((rain or {}).get(d * 24 + h, 0.0) for h in range(24))),
            precipitation_probability_pct=40.0, wind_speed_max_kmh=11.0,
        )
        for d in range(7)
    ]  # fmt: skip
    return WeatherForecast(
        provenance=prov(status=status, at=at), latitude=lat, longitude=lon,
        timezone="Asia/Kolkata", units={"precipitation_sum": "mm"}, daily=daily, hourly=hourly,
    )  # fmt: skip


# hour 30..33 (=Nov 1, 06:00-09:00 local) 4.0 mm each => 16.0 mm inside the 48 h window
RAINY = make_forecast(rain={30: 4.0, 31: 4.0, 32: 4.0, 33: 4.0}, prob={31: 82.0})
DRY = make_forecast()


class FakeGeocoder:
    def __init__(self, matches: list[tuple[str, float, float]] | None = None) -> None:
        self.matches = matches or [("Mandya, Karnataka, India", LAT, LON)]
        self.calls: list[str] = []

    async def search(self, query: str, limit: int = 5) -> GeocodeResult:
        self.calls.append(query)
        return GeocodeResult(
            provenance=prov("OpenStreetMap Nominatim"), query=query,
            matches=[GeoMatch(display_name=n, latitude=a, longitude=b, state="Karnataka",
                              district="Mandya") for n, a, b in self.matches][:limit],
        )  # fmt: skip


class FakeWeather:
    def __init__(self, forecast: WeatherForecast | None = RAINY, error: bool = False) -> None:
        self.forecast, self.error = forecast, error
        self.calls: list[tuple[float, float]] = []

    async def get_forecast(self, lat: float, lon: float) -> WeatherForecast:
        self.calls.append((lat, lon))
        if self.error or self.forecast is None:
            raise SourceUnavailable("Open-Meteo", "timeout", "no response")
        return self.forecast.model_copy(
            update={"latitude": round(lat, 2), "longitude": round(lon, 2)}
        )


class FakeMandi:
    def __init__(self, modal: tuple[float, ...] = (2400.0, 2600.0)) -> None:
        self.modal = modal
        self.calls: list[dict[str, Any]] = []

    async def get_prices(self, **kw: Any) -> MandiPrices:
        self.calls.append(kw)
        d = dt.date(2026, 10, 31)
        recs = [
            MandiRecord(state="Karnataka", district="Mandya", market=f"Market{i}", commodity="Tomato",
                        arrival_date=d, min_price=m - 200, max_price=m + 200, modal_price=m)
            for i, m in enumerate(self.modal)
        ]  # fmt: skip
        return MandiPrices(
            provenance=prov("Agmarknet (data.gov.in)"), filters={"commodity": "Tomato"},
            total_available=len(recs), truncated=False, skipped_records=0, dates_present=[d],
            latest_arrival_date=d, days_since_latest=0, records=recs,
        )  # fmt: skip


class FakeSoil:
    async def get_profile(self, lat: float, lon: float) -> SoilProfile:
        return SoilProfile(
            latitude=lat, longitude=lon, soilgrids=prov("SoilGrids"), soilgrids_error=None,
            user_override_applied=False, missing=[], notes=[],
            properties=[SoilProperty(name="phh2o", depth="0-5cm", value=6.8, unit="pH", origin="soilgrids")],
        )  # fmt: skip


class FakeRetriever:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, str | None]] = []

    def search(
        self, query: str, *, crop: str | None = None, k: int = 5, **_: Any
    ) -> list[RetrievedChunk]:
        self.calls.append((query, crop))
        if self.error:
            raise self.error
        return [
            RetrievedChunk(
                rank=1,
                score=0.8,
                text="Zebra quartz lantern marble orbit.",
                citation="Doc A, p. 3 - u",
                source_title="Doc A",
                url="https://example.test/doca.pdf",
                page=3,
                crop="tomato",
                region=None,
                ingest_date="2026-10-30",
                chunk_id="a-p3-c0",
            )
        ]


def call(name: str, **args: Any) -> ToolCall:
    return ToolCall(id=f"id_{name}", name=name, arguments=args)


def tools_turn(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(content="", tool_calls=list(calls))


def text(s: str = "done") -> LLMResponse:
    return LLMResponse(content=s)


def final_json(**over: Any) -> LLMResponse:
    body: dict[str, Any] = {"answer_text": "ok [S1]", "schedule": [], "sell_or_hold": None}
    body.update(over)
    return LLMResponse(content=json.dumps(body, ensure_ascii=False))


class ScriptedLLM:
    """Plays back canned responses; records what it was sent. Stands in for any real model."""

    def __init__(
        self, responses: Sequence[LLMResponse | Exception], name: str = "scripted"
    ) -> None:
        self.provider, self.model = "scripted", name
        self._r = list(responses)
        self.calls: list[list[ChatMessage]] = []
        self.tool_counts: list[int] = []
        self.json_modes: list[bool] = []

    async def chat(
        self, messages: Sequence[ChatMessage], tools: Sequence[ToolSpec], *, json_mode: bool = False
    ) -> LLMResponse:
        self.calls.append(list(messages))
        self.tool_counts.append(len(tools))
        self.json_modes.append(json_mode)
        item = self._r.pop(0) if len(self._r) > 1 else self._r[0]  # last response repeats
        if isinstance(item, Exception):
            raise item
        return item
