"""Tool schemas + executor wrapping the real providers. Every successful result gets citation ids
(S1, S2, ...) with source, URL and timestamp; failures come back as explicit errors, never as
substitute data. The user's resolved location is fixed once (request GPS > first geocode
match > first coordinates used), so weather, soil and alerts describe one place."""

import asyncio
import logging
import statistics
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.agent.schemas import SourceKind, SourceRef, ToolTrace
from app.config import Settings
from app.errors import SourceUnavailable
from app.llm.base import ToolCall, ToolSpec
from app.providers.base import Provenance
from app.providers.geocode import GeocodeProvider
from app.providers.mandi import MandiPrices, MandiProvider
from app.providers.soil import SoilProvider
from app.providers.weather import WeatherForecast, WeatherProvider
from app.rag.retriever import Retriever

log = logging.getLogger("agri.agent")


DOCS = {
    "geocode": (
        "Find coordinates, state and district for a place name. Use first when the user names a "
        "place and no coordinates are known."
    ),
    "get_weather": (
        "Live 7-day daily weather forecast (temperature, rain mm and probability, wind) for the "
        "user's location. Call before advising on harvest, spraying, sowing or irrigation."
    ),
    "get_mandi_prices": (
        "Latest mandi (market) prices in INR per quintal from Agmarknet for a commodity, "
        "optionally by state/district. Call before any sell-or-hold or harvest-timing advice. "
        "If the district has no price, other markets in the same state are returned and "
        "scope_note says so: always tell the user."
    ),
    "get_soil": (
        "Modelled soil properties (pH, clay, sand, silt, organic carbon, nitrogen) at the user's "
        "location. Estimates, not a lab test."
    ),
    "search_knowledge": (
        "Search ingested ICAR / state university / KVK package-of-practices documents. The ONLY "
        "allowed source for crop practice advice (sowing, varieties, pests, spraying). Returns "
        "cited chunks."
    ),
}


class ToolInputError(Exception):
    """The arguments were valid JSON but unusable (e.g. no location known yet)."""


class GeocodeArgs(BaseModel):
    query: str = Field(
        min_length=2, max_length=200, description="Place name, e.g. 'Mandya, Karnataka'"
    )
    limit: int = Field(default=3, ge=1, le=5)


class WeatherArgs(BaseModel):
    lat: float | None = Field(default=None, ge=-90, le=90, description="Optional if location known")
    lon: float | None = Field(default=None, ge=-180, le=180)


class PricesArgs(BaseModel):
    commodity: str = Field(min_length=2, max_length=60, description="e.g. 'Tomato'")
    state: str | None = Field(default=None, max_length=60)
    district: str | None = Field(default=None, max_length=60)


class SoilArgs(BaseModel):
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)


class KnowledgeArgs(BaseModel):
    query: str = Field(min_length=2, max_length=300)
    crop: str | None = Field(default=None, max_length=60, description="lowercase crop name")
    k: int = Field(default=5, ge=1, le=8)


@dataclass
class ToolResult:
    content: dict[str, Any]
    ok: bool
    source_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Location:
    lat: float
    lon: float
    label: str


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    s = model.model_json_schema()
    s.pop("title", None)
    for prop in s.get("properties", {}).values():
        prop.pop("title", None)
    return s


def summarize_prices(p: MandiPrices, max_rows: int) -> dict[str, Any]:
    """Deterministic aggregation (code, not LLM) so price facts are exact and citable."""
    by_date: dict[Any, list[Any]] = defaultdict(list)
    for r in p.records:
        by_date[r.arrival_date].append(r)
    days = sorted(by_date, reverse=True)[:7]
    daily = []
    for d in days:
        modal = [r.modal_price for r in by_date[d]]
        daily.append(
            {
                "date": d.isoformat(),
                "markets_reporting": len({r.market for r in by_date[d]}),
                "modal_price_min": min(modal),
                "modal_price_median": statistics.median(modal),
                "modal_price_max": max(modal),
            }
        )
    latest = [
        {
            "market": r.market, "district": r.district, "variety": r.variety,
            "min_price": r.min_price, "max_price": r.max_price, "modal_price": r.modal_price,
        }
        for r in (by_date[days[0]][:max_rows] if days else [])
    ]  # fmt: skip
    return {
        "filters": p.filters,
        "price_unit": p.price_unit,
        "scope": p.scope,
        "scope_note": p.scope_note,
        "latest_arrival_date": p.latest_arrival_date.isoformat() if p.latest_arrival_date else None,
        "days_since_latest": p.days_since_latest,
        "records_returned": len(p.records),
        "truncated": p.truncated,
        "by_date": daily,
        "latest_day_markets": latest,
    }


class ToolRegistry:
    def __init__(
        self,
        *,
        geocoder: GeocodeProvider,
        weather: WeatherProvider,
        mandi: MandiProvider,
        soil: SoilProvider,
        retriever: Callable[[], Retriever],
        settings: Settings,
        location: Location | None = None,
    ) -> None:
        self._geo, self._weather, self._mandi, self._soil = geocoder, weather, mandi, soil
        self._retriever = retriever
        self._s = settings
        self.location = location
        self.forecast: WeatherForecast | None = None
        self.sources: dict[str, SourceRef] = {}
        self.facts: list[Any] = []  # JSON of every successful result (number-grounding corpus)
        self.trace: list[ToolTrace] = []
        self._handlers: dict[str, tuple[type[BaseModel], Callable[[Any], Any], str]] = {
            "geocode": (GeocodeArgs, self._geocode, DOCS["geocode"]),
            "get_weather": (WeatherArgs, self._get_weather, DOCS["get_weather"]),
            "get_mandi_prices": (PricesArgs, self._get_prices, DOCS["get_mandi_prices"]),
            "get_soil": (SoilArgs, self._get_soil, DOCS["get_soil"]),
            "search_knowledge": (KnowledgeArgs, self._search, DOCS["search_knowledge"]),
        }

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=n, description=d, parameters=_schema(m))
            for n, (m, _, d) in self._handlers.items()
        ]

    # ---- execution ------------------------------------------------------------------------
    async def execute(self, call: ToolCall) -> ToolResult:
        t0 = time.perf_counter()
        before = set(self.sources)
        entry = self._handlers.get(call.name)
        if entry is None:
            result = self._error(f"unknown tool '{call.name}'")
        else:
            model, handler, _ = entry
            try:
                result = await handler(model.model_validate(call.arguments))
            except ValidationError as exc:
                bad = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
                result = self._error(f"invalid arguments ({bad})")
            except ToolInputError as exc:
                result = self._error(str(exc))
            except SourceUnavailable as exc:
                result = self._error(exc.user_message(), unavailable=True)
            except Exception:
                log.exception("tool %s crashed", call.name)
                result = self._error("the tool failed unexpectedly")
        if result.ok:
            self.facts.append(result.content)
        ms = int((time.perf_counter() - t0) * 1000)
        new_ids = [i for i in self.sources if i not in before]
        self.trace.append(
            ToolTrace(name=call.name, arguments=call.arguments, ok=result.ok, latency_ms=ms,
                      source_ids=new_ids)
        )  # fmt: skip
        log.info(
            "tool_call name=%s ok=%s latency_ms=%d args=%s",
            call.name,
            result.ok,
            ms,
            call.arguments,
        )
        return result

    @staticmethod
    def _error(message: str, *, unavailable: bool = False) -> ToolResult:
        return ToolResult({"error": message, "source_unavailable": unavailable}, ok=False)

    def _add_source(
        self, kind: SourceKind, name: str, url: str, prov: Provenance | None = None,
        detail: str | None = None,
    ) -> str:  # fmt: skip
        sid = f"S{len(self.sources) + 1}"
        self.sources[sid] = SourceRef(
            id=sid, kind=kind, name=name, url=url,
            fetched_at=prov.fetched_at if prov else None,
            cache_status=prov.cache_status if prov else None, detail=detail,
        )  # fmt: skip
        return sid

    def _resolve(self, lat: float | None, lon: float | None) -> tuple[float, float, str | None]:
        if self.location:
            note = None
            if (
                lat is not None
                and lon is not None
                and (abs(lat - self.location.lat) > 0.05 or abs(lon - self.location.lon) > 0.05)
            ):
                note = (
                    f"Used the user's resolved location ({self.location.label}), "
                    "not the coordinates passed."
                )
            return self.location.lat, self.location.lon, note
        if lat is None or lon is None:
            raise ToolInputError("no location known yet: call geocode first or pass lat and lon")
        self.location = Location(lat, lon, f"{lat:.2f},{lon:.2f}")
        return lat, lon, None

    # ---- handlers -------------------------------------------------------------------------
    async def _geocode(self, a: GeocodeArgs) -> ToolResult:
        res = await self._geo.search(a.query, a.limit)
        if not res.matches:
            return ToolResult(
                {"matches": [], "note": "no place found; ask the user to rephrase"}, ok=True
            )
        top = res.matches[0]
        if self.location is None:
            self.location = Location(top.latitude, top.longitude, top.display_name)
        sid = self._add_source("geocode", res.provenance.source, res.provenance.source_url,
                               res.provenance, res.attribution)  # fmt: skip
        content = {"source_id": sid, "provenance": res.provenance.model_dump(mode="json"),
                   "matches": [m.model_dump(mode="json") for m in res.matches],
                   "location_in_use": self.location.label}  # fmt: skip
        return ToolResult(content, True, [sid])

    async def _get_weather(self, a: WeatherArgs) -> ToolResult:
        lat, lon, note = self._resolve(a.lat, a.lon)
        f = await self._weather.get_forecast(lat, lon)
        self.forecast = f
        sid = self._add_source(
            "weather", f.provenance.source, f.provenance.source_url, f.provenance
        )
        content: dict[str, Any] = {
            "source_id": sid,
            "provenance": f.provenance.model_dump(mode="json"),
            "location": {"lat": f.latitude, "lon": f.longitude, "timezone": f.timezone},
            "units": f.units,
            "daily": [d.model_dump(mode="json") for d in f.daily],
        }
        if note:
            content["note"] = note
        return ToolResult(content, True, [sid])

    async def _get_prices(self, a: PricesArgs) -> ToolResult:
        p = await self._mandi.get_prices_with_fallback(
            state=a.state, district=a.district, commodity=a.commodity, max_records=200
        )
        sid = self._add_source("prices", p.provenance.source, p.provenance.source_url, p.provenance)
        content = {"source_id": sid, "provenance": p.provenance.model_dump(mode="json"),
                   **summarize_prices(p, self._s.tool_max_price_rows)}  # fmt: skip
        if not p.records:
            content["note"] = "No price records matched these filters."
        return ToolResult(content, True, [sid])

    async def _get_soil(self, a: SoilArgs) -> ToolResult:
        lat, lon, note = self._resolve(a.lat, a.lon)
        s = await self._soil.get_profile(lat, lon)
        prov = s.soilgrids
        sid = self._add_source("soil", prov.source if prov else "SoilGrids",
                               prov.source_url if prov else "", prov)  # fmt: skip
        content: dict[str, Any] = {
            "source_id": sid,
            "provenance": prov.model_dump(mode="json") if prov else None,
            "properties": [p.model_dump(mode="json") for p in s.properties],
            "missing": s.missing, "notes": s.notes,
        }  # fmt: skip
        if note:
            content["note"] = note
        return ToolResult(content, True, [sid])

    async def _search(self, a: KnowledgeArgs) -> ToolResult:
        retriever = self._retriever()
        chunks = await asyncio.to_thread(retriever.search, a.query, crop=a.crop, k=a.k)
        if not chunks:
            return ToolResult({"results": [], "note": "no matching documents"}, ok=True)
        results, ids = [], []
        for c in chunks:
            detail = (f"p. {c.page}, " if c.page else "") + f"ingested {c.ingest_date}"
            sid = self._add_source("knowledge", c.source_title, c.url, None, detail)
            ids.append(sid)
            results.append({"source_id": sid, "text": c.text, "document": c.source_title,
                            "page": c.page, "crop": c.crop, "region": c.region,
                            "ingested": c.ingest_date, "score": c.score})  # fmt: skip
        return ToolResult({"results": results}, True, ids)

    # ---- used by the loop for deterministic alerts ------------------------------------------
    async def ensure_forecast(self) -> tuple[WeatherForecast | None, str]:
        """The forecast for the canonical location (fetched here if the LLM never asked)."""
        if self.location is None:
            return None, "no location could be determined"
        have = self.forecast and (
            round(self.forecast.latitude, 2),
            round(self.forecast.longitude, 2),
        )
        if have != (round(self.location.lat, 2), round(self.location.lon, 2)):
            try:
                self.forecast = await self._weather.get_forecast(
                    self.location.lat, self.location.lon
                )
            except SourceUnavailable as exc:
                return None, exc.user_message()
        return self.forecast, ""

    def source_for_forecast(self, f: WeatherForecast) -> str:
        for sid, ref in self.sources.items():
            if ref.kind == "weather" and ref.fetched_at == f.provenance.fetched_at:
                return sid
        return self._add_source(
            "weather", f.provenance.source, f.provenance.source_url, f.provenance
        )
