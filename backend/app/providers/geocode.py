"""Place name -> coordinates via OpenStreetMap Nominatim (public instance).

Policy (operations.osmfoundation.org/policies/nominatim): identifying User-Agent, max 1 req/s,
cache results, show OSM attribution.
"""

from datetime import timedelta
from typing import Any

from pydantic import BaseModel

from app import __version__
from app.config import Settings
from app.errors import MalformedPayload
from app.providers.base import (
    Provenance,
    ProviderContext,
    Sourced,
    cache_key,
    cached_fetch,
)
from app.providers.http import MinIntervalLimiter, request_json

SOURCE = "OpenStreetMap Nominatim"
ATTRIBUTION = "(c) OpenStreetMap contributors (ODbL)"
TTL = timedelta(days=30)
STALE_MAX = timedelta(days=365)


class GeoMatch(BaseModel):
    display_name: str
    latitude: float
    longitude: float
    state: str | None = None
    district: str | None = None
    importance: float | None = None


class GeocodeResult(Sourced):
    query: str
    matches: list[GeoMatch]
    attribution: str = ATTRIBUTION


def parse_matches(payload: Any) -> list[GeoMatch]:
    if not isinstance(payload, list):
        raise MalformedPayload(SOURCE, "expected a JSON list of places")
    matches: list[GeoMatch] = []
    for item in payload:
        if not isinstance(item, dict):
            raise MalformedPayload(SOURCE, "place entry is not an object")
        try:
            lat = float(item["lat"])
            lon = float(item["lon"])
            name = str(item["display_name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MalformedPayload(SOURCE, "place entry missing lat/lon/display_name") from exc
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise MalformedPayload(SOURCE, "coordinates out of range")
        address = item.get("address")
        address = address if isinstance(address, dict) else {}
        importance = item.get("importance")
        matches.append(
            GeoMatch(
                display_name=name,
                latitude=lat,
                longitude=lon,
                state=address.get("state"),
                district=address.get("state_district") or address.get("county"),
                importance=float(importance) if isinstance(importance, int | float) else None,
            )
        )
    return matches


class GeocodeProvider:
    def __init__(
        self, ctx: ProviderContext, settings: Settings, limiter: MinIntervalLimiter
    ) -> None:
        self._ctx = ctx
        self._base = settings.nominatim_base_url.rstrip("/")
        self._timeout = settings.http_timeout_s
        self._limiter = limiter
        agent = f"agri-advisor/{__version__}"
        if settings.contact_email:
            agent += f" ({settings.contact_email})"
        self._user_agent = agent

    async def search(self, query: str, limit: int = 5) -> GeocodeResult:
        q = " ".join(query.split())
        if not q:
            raise ValueError("query must not be empty")
        return await cached_fetch(
            self._ctx,
            key=cache_key("geocode", q.lower(), limit),
            ttl=TTL,
            stale_max=STALE_MAX,
            fetch=lambda: self._fetch(q, limit),
            model=GeocodeResult,
        )

    async def _fetch(self, q: str, limit: int) -> GeocodeResult:
        url = f"{self._base}/search"
        payload = await request_json(
            self._ctx.client,
            url,
            source=SOURCE,
            retry=self._ctx.retry,
            sleep=self._ctx.sleep,
            params={
                "q": q,
                "format": "jsonv2",
                "addressdetails": 1,
                "limit": limit,
                "countrycodes": "in",
            },
            headers={"User-Agent": self._user_agent, "Accept-Language": "en"},
            timeout=self._timeout,
            before_attempt=self._limiter.wait,
        )
        return GeocodeResult(
            provenance=Provenance(
                source=SOURCE, source_url=url, fetched_at=self._ctx.clock(), cache_status="live"
            ),
            query=q,
            matches=parse_matches(payload),
        )
