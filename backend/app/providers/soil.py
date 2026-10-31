"""Soil properties from ISRIC SoilGrids v2.0 (250 m model) + optional user override.

Endpoint: /properties/query?lon&lat&property=..&depth=..&value=mean. Values are scaled integers;
unit_measure.d_factor converts mapped_units -> target_units. SoilGrids values are MODELLED
estimates, not field measurements: the output says so.
"""

from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import MalformedPayload, SourceUnavailable
from app.providers.base import (
    Provenance,
    ProviderContext,
    Sourced,
    cache_key,
    cached_fetch,
)
from app.providers.http import request_json

SOURCE = "ISRIC SoilGrids v2.0"
TTL = timedelta(days=30)
STALE_MAX = timedelta(days=365)
PROPERTIES = ("phh2o", "clay", "sand", "silt", "soc", "nitrogen")
DEPTHS = ("0-5cm", "5-15cm", "15-30cm")
Origin = Literal["soilgrids", "user-provided"]


class SoilProperty(BaseModel):
    name: str
    depth: str | None
    value: float
    unit: str | None
    origin: Origin


class SoilGridsData(Sourced):
    latitude: float
    longitude: float
    properties: list[SoilProperty]
    missing: list[str]  # property@depth combos the model has no value for here
    note: str = "Modelled estimates (250 m grid), not a lab soil test."


class SoilOverride(BaseModel):
    """Values the farmer typed in or took from a soil-health card. Labelled 'user-provided'."""

    ph: float | None = Field(default=None, ge=0, le=14)
    clay: float | None = Field(default=None, ge=0, le=100)  # %
    sand: float | None = Field(default=None, ge=0, le=100)  # %
    silt: float | None = Field(default=None, ge=0, le=100)  # %
    soc: float | None = Field(default=None, ge=0)  # g/kg
    nitrogen: float | None = Field(default=None, ge=0)  # g/kg

    def as_properties(self) -> list[SoilProperty]:
        units = {
            "ph": ("phh2o", "pH"),
            "clay": ("clay", "%"),
            "sand": ("sand", "%"),
            "silt": ("silt", "%"),
            "soc": ("soc", "g/kg"),
            "nitrogen": ("nitrogen", "g/kg"),
        }
        out: list[SoilProperty] = []
        for field_name, (prop, unit) in units.items():
            value = getattr(self, field_name)
            if value is not None:
                out.append(
                    SoilProperty(
                        name=prop, depth=None, value=float(value), unit=unit, origin="user-provided"
                    )
                )
        return out


class SoilProfile(BaseModel):
    latitude: float
    longitude: float
    properties: list[SoilProperty]
    soilgrids: Provenance | None  # None if SoilGrids was unavailable
    soilgrids_error: str | None
    user_override_applied: bool
    missing: list[str]
    notes: list[str]


def parse_soilgrids(payload: Any) -> tuple[list[SoilProperty], list[str]]:
    try:
        layers = payload["properties"]["layers"]
    except (KeyError, TypeError) as exc:
        raise MalformedPayload(SOURCE, "no properties.layers in response") from exc
    if not isinstance(layers, list) or not layers:
        raise MalformedPayload(SOURCE, "properties.layers empty or not a list")
    props: list[SoilProperty] = []
    missing: list[str] = []
    for layer in layers:
        try:
            name = str(layer["name"])
            depths = layer["depths"]
            um = layer.get("unit_measure") or {}
            d_factor = um.get("d_factor")
            if (
                isinstance(d_factor, int | float)
                and not isinstance(d_factor, bool)
                and d_factor != 0
            ):
                scale, unit = float(d_factor), um.get("target_units") or um.get("mapped_units")
            else:  # unit info absent: keep the raw number and say so via the raw unit
                scale, unit = 1.0, um.get("mapped_units")
            for depth in depths:
                label = str(depth["label"])
                mean = (depth.get("values") or {}).get("mean")
                if mean is None:
                    missing.append(f"{name}@{label}")
                    continue
                props.append(
                    SoilProperty(
                        name=name,
                        depth=label,
                        value=float(mean) / scale,
                        unit=unit if isinstance(unit, str) else None,
                        origin="soilgrids",
                    )
                )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise MalformedPayload(SOURCE, "unexpected layer structure") from exc
    return props, missing


class SoilProvider:
    def __init__(self, ctx: ProviderContext, settings: Settings) -> None:
        self._ctx = ctx
        self._url = settings.soilgrids_base_url
        self._timeout = settings.slow_http_timeout_s

    async def fetch_soilgrids(self, lat: float, lon: float) -> SoilGridsData:
        lat, lon = round(lat, 3), round(lon, 3)
        return await cached_fetch(
            self._ctx,
            key=cache_key("soilgrids", f"{lat:.3f}", f"{lon:.3f}"),
            ttl=TTL,
            stale_max=STALE_MAX,
            fetch=lambda: self._fetch(lat, lon),
            model=SoilGridsData,
        )

    async def _fetch(self, lat: float, lon: float) -> SoilGridsData:
        payload = await request_json(
            self._ctx.client,
            self._url,
            source=SOURCE,
            retry=self._ctx.retry,
            sleep=self._ctx.sleep,
            params={
                "lon": lon,
                "lat": lat,
                "property": list(PROPERTIES),
                "depth": list(DEPTHS),
                "value": "mean",
            },
            timeout=self._timeout,
        )
        props, missing = parse_soilgrids(payload)
        return SoilGridsData(
            provenance=Provenance(
                source=SOURCE,
                source_url=self._url,
                fetched_at=self._ctx.clock(),
                cache_status="live",
            ),
            latitude=lat,
            longitude=lon,
            properties=props,
            missing=missing,
        )

    async def get_profile(
        self, lat: float, lon: float, override: SoilOverride | None = None
    ) -> SoilProfile:
        user_props = override.as_properties() if override else []
        data: SoilGridsData | None
        error: str | None = None
        try:
            data = await self.fetch_soilgrids(lat, lon)
        except SourceUnavailable as exc:
            if not user_props:
                raise
            data, error = None, exc.user_message()
        overridden = {p.name for p in user_props}
        grid_props = [p for p in data.properties if p.name not in overridden] if data else []
        notes = (
            ["Values labelled 'user-provided' replace SoilGrids for that property."]
            if user_props
            else []
        )
        if data:
            notes.append(data.note)
        return SoilProfile(
            latitude=round(lat, 3),
            longitude=round(lon, 3),
            properties=grid_props + user_props,
            soilgrids=data.provenance if data else None,
            soilgrids_error=error,
            user_override_applied=bool(user_props),
            missing=data.missing if data else [],
            notes=notes,
        )
