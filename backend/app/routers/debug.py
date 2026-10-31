"""Debug endpoints exposing provider output verbatim (with provenance). Not the farmer API."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.deps import get_geocoder, get_mandi, get_soil, get_weather
from app.providers.geocode import GeocodeProvider, GeocodeResult
from app.providers.mandi import MandiPrices, MandiProvider
from app.providers.soil import SoilOverride, SoilProfile, SoilProvider
from app.providers.weather import WeatherForecast, WeatherProvider

router = APIRouter(tags=["debug"])

Lat = Annotated[float, Query(ge=-90, le=90)]
Lon = Annotated[float, Query(ge=-180, le=180)]


@router.get("/geocode", response_model=GeocodeResult)
async def geocode(
    *,
    q: Annotated[str, Query(min_length=2, max_length=200)],
    limit: Annotated[int, Query(ge=1, le=10)] = 5,
    provider: Annotated[GeocodeProvider, Depends(get_geocoder)],
) -> GeocodeResult:
    return await provider.search(q, limit)


@router.get("/weather", response_model=WeatherForecast)
async def weather(
    *,
    lat: Lat,
    lon: Lon,
    hourly_hours: Annotated[int, Query(ge=0, le=168, description="first N hourly rows")] = 48,
    provider: Annotated[WeatherProvider, Depends(get_weather)],
) -> WeatherForecast:
    forecast = await provider.get_forecast(lat, lon)
    return forecast.model_copy(update={"hourly": forecast.hourly[:hourly_hours]})


@router.get("/prices", response_model=MandiPrices)
async def prices(
    *,
    state: str | None = None,
    district: str | None = None,
    commodity: str | None = None,
    max_records: Annotated[int, Query(ge=1, le=1000)] = 50,
    provider: Annotated[MandiProvider, Depends(get_mandi)],
) -> MandiPrices:
    return await provider.get_prices(
        state=state, district=district, commodity=commodity, max_records=max_records
    )


@router.get("/soil", response_model=SoilProfile)
async def soil(
    *,
    lat: Lat,
    lon: Lon,
    ph: Annotated[float | None, Query(ge=0, le=14)] = None,
    clay: Annotated[float | None, Query(ge=0, le=100)] = None,
    sand: Annotated[float | None, Query(ge=0, le=100)] = None,
    silt: Annotated[float | None, Query(ge=0, le=100)] = None,
    soc: Annotated[float | None, Query(ge=0)] = None,
    nitrogen: Annotated[float | None, Query(ge=0)] = None,
    provider: Annotated[SoilProvider, Depends(get_soil)],
) -> SoilProfile:
    override = SoilOverride(ph=ph, clay=clay, sand=sand, silt=silt, soc=soc, nitrogen=nitrogen)
    return await provider.get_profile(lat, lon, override)
