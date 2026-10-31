"""Deterministic safety alerts computed from a REAL forecast.

Pure functions: no I/O, no LLM, no globals, no hidden thresholds (all come from Thresholds, which
come from config). The same forecast + thresholds + time always give the same alerts, whichever
LLM is plugged in. An empty alert list is never presented as "safe": missing/stale/incomplete data
produces its own alert.
"""

from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import NotConfigured
from app.providers.weather import HourlyForecast, WeatherForecast

AlertCode = Literal[
    "rain_48h", "heat", "frost", "forecast_incomplete", "forecast_stale", "no_forecast"
]
Restriction = Literal["no_spraying", "no_harvest"]
Param = float | int | str


class Thresholds(BaseModel, frozen=True):
    window_hours: int
    rain_total_mm: float
    rain_prob_pct: float
    heat_max_c: float
    frost_min_c: float

    @classmethod
    def from_settings(cls, s: Settings) -> "Thresholds":
        named = {
            "GUARD_RAIN_TOTAL_MM_48H": s.guard_rain_total_mm_48h,
            "GUARD_RAIN_PROB_PCT": s.guard_rain_prob_pct,
            "GUARD_HEAT_MAX_C": s.guard_heat_max_c,
            "GUARD_FROST_MIN_C": s.guard_frost_min_c,
        }
        missing = [k for k, v in named.items() if v is None]
        rain_mm, rain_prob = s.guard_rain_total_mm_48h, s.guard_rain_prob_pct
        heat, frost = s.guard_heat_max_c, s.guard_frost_min_c
        if rain_mm is None or rain_prob is None or heat is None or frost is None:
            raise NotConfigured(
                "Safety guards",
                f"set {', '.join(missing)} in .env (there are no built-in defaults)",
            )
        return cls(
            window_hours=s.guard_window_hours,
            rain_total_mm=rain_mm,
            rain_prob_pct=rain_prob,
            heat_max_c=heat,
            frost_min_c=frost,
        )


class Alert(BaseModel):
    code: AlertCode
    severity: Literal["warning", "info"]
    restrictions: list[Restriction] = Field(default_factory=list)
    threshold: float | None = None  # the configured value that was crossed (echoed for audit)
    params: dict[str, Param] = Field(default_factory=dict)
    window_start: datetime | None = None  # local time at the forecast location
    window_end: datetime | None = None
    source: str | None = None
    forecast_fetched_at: datetime | None = None
    cache_status: str | None = None
    source_id: str | None = None  # citation id, attached by the agent
    message: str = ""  # localized text, attached by app.agent.i18n.localize_alerts


class GuardReport(BaseModel):
    alerts: list[Alert]
    window_start: datetime
    window_end: datetime
    hours_covered: int

    @property
    def restrictions(self) -> list[Restriction]:
        return sorted({r for a in self.alerts for r in a.restrictions})


def no_forecast_alert(reason: str) -> Alert:
    return Alert(code="no_forecast", severity="warning", params={"reason": reason})


def _window(forecast: WeatherForecast, now: datetime, hours: int) -> tuple[datetime, datetime]:
    local = now.astimezone(ZoneInfo(forecast.timezone)).replace(tzinfo=None)
    start = local.replace(minute=0, second=0, microsecond=0)
    return start, start + timedelta(hours=hours)


def compute_alerts(forecast: WeatherForecast, t: Thresholds, now: datetime) -> GuardReport:
    """`now` must be timezone-aware. Hourly rows from `now` (floored to the hour) for
    t.window_hours are examined; thresholds are inclusive (>= rain/heat, <= frost)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    start, end = _window(forecast, now, t.window_hours)
    rows: list[HourlyForecast] = [h for h in forecast.hourly if start <= h.time < end]
    covered = sum(1 for h in rows if h.precipitation_mm is not None and h.temperature_c is not None)
    prov = forecast.provenance
    common = {
        "window_start": start,
        "window_end": end,
        "source": prov.source,
        "forecast_fetched_at": prov.fetched_at,
        "cache_status": prov.cache_status,
    }
    alerts: list[Alert] = []

    total = round(sum(h.precipitation_mm or 0.0 for h in rows), 1)
    probs = [
        h.precipitation_probability_pct for h in rows if h.precipitation_probability_pct is not None
    ]
    max_prob = max(probs) if probs else None
    by_mm = total >= t.rain_total_mm
    by_prob = max_prob is not None and max_prob >= t.rain_prob_pct and total > 0
    if by_mm or by_prob:
        first_wet = next((h.time for h in rows if (h.precipitation_mm or 0.0) > 0), None)
        alerts.append(
            Alert(
                code="rain_48h",
                severity="warning",
                restrictions=["no_harvest", "no_spraying"],
                threshold=t.rain_total_mm if by_mm else t.rain_prob_pct,
                params={
                    "hours": t.window_hours,
                    "total_mm": total,
                    "max_prob": max_prob if max_prob is not None else "n/a",
                    "first_rain_at": first_wet.isoformat() if first_wet else "n/a",
                    "triggered_by": "total_mm" if by_mm else "probability",
                    "threshold_mm": t.rain_total_mm,
                    "threshold_prob": t.rain_prob_pct,
                },
                **common,  # type: ignore[arg-type]
            )
        )

    temps = [(h.temperature_c, h.time) for h in rows if h.temperature_c is not None]
    if temps:
        hot_c, hot_at = max(temps, key=lambda x: x[0])
        if hot_c >= t.heat_max_c:
            alerts.append(
                Alert(
                    code="heat",
                    severity="warning",
                    threshold=t.heat_max_c,
                    params={
                        "hours": t.window_hours,
                        "max_c": hot_c,
                        "threshold_c": t.heat_max_c,
                        "at": hot_at.isoformat(),
                    },
                    **common,  # type: ignore[arg-type]
                )
            )
        cold_c, cold_at = min(temps, key=lambda x: x[0])
        if cold_c <= t.frost_min_c:
            alerts.append(
                Alert(
                    code="frost",
                    severity="warning",
                    threshold=t.frost_min_c,
                    params={
                        "hours": t.window_hours,
                        "min_c": cold_c,
                        "threshold_c": t.frost_min_c,
                        "at": cold_at.isoformat(),
                    },
                    **common,  # type: ignore[arg-type]
                )
            )

    if covered < t.window_hours:
        alerts.append(
            Alert(
                code="forecast_incomplete",
                severity="warning",
                params={"hours": t.window_hours, "covered": covered},
                **common,  # type: ignore[arg-type]
            )
        )
    if prov.cache_status == "stale":
        age_h = round((now - prov.fetched_at).total_seconds() / 3600, 1)
        alerts.append(
            Alert(
                code="forecast_stale",
                severity="warning",
                params={"age_hours": age_h},
                **common,  # type: ignore[arg-type]
            )
        )
    return GuardReport(alerts=alerts, window_start=start, window_end=end, hours_covered=covered)
