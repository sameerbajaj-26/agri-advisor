"""Guards are pure and threshold-driven: these tests pin the boundaries."""

import datetime as dt

import pytest

from app.agent.guards import Thresholds, compute_alerts
from app.errors import NotConfigured
from tests.agent_helpers import DRY, NOW, RAINY, T, make_forecast
from tests.helpers import make_settings


def codes(report) -> list[str]:  # type: ignore[no-untyped-def]
    return [a.code for a in report.alerts]


def test_dry_mild_forecast_has_no_alerts() -> None:
    r = compute_alerts(DRY, T, NOW)
    assert r.alerts == [] and r.hours_covered == 48 and r.restrictions == []


def test_rain_total_at_threshold_is_inclusive_and_blocks_spray_and_harvest() -> None:
    f = make_forecast(rain={30: 5.0, 31: 5.0})  # exactly 10.0
    r = compute_alerts(f, T, NOW)
    assert codes(r) == ["rain_48h"]
    assert r.restrictions == ["no_harvest", "no_spraying"]
    a = r.alerts[0]
    assert (
        a.threshold == 10
        and a.params["total_mm"] == 10.0
        and a.params["triggered_by"] == "total_mm"
    )


def test_rain_just_below_threshold_does_not_alert() -> None:
    assert compute_alerts(make_forecast(rain={30: 9.9}), T, NOW).alerts == []


def test_rain_window_edges() -> None:
    # window = 11:00 Oct 31 .. 11:00 Nov 2  => hours 11..58 from local midnight
    assert codes(compute_alerts(make_forecast(rain={58: 12.0}), T, NOW)) == ["rain_48h"]
    assert compute_alerts(make_forecast(rain={59: 12.0}), T, NOW).alerts == []
    assert compute_alerts(make_forecast(rain={10: 12.0}), T, NOW).alerts == []  # already past
    assert codes(compute_alerts(make_forecast(rain={11: 12.0}), T, NOW)) == ["rain_48h"]


def test_probability_trigger_needs_some_forecast_rain() -> None:
    wet = make_forecast(rain={30: 0.5}, prob={30: 60.0})
    r = compute_alerts(wet, T, NOW)
    assert codes(r) == ["rain_48h"] and r.alerts[0].params["triggered_by"] == "probability"
    assert compute_alerts(make_forecast(prob={30: 95.0}), T, NOW).alerts == []  # 0 mm total
    assert compute_alerts(make_forecast(rain={30: 0.5}, prob={30: 59.9}), T, NOW).alerts == []


def test_rain_alert_reports_first_rain_hour() -> None:
    a = compute_alerts(RAINY, T, NOW).alerts[0]
    assert a.params["first_rain_at"] == "2026-11-01T06:00:00" and a.params["total_mm"] == 16.0


def test_heat_and_frost_boundaries_inclusive() -> None:
    assert codes(compute_alerts(make_forecast(temp={20: 40.0}), T, NOW)) == ["heat"]
    assert compute_alerts(make_forecast(temp={20: 39.9}), T, NOW).alerts == []
    assert codes(compute_alerts(make_forecast(temp={20: 2.0}), T, NOW)) == ["frost"]
    assert compute_alerts(make_forecast(temp={20: 2.1}), T, NOW).alerts == []
    assert compute_alerts(make_forecast(temp={100: 45.0, 101: -3.0}), T, NOW).alerts == []


def test_heat_and_frost_do_not_restrict_activities() -> None:
    r = compute_alerts(make_forecast(temp={20: 41.0, 21: 1.0}), T, NOW)
    assert codes(r) == ["heat", "frost"] and r.restrictions == []


def test_thresholds_come_from_arguments_not_constants() -> None:
    strict = Thresholds(
        window_hours=48, rain_total_mm=1, rain_prob_pct=10, heat_max_c=30, frost_min_c=25
    )
    assert set(codes(compute_alerts(make_forecast(rain={30: 2.0}), strict, NOW))) == {
        "rain_48h",
        "frost",
    }
    short = Thresholds(
        window_hours=12, rain_total_mm=10, rain_prob_pct=60, heat_max_c=40, frost_min_c=2
    )
    assert compute_alerts(RAINY, short, NOW).alerts == []  # rain is beyond a 12 h window


def test_incomplete_forecast_is_flagged_not_silently_safe() -> None:
    r = compute_alerts(make_forecast(hours=20), T, NOW)  # only 9 hours remain after 11:00
    assert codes(r) == ["forecast_incomplete"] and r.alerts[0].params["covered"] == 9


def test_stale_forecast_is_flagged_with_age() -> None:
    old = NOW - dt.timedelta(hours=5)
    r = compute_alerts(make_forecast(status="stale", at=old), T, NOW)
    assert codes(r) == ["forecast_stale"] and r.alerts[0].params["age_hours"] == 5.0
    assert r.alerts[0].cache_status == "stale"


def test_window_follows_forecast_timezone() -> None:
    late = dt.datetime(2026, 10, 31, 20, 0, tzinfo=dt.UTC)  # 01:30 IST on Nov 1
    r = compute_alerts(make_forecast(rain={24: 12.0}), T, late)  # Nov 1 00:00 local is in the past
    assert r.window_start == dt.datetime(2026, 11, 1, 1, 0) and r.alerts == []


def test_naive_now_rejected() -> None:
    with pytest.raises(ValueError):
        compute_alerts(DRY, T, dt.datetime(2026, 10, 31, 5, 30))


def test_pure_and_deterministic() -> None:
    assert compute_alerts(RAINY, T, NOW) == compute_alerts(RAINY, T, NOW)
    assert RAINY.model_copy() == RAINY  # input is not mutated


def test_alert_carries_provenance() -> None:
    a = compute_alerts(RAINY, T, NOW).alerts[0]
    assert (a.source, a.cache_status, a.forecast_fetched_at) == (
        "Open-Meteo",
        "live",
        RAINY.provenance.fetched_at,
    )


def test_thresholds_from_settings_fail_closed() -> None:
    with pytest.raises(NotConfigured, match="GUARD_RAIN_TOTAL_MM_48H.*GUARD_FROST_MIN_C"):
        Thresholds.from_settings(make_settings(guard_rain_prob_pct=60, guard_heat_max_c=40))
    ok = make_settings(
        guard_rain_total_mm_48h=10, guard_rain_prob_pct=60, guard_heat_max_c=40, guard_frost_min_c=2
    )
    assert Thresholds.from_settings(ok) == T


def test_blank_guard_env_values_mean_unset() -> None:
    s = make_settings(guard_rain_total_mm_48h="", guard_heat_max_c=" ")
    assert s.guard_rain_total_mm_48h is None and s.guard_heat_max_c is None
