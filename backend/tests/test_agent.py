"""Agent loop: scripted LLMs + fake providers (no network, no real model)."""

import asyncio
import datetime as dt
import json
from typing import Any

from app.agent.i18n import DEGRADED, DISCLAIMER, NO_PRICE_DATA, detect_language
from app.agent.loop import run_agent
from app.agent.schemas import AskRequest, AskResponse
from app.agent.tools import Location, ToolRegistry, summarize_prices
from app.errors import NotConfigured, SourceUnavailable
from tests.agent_helpers import (
    LAT,
    LON,
    NOW,
    RAINY,
    FakeGeocoder,
    FakeMandi,
    FakeRetriever,
    FakeSoil,
    FakeWeather,
    ScriptedLLM,
    T,
    call,
    final_json,
    text,
    tools_turn,
)  # fmt: skip
from tests.helpers import make_settings

SETTINGS = make_settings(agent_max_iterations=6, agent_json_retries=2, tool_max_price_rows=20)
Q = "Should I harvest tomatoes this week in Mandya?"

ANSWER_A = (
    "Open-Meteo forecast fetched 2026-10-31 05:09 UTC shows about 16 mm of rain in the next 48 hours "
    "[S2]. Tomato modal price is ₹2400 per quintal (Agmarknet, 2026-10-31) [S3]. Wait for the rain "
    "to pass [S2]."
)
SCHEDULE_A = [
    {
        "date": "2026-11-01",
        "activity": "harvest",
        "description": "Harvest tomatoes",
        "sources": ["S2"],
    },
    {
        "date": "2026-11-03",
        "activity": "harvest",
        "description": "Harvest after rain",
        "sources": ["S2"],
    },
    {"date": "2026-11-01", "activity": "sell", "description": "Sell at market", "sources": ["S3"]},
]
SELL_A = {"recommendation": "hold", "reasoning": "Latest report 2026-10-31 [S3]", "sources": ["S3"]}


def make_tools(**kw: Any) -> ToolRegistry:
    return ToolRegistry(
        geocoder=kw.get("geocoder", FakeGeocoder()),
        weather=kw.get("weather", FakeWeather()),
        mandi=kw.get("mandi", FakeMandi()),
        soil=FakeSoil(),  # type: ignore[arg-type]
        retriever=lambda: kw.get("retriever", FakeRetriever()),
        settings=SETTINGS,
        location=kw.get("location"),
    )


def run(
    llm: ScriptedLLM, tools: ToolRegistry | None = None, question: str = Q, **req: Any
) -> AskResponse:
    return asyncio.run(
        run_agent(
            AskRequest(question=question, **req), llm=llm, tools=tools or make_tools(),
            thresholds=T, settings=SETTINGS, now=NOW,
        )
    )  # fmt: skip


def llm_a() -> ScriptedLLM:
    return ScriptedLLM(
        [
            tools_turn(
                call("geocode", query="Mandya"),
                call("get_weather", lat=LAT, lon=LON),
                call("get_mandi_prices", commodity="Tomato", state="Karnataka"),
                call("search_knowledge", query="tomato harvest maturity", crop="tomato"),
            ),
            text("I have what I need"),
            final_json(answer_text=ANSWER_A, schedule=SCHEDULE_A, sell_or_hold=SELL_A),
        ],
        name="A",
    )


def stable(r: AskResponse) -> list[dict[str, Any]]:
    """Alerts minus the citation pointer (which depends on tool order, not on the weather)."""
    return [a.model_dump(mode="json", exclude={"source_id"}) for a in r.alerts]


# ---- acceptance -----------------------------------------------------------------------------


def test_harvest_question_calls_weather_prices_and_rag_and_cites_everything() -> None:
    r = run(llm_a())
    assert [t.name for t in r.trace] == [
        "geocode",
        "get_weather",
        "get_mandi_prices",
        "search_knowledge",
    ]
    assert all(t.ok and t.latency_ms >= 0 for t in r.trace)
    assert not r.degraded and r.language == "en" and r.disclaimer == DISCLAIMER["en"]
    by_kind = {s.kind: s for s in r.sources}
    assert set(by_kind) == {"geocode", "weather", "prices", "knowledge"}
    assert by_kind["weather"].fetched_at and by_kind["weather"].cache_status == "live"
    assert by_kind["knowledge"].detail == "p. 3, ingested 2026-10-30"
    assert by_kind["weather"].cited and by_kind["prices"].cited and not by_kind["geocode"].cited
    assert [a.code for a in r.alerts] == ["rain_48h"] and r.alerts[0].source_id == by_kind[
        "weather"
    ].id
    assert "16.0 mm" in r.alerts[0].message


def test_alerts_block_restricted_schedule_items_in_code() -> None:
    r = run(llm_a())
    blocked = [(i.date.isoformat() if i.date else None, i.blocked_by) for i in r.schedule]
    assert blocked == [("2026-11-01", "rain_48h"), ("2026-11-03", None), ("2026-11-01", None)]


def test_swapping_the_llm_does_not_change_alerts() -> None:
    a = run(llm_a())
    b = run(  # different model: other tool order, no geocode/prices/RAG, contradicts the alert
        ScriptedLLM(
            [
                tools_turn(call("get_weather", lat=LAT, lon=LON)),
                text(),
                final_json(answer_text="No rain is expected, harvest today [S1]."),
            ],
            name="B",
        )
    )
    c = run(  # a third model that never calls a tool; GPS comes with the request
        ScriptedLLM([text(), final_json(answer_text="See the alert [S1].")], name="C"),
        lat=LAT,
        lon=LON,
    )
    assert stable(a) == stable(b) == stable(c) != []
    assert {a.llm, b.llm, c.llm} == {"scripted:A", "scripted:B", "scripted:C"}
    assert b.alerts[0].restrictions == ["no_harvest", "no_spraying"]  # LLM cannot talk them away


def test_alerts_use_canonical_location_even_if_llm_passes_other_coordinates() -> None:
    weather = FakeWeather()
    llm = ScriptedLLM(
        [tools_turn(call("geocode", query="Mandya"), call("get_weather", lat=1.0, lon=2.0)), text(),
         final_json(answer_text="Rain [S2].")]
    )  # fmt: skip
    run(llm, make_tools(weather=weather))
    assert set(weather.calls) == {(LAT, LON)}


# ---- prompts and call shape -------------------------------------------------------------------


def test_prompts_forbid_unsourced_facts_and_pass_alerts_to_the_llm() -> None:
    llm = llm_a()
    run(llm)
    system = llm.calls[0][0].content
    assert "NO weather, price, soil or crop-practice facts of your own" in system
    assert "[S2]" in system and "fetched time" in system and "authoritative" in system
    assert "Current date/time in India: 2026-10-31 11:00 IST" in system
    final = llm.calls[-1][-1].content
    assert "Rain expected in the next 48 hours" in final and "no_harvest" in final
    assert llm.tool_counts == [5, 5, 0] and llm.json_modes == [False, False, True]


def test_tool_results_reach_the_llm_with_provenance() -> None:
    llm = llm_a()
    run(llm)
    tool_msgs = {m.name: json.loads(m.content) for m in llm.calls[1] if m.role == "tool"}
    w = tool_msgs["get_weather"]
    assert w["source_id"] == "S2" and w["provenance"]["fetched_at"].startswith("2026-10-31T05:09")
    assert tool_msgs["get_mandi_prices"]["by_date"][0]["modal_price_median"] == 2500.0


# ---- validation, retry, degradation -----------------------------------------------------------


def test_malformed_json_is_retried_then_accepted() -> None:
    llm = ScriptedLLM([text(), text("not json at all"), final_json(answer_text="Rain [S1].")])
    r = run(llm, lat=LAT, lon=LON)
    assert not r.degraded and len(llm.calls) == 3
    assert llm.calls[-1][-1].content.startswith("Rejected:")


def test_invented_numbers_are_rejected() -> None:
    llm = ScriptedLLM(
        [text(), final_json(answer_text="Max temperature will be 35 °C [S1]."),
         final_json(answer_text="About 16 mm of rain is forecast [S1].")]
    )  # fmt: skip
    r = run(llm, lat=LAT, lon=LON)
    assert not r.degraded and "16 mm" in r.answer_text
    assert "35" in llm.calls[-1][-1].content


def test_missing_or_unknown_citations_are_rejected() -> None:
    llm = ScriptedLLM(
        [text(), final_json(answer_text="Rain is coming."), final_json(answer_text="Rain [S9]."),
         final_json(answer_text="Rain [S1].")]
    )  # fmt: skip
    r = run(llm, lat=LAT, lon=LON)
    assert not r.degraded and len(llm.calls) == 4


def test_degrades_to_alerts_plus_sources_when_output_never_validates() -> None:
    llm = ScriptedLLM([text(), text("garbage")])
    r = run(llm, lat=LAT, lon=LON)
    assert r.degraded and r.answer_text == DEGRADED["en"] and "retries" in (r.degraded_reason or "")
    assert [a.code for a in r.alerts] == [
        "rain_48h"
    ] and r.sell_or_hold.recommendation == "insufficient_data"
    assert len(llm.calls) == 1 + 3  # phase 1 + (1 + agent_json_retries) attempts


def test_llm_outage_still_returns_alerts() -> None:
    llm = ScriptedLLM([SourceUnavailable("LLM (ollama)", "network", "connection failed")])
    r = run(llm, lat=LAT, lon=LON)
    assert r.degraded and "LLM unavailable" in (r.degraded_reason or "")
    assert [a.code for a in r.alerts] == ["rain_48h"] and r.disclaimer == DISCLAIMER["en"]


def test_iteration_cap_forces_final_answer() -> None:
    llm = ScriptedLLM(
        [tools_turn(call("get_weather", lat=LAT, lon=LON))] * 6
        + [final_json(answer_text="Rain [S1].")]
    )
    r = run(llm)
    assert len(r.trace) == 6 and len(llm.calls) == 7 and not r.degraded
    assert llm.json_modes[-1] is True and llm.tool_counts[-1] == 0


# ---- honesty when data is missing -------------------------------------------------------------


def test_no_location_means_explicit_no_forecast_alert_not_silence() -> None:
    llm = ScriptedLLM([text(), final_json(answer_text="Which district are you in?")])
    r = run(llm)
    assert [a.code for a in r.alerts] == ["no_forecast"] and r.sources == [] and not r.degraded
    assert "NOT run" in r.alerts[0].message


def test_weather_outage_is_an_error_to_the_llm_and_a_no_forecast_alert() -> None:
    llm = ScriptedLLM(
        [tools_turn(call("geocode", query="Mandya"), call("get_weather")), text(),
         final_json(answer_text="Weather is unavailable right now [S1].")]
    )  # fmt: skip
    r = run(llm, make_tools(weather=FakeWeather(error=True)))
    assert [a.code for a in r.alerts] == ["no_forecast"]
    assert [(t.name, t.ok) for t in r.trace] == [("geocode", True), ("get_weather", False)]
    err = next(json.loads(m.content) for m in llm.calls[1] if m.name == "get_weather")
    assert err["source_unavailable"] is True and "Open-Meteo" in err["error"] and "daily" not in err


def test_stale_forecast_is_labelled_stale_everywhere() -> None:
    from tests.agent_helpers import make_forecast

    old = NOW - dt.timedelta(hours=7)
    w = FakeWeather(make_forecast(status="stale", at=old, rain={30: 12.0}))
    llm = ScriptedLLM(
        [
            tools_turn(call("get_weather", lat=LAT, lon=LON)),
            text(),
            final_json(answer_text="Rain [S1]."),
        ]
    )
    r = run(llm, make_tools(weather=w))
    assert {a.code for a in r.alerts} == {"rain_48h", "forecast_stale"}
    assert r.sources[0].cache_status == "stale"


def test_bad_tool_calls_become_errors_not_crashes() -> None:
    llm = ScriptedLLM(
        [tools_turn(call("nope"), call("get_weather", lat="abc", lon=LON), call("get_mandi_prices")),
         text(), final_json(answer_text="Cannot say.")]
    )  # fmt: skip
    r = run(llm)
    assert [t.ok for t in r.trace] == [False, False, False] and not r.degraded


def test_knowledge_base_missing_is_reported_to_llm() -> None:
    err = NotConfigured("Knowledge base", "not built yet")
    llm = ScriptedLLM(
        [
            tools_turn(call("search_knowledge", query="tomato sowing")),
            text(),
            final_json(answer_text="No info."),
        ]
    )
    r = run(llm, make_tools(retriever=FakeRetriever(error=err)))
    assert r.trace[0].ok is False and r.sources == []


def test_sell_or_hold_requires_cited_price_data() -> None:
    llm = ScriptedLLM(
        [tools_turn(call("get_weather", lat=LAT, lon=LON)), text(),
         final_json(answer_text="Rain [S1].",
                    sell_or_hold={"recommendation": "hold", "reasoning": "feels right", "sources": ["S1"]})]
    )  # fmt: skip
    r = run(llm)
    assert r.sell_or_hold.recommendation == "insufficient_data"
    assert r.sell_or_hold.reasoning == NO_PRICE_DATA["en"]


# ---- language ---------------------------------------------------------------------------------


def test_language_detection_and_localized_disclaimer_and_alerts() -> None:
    assert detect_language("ಈ ವಾರ ಟೊಮೆಟೊ ಕೊಯ್ಲು") == "kn"
    assert detect_language("क्या मैं टमाटर की कटाई करूँ") == "hi"
    assert detect_language("harvest tomato") == "en"
    llm = ScriptedLLM([text(), final_json(answer_text="बारिश की संभावना है [S1]")])
    r = run(llm, question="क्या मैं इस हफ्ते टमाटर की कटाई कर सकता हूँ?", lat=LAT, lon=LON)
    assert (
        r.language == "hi" and r.disclaimer == DISCLAIMER["hi"] and "बारिश" in r.alerts[0].message
    )
    assert "Hindi" in llm.calls[0][0].content
    r2 = run(
        ScriptedLLM([text(), final_json(answer_text="Rain [S1].")]), lat=LAT, lon=LON, language="kn"
    )
    assert (
        r2.language == "kn" and r2.disclaimer == DISCLAIMER["kn"] and "ಮಳೆ" in r2.alerts[0].message
    )


# ---- tools ------------------------------------------------------------------------------------


def test_location_is_fixed_once_and_note_explains_override() -> None:
    weather = FakeWeather()
    tools = make_tools(weather=weather, location=Location(LAT, LON, "user GPS"))

    async def go() -> dict[str, Any]:
        res = await tools.execute(call("get_weather", lat=1.0, lon=2.0))
        return res.content

    content = asyncio.run(go())
    assert weather.calls == [(LAT, LON)] and "user GPS" in content["note"]


def test_price_summary_is_computed_in_code() -> None:
    p = asyncio.run(FakeMandi(modal=(1000.0, 3000.0, 2000.0)).get_prices())
    s = summarize_prices(p, max_rows=2)
    assert s["by_date"][0] == {"date": "2026-10-31", "markets_reporting": 3, "modal_price_min": 1000.0,
                               "modal_price_median": 2000.0, "modal_price_max": 3000.0}  # fmt: skip
    assert len(s["latest_day_markets"]) == 2


def test_tool_specs_are_valid_json_schema_objects() -> None:
    specs = make_tools().specs()
    assert {s.name for s in specs} == {
        "geocode",
        "get_weather",
        "get_mandi_prices",
        "get_soil",
        "search_knowledge",
    }
    for s in specs:
        assert s.parameters["type"] == "object" and "title" not in s.parameters and s.description


def test_rain_fixture_sanity() -> None:
    assert RAINY.hourly[30].precipitation_mm == 4.0
