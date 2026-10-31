"""POST /ask end to end through FastAPI with fake providers and a scripted LLM."""

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.deps import (
    get_app_settings,
    get_clock,
    get_geocoder,
    get_llm,
    get_mandi,
    get_retriever_factory,
    get_soil,
    get_weather,
)  # fmt: skip
from app.main import app
from tests.agent_helpers import (
    NOW,
    FakeGeocoder,
    FakeMandi,
    FakeRetriever,
    FakeSoil,
    FakeWeather,
    ScriptedLLM,
    call,
    final_json,
    text,
    tools_turn,
)  # fmt: skip
from tests.helpers import make_settings
from tests.test_agent import ANSWER_A, SCHEDULE_A, SELL_A

GUARDED = make_settings(
    guard_rain_total_mm_48h=10, guard_rain_prob_pct=60, guard_heat_max_c=40, guard_frost_min_c=2
)


@pytest.fixture
def wire() -> Iterator[Any]:
    def setup(settings: Settings, llm: ScriptedLLM) -> None:
        app.dependency_overrides.update(
            {
                get_app_settings: lambda: settings, get_llm: lambda: llm, get_clock: lambda: NOW,
                get_geocoder: lambda: FakeGeocoder(), get_weather: lambda: FakeWeather(),
                get_mandi: lambda: FakeMandi(), get_soil: lambda: FakeSoil(),
                get_retriever_factory: lambda: (lambda: FakeRetriever()),
            }
        )  # fmt: skip

    yield setup
    app.dependency_overrides.clear()


def test_ask_returns_validated_structured_json(wire: Any) -> None:
    llm = ScriptedLLM(
        [
            tools_turn(
                call("geocode", query="Mandya"), call("get_weather"),
                call("get_mandi_prices", commodity="Tomato"), call("search_knowledge", query="harvest"),
            ),
            text(),
            final_json(answer_text=ANSWER_A, schedule=SCHEDULE_A, sell_or_hold=SELL_A),
        ]
    )  # fmt: skip
    wire(GUARDED, llm)
    with TestClient(app) as c:
        r = c.post("/ask", json={"question": "Should I harvest tomatoes this week in Mandya?"})
    body = r.json()
    assert r.status_code == 200
    assert set(body) >= {"answer_text", "language", "alerts", "schedule", "sell_or_hold", "sources",
                         "disclaimer", "degraded", "trace"}  # fmt: skip
    assert (
        body["alerts"][0]["code"] == "rain_48h" and body["schedule"][0]["blocked_by"] == "rain_48h"
    )
    assert body["language"] == "en" and body["degraded"] is False
    assert {t["name"] for t in body["trace"]} >= {
        "get_weather",
        "get_mandi_prices",
        "search_knowledge",
    }


def test_ask_503_when_guard_thresholds_are_not_configured(wire: Any) -> None:
    llm = ScriptedLLM([RuntimeError("the LLM must not be called")])
    wire(make_settings(), llm)
    with TestClient(app) as c:
        r = c.post("/ask", json={"question": "Should I harvest tomatoes?"})
    assert r.status_code == 503 and "GUARD_RAIN_TOTAL_MM_48H" in r.json()["message"]
    assert llm.calls == []


def test_ask_validates_request(wire: Any) -> None:
    wire(GUARDED, ScriptedLLM([text()]))
    with TestClient(app) as c:
        assert c.post("/ask", json={"question": "x"}).status_code == 422
        assert (
            c.post("/ask", json={"question": "hello there", "lat": 120, "lon": 0}).status_code
            == 422
        )
        assert c.post("/ask", json={"question": "hello there", "language": "fr"}).status_code == 422
