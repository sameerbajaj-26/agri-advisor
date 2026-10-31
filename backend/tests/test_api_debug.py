from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from app.cache import DiskCache
from app.deps import get_app_settings, get_context
from app.main import app
from app.providers.base import ProviderContext
from app.providers.http import RetryPolicy
from tests.helpers import FakeClock, make_settings, no_sleep, weather_payload


def _install(handler: httpx.MockTransport, tmp_path: Path, **settings: object) -> None:
    ctx = ProviderContext(
        client=httpx.AsyncClient(transport=handler),
        cache=DiskCache(str(tmp_path / "c")),
        retry=RetryPolicy(attempts=2, base_delay=0.0),
        clock=FakeClock(),
        sleep=no_sleep,
    )
    app.dependency_overrides[get_context] = lambda: ctx
    app.dependency_overrides[get_app_settings] = lambda: make_settings(**settings)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def test_weather_endpoint_trims_hourly_and_has_provenance(tmp_path: Path) -> None:
    _install(httpx.MockTransport(lambda _: httpx.Response(200, json=weather_payload())), tmp_path)
    with TestClient(app) as client:
        r = client.get("/weather", params={"lat": 12.97, "lon": 77.59, "hourly_hours": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["provenance"]["cache_status"] == "live"
    assert body["provenance"]["fetched_at"] and len(body["hourly"]) == 2


def test_upstream_failure_becomes_clear_503(tmp_path: Path) -> None:
    def down(_: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    _install(httpx.MockTransport(down), tmp_path)
    with TestClient(app) as client:
        r = client.get("/weather", params={"lat": 12.97, "lon": 77.59})
    assert r.status_code == 503
    body = r.json()
    assert body["error"] == "source_unavailable" and body["kind"] == "timeout"
    assert "No substitute values" in body["message"]


def test_prices_without_key_is_explicit(tmp_path: Path) -> None:
    _install(httpx.MockTransport(lambda _: httpx.Response(200, json={})), tmp_path)
    with TestClient(app) as client:
        r = client.get("/prices", params={"state": "Karnataka"})
    assert r.status_code == 503 and r.json()["kind"] == "not_configured"


def test_input_validation(tmp_path: Path) -> None:
    _install(httpx.MockTransport(lambda _: httpx.Response(200, json={})), tmp_path)
    with TestClient(app) as client:
        assert client.get("/weather", params={"lat": 120, "lon": 77}).status_code == 422
        assert client.get("/geocode", params={"q": "a"}).status_code == 422
