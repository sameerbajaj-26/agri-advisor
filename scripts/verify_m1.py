#!/usr/bin/env python3
"""Call every debug endpoint of a RUNNING API against the live upstream sources and print a
compact summary. Run:  python scripts/verify_m1.py [--base http://localhost:8000]
Requires the API to be up (docker compose up) with DATA_GOV_API_KEY set for /prices.
"""

import argparse
import json
import sys
from typing import Any

import httpx


def show(title: str, resp: httpx.Response, pick: dict[str, Any] | None = None) -> Any:
    print(f"\n=== {title}  [{resp.status_code}]  {resp.request.url.path}")
    try:
        body = resp.json()
    except ValueError:
        print(resp.text[:500])
        return None
    print(json.dumps(pick(body) if callable(pick) else body, indent=2, ensure_ascii=False)[:2500])
    return body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--place", default="Mandya")
    ap.add_argument("--state", default="Karnataka")
    ap.add_argument("--commodity", default="Tomato")
    a = ap.parse_args()
    ok = True
    with httpx.Client(base_url=a.base, timeout=120) as c:
        geo = show("geocode", c.get("/geocode", params={"q": a.place}))
        if not geo or not geo.get("matches"):
            print("no geocode match; cannot continue")
            return 1
        lat, lon = geo["matches"][0]["latitude"], geo["matches"][0]["longitude"]

        def wx(b: dict[str, Any]) -> dict[str, Any]:
            return {"provenance": b["provenance"], "timezone": b["timezone"],
                    "daily": b["daily"][:3], "hourly_sample": b["hourly"][:3]}

        w = show("weather (live)", c.get("/weather", params={"lat": lat, "lon": lon}), wx)
        w2 = c.get("/weather", params={"lat": lat, "lon": lon}).json()
        print("second weather call cache_status:", w2.get("provenance", {}).get("cache_status"))

        def px(b: dict[str, Any]) -> dict[str, Any]:
            keys = ("provenance", "filters", "price_unit", "total_available", "truncated",
                    "skipped_records", "latest_arrival_date", "days_since_latest")
            return {**{k: b.get(k) for k in keys}, "records_sample": b.get("records", [])[:3]}

        p = show("prices", c.get("/prices", params={"state": a.state, "commodity": a.commodity,
                                                       "max_records": 20}), px)
        s = show("soil (SoilGrids; may be slow)", c.get("/soil", params={"lat": lat, "lon": lon}))
        show("soil + user override ph=6.8", c.get("/soil", params={"lat": lat, "lon": lon, "ph": 6.8}),
             lambda b: [x for x in b["properties"] if x["name"] == "phh2o"])
        ok = all(x is not None and "error" not in x for x in (w, p, s))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
