"""Live check of POST /ask against the running API (uvicorn) and your real LLM.

    python scripts/verify_m3.py "Should I harvest tomatoes this week in Mandya?"
    python scripts/verify_m3.py "..." --lat 12.52 --lon 76.9     # GPS => exact, repeatable alerts
Prints the acceptance evidence: tools called, alerts (computed by code), citations, answer.
"""

import argparse
import json
import os
import sys

import httpx

ap = argparse.ArgumentParser()
ap.add_argument("question", nargs="?", default="Should I harvest tomatoes this week in Mandya?")
ap.add_argument("--lat", type=float)
ap.add_argument("--lon", type=float)
ap.add_argument("--language", choices=["en", "hi", "kn"])
args = ap.parse_args()
body = {k: v for k, v in vars(args).items() if v is not None}
base = os.environ.get("API_URL", "http://127.0.0.1:8000")
r = httpx.post(f"{base}/ask", json=body, timeout=600)
print(f"POST /ask -> {r.status_code}")
if r.status_code != 200:
    print(r.text)
    sys.exit(1)
d = r.json()
print(f"\nLLM: {d['llm']}   language: {d['language']}")
print(f"degraded: {d['degraded']} {d['degraded_reason'] or ''}")
print("\n--- tools called (name, ok, ms) ---")
for t in d["trace"]:
    print(f"  {t['name']:18} ok={t['ok']!s:5} {t['latency_ms']:>6} ms  sources={t['source_ids']}")
print("\n--- ALERTS (computed by code, not by the LLM) ---")
for a in d["alerts"] or [{"code": "none", "message": "(no alerts)", "restrictions": []}]:
    print(f"  [{a['code']}] {a['message']}  restrictions={a['restrictions']}")
print("\n--- sources ---")
for s in d["sources"]:
    print(f"  {s['id']} {'*' if s['cited'] else ' '} {s['kind']:9} {s['name']} | {s['fetched_at']} "
          f"{s['cache_status'] or ''} {s['detail'] or ''}")
print("\n--- answer ---\n" + d["answer_text"])
print("\n--- sell_or_hold ---\n" + json.dumps(d["sell_or_hold"], ensure_ascii=False, indent=2))
print("\n--- schedule ---")
for i in d["schedule"]:
    print(f"  {i['date']} {i['activity']:9} blocked_by={i['blocked_by']}  {i['description']}")
print("\n--- disclaimer ---\n" + d["disclaimer"])
