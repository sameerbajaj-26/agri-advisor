# PROGRESS

_Last updated: end of Milestone 3 code (agent + guards unit-tested with scripted LLMs; M2 RAG built but not yet run on real documents; neither run against a real LLM)._

## Current milestone
**Milestone 3: Tool-calling agent with deterministic safety** - code complete and unit-tested with scripted LLMs; NOT yet run against your real LLM / live APIs.
M2 (RAG) is still blocked on `urls.txt` sources + a real-model run (see M2 notes below).

| Sub-step | Status |
|---|---|
| 3.1 `app/llm/{base,ollama,hosted}.py`, selected by LLM_PROVIDER | done, tested (mock transport) |
| 3.2 `app/agent/tools.py` (geocode, get_weather, get_mandi_prices, get_soil, search_knowledge) | done, tested |
| 3.3 `app/agent/guards.py` pure alert functions, thresholds in config | done, 17 boundary tests |
| 3.4 `app/agent/loop.py` iteration cap, per-call latency log, strict prompts, citations | done, tested |
| 3.5 Pydantic structured output + retry + grounding + degrade | done, tested |
| 3.6 Alerts injected by code; LLM only explains | done, test: 3 different scripted LLMs => identical alerts |
| 3.7 `POST /ask`, disclaimer in user's language | done, tested |
| 3.8 Acceptance with a REAL LLM ("harvest tomatoes this week in <district>") | **NOT DONE - run `python scripts/verify_m3.py`** |

## M3 files
`backend/app/llm/*`, `backend/app/agent/{guards,i18n,schemas,grounding,tools,prompts,loop}.py`, `backend/app/routers/ask.py`, `deps.py` (+get_llm, get_clock, get_retriever_factory), `config.py` (+agent + GUARD_* settings), `scripts/verify_m3.py`, tests: `test_guards, test_agent, test_grounding, test_llm_clients, test_api_ask` (+`agent_helpers.py`), `pyproject.toml` (+tzdata dep, ruff per-file ignores), `.env.example` (+GUARD_*, agent).

## M3 decisions
- Guard thresholds have NO code defaults (`GUARD_RAIN_TOTAL_MM_48H, GUARD_RAIN_PROB_PCT, GUARD_HEAT_MAX_C, GUARD_FROST_MIN_C`); unset => /ask 503 before any LLM call. `.env.example` values are review-me starting points. Blank env value = unset.
- Alerts are computed from the forecast of ONE canonical location: request lat/lon > first geocode match > first coordinates used. The weather/soil tools ignore other coordinates the LLM passes (and say so). Residual LLM-dependence: the geocode query text it chooses; pass lat/lon for full determinism.
- Rain alert: total mm in window >= threshold OR (max hourly probability >= threshold AND total > 0). Heat: max temp >= threshold. Frost: min temp <= threshold. Hourly data from the current local hour; only rain restricts (no_spraying, no_harvest); heat/frost inform only.
- Absence of alerts is never "safe": forecast_incomplete / forecast_stale / no_forecast alerts exist. Naive `now` rejected.
- Two-phase LLM use: (1) tool calling, (2) one JSON-mode call with alerts + citable sources in context. Phase 2 validated: schema, known `[S#]` ids, citation present, ungrounded numbers (best effort), retries then degraded response (alerts + sources, `degraded: true`). LLM outage also degrades but still returns alerts.
- `sell_or_hold` forced to `insufficient_data` unless a prices source is cited. Schedule items for harvest/spray inside an alert window get `blocked_by`.
- Source ids S1.. are per request; each RAG chunk has its own id. Prices summarised in code (by_date min/median/max).
- Language: script detection (Kannada/Devanagari/else English) or request `language`; disclaimer and alert texts are fixed strings in en/hi/kn (need native review). The LLM writes answer_text in that language.

## Known gaps / risks (M3)
- Never run against a real model: tool-calling quality, JSON adherence and latency of your chosen Ollama model are unknown. Small models may degrade often; check `degraded_reason` and the `agri.agent` log.
- Number-grounding is heuristic (units / values >= 100); it can reject correct rounding and miss unitless claims. Free-text answer is not scanned for "go harvest" contradictions; only structured schedule is blocked.
- Multi-location questions are out of scope (one canonical location per request).
- Need `pip install -e ".[dev,rag]"` again (new dep: tzdata). Add the GUARD_* lines to your own `.env`.
- Pre-existing: 3 failing tests in `tests/test_config.py` (see M2 notes).

## NEXT ACTION
1. Add the GUARD_* + OLLAMA_MODEL (or hosted) lines to `.env`; restart uvicorn.
2. `python scripts/verify_m3.py "Should I harvest tomatoes this week in <your district>?" --lat <lat> --lon <lon>`; paste the output.
3. Still pending from M2: real sources in `urls.txt`, then `python scripts/ingest_docs.py` and the "wheat sowing time Madhya Pradesh" check.

## Milestone 2 notes (still pending real-data run)
**Milestone 2** - code complete, blocked on YOUR sources in `urls.txt` + a real-model run.

| Sub-step | Status |
|---|---|
| M0 scaffold | running on your machine (/health OK, screenshot) |
| M1 geocode + weather | **verified live** (verify_m1 output: 200, cache_status live) |
| M1 prices + soil | not seen in the output you sent - still unverified |
| 2.1 ingest pipeline (download, PyMuPDF parse, chunk+overlap, embed, Chroma) | done, tested with fake embedder |
| 2.2 retriever + citations + crop filter | done, tested |
| 2.3 scanned / unreadable PDFs logged and skipped | done, tested |
| 2.4 idempotent re-ingestion, `--rebuild`, `--refresh`, `--prune` | done, tested |
| 2.5 `GET /kb/search?q=&crop=&k=` | done, tested (503 with clear message if KB not built) |
| 2.6 README "add a source" | done |
| 2.7 Done-when: "wheat sowing time Madhya Pradesh" returns real chunks | **NOT DONE - needs urls.txt + real embedding model on your machine** |

## M2 files
- `urls.txt` (repo root, no entries) - `source | crop | title | region`
- `scripts/ingest_docs.py` - CLI; exits 1 with instructions if urls.txt is empty
- `backend/app/rag/{parse,chunk,embed,store,ingest,retriever}.py`
- `backend/app/routers/kb.py`; `deps.py` (get_retriever, auto-reopens after external ingest); `main.py` (router); `config.py` (+kb_raw_dir, kb_collection, embedding_model default, path resolution vs backend/)
- `backend/tests/test_rag.py` (18 tests; fixtures use meaningless filler text, no agronomy)
- `pyproject.toml` (+`rag` extra, mypy overrides), `Makefile` (+ingest, rebuild-kb), `Dockerfile` (EXTRAS arg, default dev,rag), `.env.example` (EMBEDDING_MODEL)

## M2 decisions
- Embedding default `intfloat/multilingual-e5-small` (CPU-friendly, multilingual incl. Hindi/Kannada). e5 prefixes (`query: `/`passage: `) added automatically; bge-m3 works with no prefix. Model name stored in the collection; mismatch => explicit error telling you to `--rebuild`.
- Chunks: ~1000 chars, 150 overlap, sentence-aware (incl. Devanagari danda), never cross pages (exact page citations). HTML pages have page 0 (citation omits page).
- Embedded text = "title. Crop. Region." + chunk (helps queries naming a state/crop absent from the chunk); only the raw chunk is stored/returned.
- `crop` filter also returns `general` docs (multi-crop) by default.
- Idempotency: doc_id = sha1(source); fingerprint = content + crop/title/region + chunking + model; unchanged => skipped; changed => embed first, then replace old chunks.
- Chroma's in-process index goes stale when another process ingests (verified); `get_retriever` reopens when the store's mtime changes, so no API restart is needed.
- No OCR dependency added. Garbled legacy-font text detected only via U+FFFD / private-use ratio (heuristic).

## Known gaps / risks
- Real embedding model never run in my sandbox (no HuggingFace access): first run downloads it; check memory/speed on your machine.
- Tables in POP PDFs (sowing windows are often tabular) are extracted as plain text; retrieval quality on them is unproven until real docs are ingested. If poor, next step is PyMuPDF `find_tables()`.
- `tests/test_config.py`: 3 tests fail (pre-existing, not from M2): `_settings()` takes no argument but is called as `_settings(_env_file=None)`; also `database_url` now defaults to "" so `test_database_url_is_required` cannot pass.
- Debug endpoints (incl. /kb/search) are always mounted; gate before production.

## M2 next action
1. Add real sources to `urls.txt` (at least one wheat document covering Madhya Pradesh).
2. `make install` (pulls torch; large), then `python scripts/ingest_docs.py`, then `curl "localhost:8000/kb/search?q=wheat%20sowing%20time%20Madhya%20Pradesh&crop=wheat"`.
3. Paste the ingest summary and the search output; then continue to Milestone 3.

## Commands to verify
```bash
cp .env.example .env     # set POSTGRES_PASSWORD, DATABASE_URL, DATA_GOV_API_KEY, CONTACT_EMAIL
python3.11 -m venv .venv && source .venv/bin/activate
make install
make fmt                 # hand-written code has never been through ruff format
make test && make lint
docker compose up --build            # other shell:
make verify-m1                       # or: python scripts/verify_m1.py --base http://localhost:8000
curl "localhost:8000/geocode?q=Mandya"
curl "localhost:8000/weather?lat=12.52&lon=76.90&hourly_hours=6"
curl "localhost:8000/prices?state=Karnataka&commodity=Tomato&max_records=10"
curl "localhost:8000/soil?lat=12.52&lon=76.90&ph=6.8"
```

## Decisions
- cache_status has THREE values: `live` (fetched now), `cached` (real response inside TTL; see fetched_at), `stale` (upstream failed; older real response served with stale_reason). You asked for live/stale; `cached` avoids calling a 20-minute-old response "live".
- TTL / stale-max: geocode 30d/365d, weather 30m/12h, mandi 3h/3d, soil 30d/365d. Beyond stale-max we raise instead of serving.
- Failures and empty mandi answers are never cached.
- Retries: timeout, connection error, 429, 5xx (Retry-After honoured, capped 30 s); other 4xx fail fast. Default 3 attempts, 0.5 s base backoff.
- Errors never contain URLs (data.gov.in key is a query param). httpx INFO logger muted for the same reason.
- Mandi filter values: all-lower/all-upper input is Title-Cased (e.g. "tomato"->"Tomato"); mixed-case passed through. Filters are exact-match upstream.
- Soil: SoilGrids values are modelled estimates; output says so. User override replaces that property (all depths) and is labelled `user-provided`; override alone is returned if SoilGrids is down (with soilgrids_error); no override + outage => 503.
- Deps added beyond M0: `diskcache` (in your stack list). No pytest-asyncio (tests use asyncio.run + httpx.MockTransport).
- Debug endpoints are always mounted; gate/remove before production.

## API quirks / limits (from docs; UNVERIFIED LIVE)
- data.gov.in mandi resource id `9ef84268-d588-465a-a308-a864a43d0070`; fields state, district, market, commodity, variety, grade, arrival_date (dd/mm/yyyy), min/max/modal_price. Some sources show legacy `Modal_x0020_Price` style names: parser normalises both.
- Filters: `filters[state.keyword]`, `filters[district]`, `filters[commodity]`; paging `offset`/`limit`; envelope `total,count,records`.
- One data.gov.in catalogue page claimed "the API for this resource does not exist" - if live calls fail with 404/empty, check the resource page and update MANDI_RESOURCE_ID.
- A third-party sample showed state "Keralam" (renamed): state/district strings must match what the source emits; a name-resolution layer is a later task.
- Price unit label "INR per quintal" is Agmarknet convention, not confirmed from a live response.
- Nominatim: max 1 req/s, identifying User-Agent, cache, show OSM attribution (included in /geocode output).
- Open-Meteo: free for non-commercial use; check licence before any commercial launch.
- SoilGrids: slow (30 s timeout), a rate limit exists (believed ~5 calls/min, not verified), `d_factor`/`target_units` keys in unit_measure assumed from docs; if absent the raw value is returned with mapped_units.

## Env vars added in M1
HTTP_RETRIES, HTTP_TIMEOUT_S, SLOW_HTTP_TIMEOUT_S, CONTACT_EMAIL, NOMINATIM_BASE_URL, OPEN_METEO_BASE_URL, SOILGRIDS_BASE_URL, DATA_GOV_BASE_URL, MANDI_RESOURCE_ID (plus existing DATA_GOV_API_KEY now used)

## Known bugs / risks
- NOTHING has been executed: expect first-run fixes (ruff format, mypy strict, a test or two).
- No live sample output exists yet; do not trust field mappings until `make verify-m1` passes.
- Lazy per-request provider objects; Nominatim limiter is per-process only.
- weather `hourly_hours` returns the first N hourly rows (from 00:00 local today), not "from now".

## NEXT ACTION
1. Run the verification commands; paste failures and the verify-m1 output.
2. Then "continue" for Milestone 2 (safety guards in guards.py from WeatherForecast, then ICAR/KVK RAG ingestion).
