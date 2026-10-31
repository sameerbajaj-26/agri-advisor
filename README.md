# Agri Advisor

Multilingual (Hindi / Kannada / English) agritech advisory assistant for Indian farmers.
Voice or text in; live weather, live mandi prices, soil data and ICAR/KVK guidance in;
plain-language answer (text + audio) with a schedule and sell-or-hold advice out.

## Hard rules
1. No mock/hardcoded data in app code. Every number shown comes from a real API call or ingested document. Mocks live only in `backend/tests/`.
2. Source failure => explicit error or a clearly labelled stale cache entry with timestamp.
3. The LLM may only state facts present in tool output, citing source + timestamp.
4. Safety alerts (heavy rain, heat, frost) are deterministic code in `guards.py`.
5. Secrets only in `.env` (keep `.env.example` current).

## Layout
```
backend/    FastAPI app (app/), Alembic (alembic/), tests (tests/)
frontend/   Next.js PWA (empty until a later milestone)
scripts/    ingestion and maintenance scripts
docker-compose.yml   api, postgres, redis; ollama via profile
PROGRESS.md          resume-from-here log (read this first)
```

## Quick start
```bash
cp .env.example .env            # then edit values
docker compose up --build       # api on http://localhost:8000
curl http://localhost:8000/health
docker compose --profile ollama up   # include local Ollama
```

## Local development (without Docker for the API)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
make install
make test
make lint
make run          # needs Postgres reachable at DATABASE_URL
pre-commit install
```

## Debug endpoints (Milestone 1)
`GET /geocode?q=`, `/weather?lat=&lon=`, `/prices?state=&district=&commodity=`, `/soil?lat=&lon=[&ph=&clay=...]`, and (M2) `GET /kb/search?q=&crop=&k=`.
Every response carries `provenance` (source, fetched_at, cache_status = live | cached | stale).
Upstream failure returns HTTP 503 with a plain-language message and no substitute numbers.
Needs `DATA_GOV_API_KEY` (free, data.gov.in) for `/prices`. Live check: `make verify-m1`.

## Crop knowledge base (Milestone 2)
RAG over **real documents only** (ICAR / state agri university / KVK package of practices).
No crop guidance is written by hand anywhere in this repo; if a document is not ingested, the
KB does not know it.

**Install**: `make install` (adds the `rag` extra: PyMuPDF, ChromaDB, sentence-transformers).
First ingest downloads the embedding model (default `intfloat/multilingual-e5-small`; set
`EMBEDDING_MODEL=intfloat/multilingual-e5-base` or `BAAI/bge-m3` for higher quality, then `--rebuild`).

**Add a new source**
1. Open `urls.txt` and add one line: `<URL or local PDF path> | <crop> | <title> | <region>`
   (`crop` lowercase, `general` for multi-crop documents; title and region optional but they
   improve citations and search; titles cannot contain `|`).
2. Run `python scripts/ingest_docs.py` (or `make ingest`). Output lists each document as
   `ingested | unchanged | skipped | failed`.
3. Check it: `curl "localhost:8000/kb/search?q=wheat%20sowing%20time%20Madhya%20Pradesh&crop=wheat"`
   (no API restart needed). Every hit has `citation`, `source_title`, `url`, `page`, `ingest_date`.

**Flags**: `--rebuild` drops and rebuilds the collection (required after changing the embedding
model); `--refresh` re-downloads (otherwise the copy saved in `backend/data/kb_raw/` is used);
`--prune` removes documents no longer listed. Re-running is idempotent: a document is re-ingested
only if its content, crop/title/region, chunking or model changed; its old chunks are replaced.

**Limits (by design)**: scanned / image-only PDFs are logged and skipped (no OCR dependency
approved). PDFs whose text extracts as garbage (legacy fonts) are skipped the same way when
detectable. Tables are read as plain text. Government sites with broken certificates: download the
file in a browser and put its local path in `urls.txt`.

## Agent: `POST /ask` (Milestone 3)
`{"question": "...", "language": "en|hi|kn" (optional), "lat": .., "lon": .. (optional GPS)}` returns
`answer_text, language, alerts, schedule, sell_or_hold, sources, disclaimer, degraded, llm, trace`.

**Division of labour.** The LLM only calls tools (geocode, get_weather, get_mandi_prices, get_soil,
search_knowledge) and writes the explanation. Everything safety-relevant is code: `alerts`
(`app/agent/guards.py`, from the real forecast), `language` (script detection), `disclaimer` (fixed
text in en/hi/kn), `sources` (real provenance + timestamps), and `blocked_by` on schedule items that
an alert forbids. Swapping the LLM cannot change the alerts.

**Setup**
1. Pick a model: `LLM_PROVIDER=ollama` + `OLLAMA_MODEL=<a tool-calling model you have pulled>`, or
   `LLM_PROVIDER=hosted` + `HOSTED_LLM_BASE_URL/API_KEY/MODEL` (any OpenAI-compatible API).
2. Set the four `GUARD_*` thresholds in `.env` (see `.env.example`). They have NO defaults: `/ask`
   returns 503 until they are set. The example values are starting points to review with IMD/your
   KVK, not agronomic advice. Each alert echoes the threshold it crossed.
3. Run `python scripts/verify_m3.py "Should I harvest tomatoes this week in <district>?"`.

**What is enforced in code** (the model is retried up to `AGENT_JSON_RETRIES` times, then the response
degrades to alerts + sources only, `degraded: true`): valid JSON schema; every `[S#]` citation exists;
answer must cite when sources exist; numbers next to units (and any number >= 100) must appear in tool
output or alerts; `sell_or_hold` becomes `insufficient_data` unless a price source is cited. Tool
calls are capped at `AGENT_MAX_ITERATIONS` rounds and each is logged with latency (`agri.agent`).

**Alerts** (window `GUARD_WINDOW_HOURS`, default 48 h, from "now" at the forecast location):
`rain_48h` (total mm >= threshold, or hourly probability >= threshold with any rain; restricts
`no_spraying`, `no_harvest`), `heat`, `frost`, plus `forecast_incomplete`, `forecast_stale` and
`no_forecast` so that missing data is never presented as "safe". Pass `lat`/`lon` for exact, repeatable
alerts; otherwise the first geocode match is used.

**Known limits.** The grounding check is best-effort, not proof. Alert-blocking applies to structured
`schedule` items, not to free text. Hindi/Kannada fixed strings need native-speaker review.

## Status
See [PROGRESS.md](PROGRESS.md).
