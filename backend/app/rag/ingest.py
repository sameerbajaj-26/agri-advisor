"""Ingestion pipeline: urls.txt -> download -> parse -> chunk -> embed -> ChromaDB.

Idempotent: every document has a stable doc_id (hash of its source) and a fingerprint (content +
metadata + chunking + model). Same fingerprint => skipped; changed => old chunks replaced.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import httpx

from app.config import Settings
from app.rag.chunk import chunk_text
from app.rag.embed import Embedder
from app.rag.parse import ParsedDocument, UnusableDocument, parse_document
from app.rag.store import open_collection

log = logging.getLogger("agri.ingest")

CHUNK_MAX_CHARS = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_CHARS = 40
BATCH = 64
PIPELINE_VERSION = "1"  # bump when parsing/chunking logic changes to force re-ingestion

Status = Literal["ingested", "unchanged", "skipped", "failed"]


class UrlsFileError(Exception):
    pass


@dataclass(frozen=True)
class SourceEntry:
    source: str  # http(s) URL, or a local file path for manually downloaded documents
    crop: str  # lowercase crop name, or "general" for multi-crop documents
    title: str | None
    region: str | None
    line_no: int

    @property
    def doc_id(self) -> str:
        return hashlib.sha1(self.source.encode()).hexdigest()[:16]

    @property
    def is_url(self) -> bool:
        return self.source.lower().startswith(("http://", "https://"))


@dataclass(frozen=True)
class DocResult:
    source: str
    status: Status
    chunks: int = 0
    detail: str = ""


def parse_urls_file(path: Path) -> list[SourceEntry]:
    """Format, one per line:  source | crop | title (optional) | region (optional)."""
    if not path.exists():
        raise UrlsFileError(f"{path} does not exist")
    entries: list[SourceEntry] = []
    errors: list[str] = []
    seen: set[str] = set()
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        f = [p.strip() for p in line.split("|")]
        if len(f) < 2 or not f[0] or not f[1]:
            errors.append(f"line {n}: need at least 'source | crop'")
        elif f[0] in seen:
            errors.append(f"line {n}: duplicate source")
        elif len(f) > 4:
            errors.append(f"line {n}: too many '|' fields (max 4; titles cannot contain '|')")
        else:
            seen.add(f[0])
            entries.append(
                SourceEntry(
                    f[0], f[1].lower(), (f[2] if len(f) > 2 else "") or None,
                    (f[3] if len(f) > 3 else "") or None, n,
                )
            )  # fmt: skip
    if errors:
        raise UrlsFileError("; ".join(errors))
    return entries


def fetch_bytes(
    entry: SourceEntry,
    raw_dir: Path,
    client: httpx.Client,
    *,
    refresh: bool = False,
    attempts: int = 3,
) -> bytes:
    """Local path -> read it. URL -> use the saved copy unless refresh, else download."""
    if not entry.is_url:
        p = Path(entry.source).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"local file not found: {p}")
        return p.read_bytes()
    raw_dir.mkdir(parents=True, exist_ok=True)
    saved = raw_dir / f"{entry.doc_id}.bin"
    if saved.exists() and not refresh:
        return saved.read_bytes()
    last: Exception | None = None
    for i in range(attempts):
        try:
            resp = client.get(entry.source)
            resp.raise_for_status()
            if not resp.content:
                raise ValueError("empty response body")
            saved.write_bytes(resp.content)
            return resp.content
        except (httpx.HTTPError, ValueError) as exc:
            last = exc
            time.sleep(min(2.0**i, 8.0))
    raise RuntimeError(f"download failed after {attempts} attempts: {last}")


def _fallback_title(source: str) -> str:
    return Path(source.split("?")[0].rstrip("/")).name or source


def _embedding_text(title: str, crop: str, region: str | None, chunk: str) -> str:
    """Metadata prefix improves recall (a chunk may never mention the state or crop by name).
    Only the raw chunk is stored/returned; the prefix is used for embedding alone."""
    head = f"{title}. Crop: {crop}." + (f" Region: {region}." if region else "")
    return f"{head}\n{chunk}"


def _fingerprint(content: bytes, entry: SourceEntry, model: str) -> str:
    h = hashlib.sha256(content)
    for part in (entry.crop, entry.title, entry.region, model, PIPELINE_VERSION,
                 str(CHUNK_MAX_CHARS), str(CHUNK_OVERLAP)):  # fmt: skip
        h.update(b"\x00" + (part or "").encode())
    return h.hexdigest()[:24]


def _build_records(
    entry: SourceEntry, parsed: ParsedDocument, fingerprint: str, ingest_date: str
) -> tuple[list[str], list[str], list[dict[str, Any]], list[str]]:
    title = entry.title or parsed.meta_title or _fallback_title(entry.source)
    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, Any]] = []
    embed_in: list[str] = []
    for page_no, text in parsed.pages:
        chunks = [c for c in chunk_text(text, CHUNK_MAX_CHARS, CHUNK_OVERLAP)
                  if len(c) >= MIN_CHUNK_CHARS]  # fmt: skip
        for i, chunk in enumerate(chunks):
            ids.append(f"{entry.doc_id}-p{page_no}-c{i}")
            docs.append(chunk)
            embed_in.append(_embedding_text(title, entry.crop, entry.region, chunk))
            metas.append(
                {
                    "doc_id": entry.doc_id,
                    "fingerprint": fingerprint,
                    "source_title": title,
                    "url": entry.source,
                    "page": page_no,  # 0 = web page (no pages)
                    "crop": entry.crop,
                    "region": entry.region or "",
                    "doc_type": parsed.doc_type,
                    "ingest_date": ingest_date,
                }
            )
    return ids, docs, metas, embed_in


def _ingest_one(
    entry: SourceEntry, col: Any, embedder: Embedder, settings: Settings,
    client: httpx.Client, *, refresh: bool, ingest_date: str,
) -> DocResult:  # fmt: skip
    data = fetch_bytes(entry, settings.kb_raw_path, client, refresh=refresh)
    fp = _fingerprint(data, entry, embedder.name)
    existing = col.get(where={"doc_id": entry.doc_id}, limit=1, include=["metadatas"])
    if existing["ids"] and existing["metadatas"][0].get("fingerprint") == fp:
        return DocResult(entry.source, "unchanged")
    try:
        parsed = parse_document(data)
    except UnusableDocument as exc:
        if existing["ids"]:  # a stale copy of a now-unreadable doc must not linger
            col.delete(where={"doc_id": entry.doc_id})
        return DocResult(entry.source, "skipped", detail=str(exc))
    ids, docs, metas, embed_in = _build_records(entry, parsed, fp, ingest_date)
    if not ids:
        return DocResult(entry.source, "skipped", detail="no text chunks produced")
    vectors: list[list[float]] = []
    for i in range(0, len(embed_in), BATCH):  # embed everything BEFORE touching the old data
        vectors.extend(embedder.embed_passages(embed_in[i : i + BATCH]))
    col.delete(where={"doc_id": entry.doc_id})
    for i in range(0, len(ids), BATCH):
        s = slice(i, i + BATCH)
        col.upsert(ids=ids[s], documents=docs[s], metadatas=metas[s], embeddings=vectors[s])
    detail = f"{parsed.total_pages - len(parsed.pages)} unreadable page(s) skipped" \
        if len(parsed.pages) < parsed.total_pages else ""  # fmt: skip
    return DocResult(entry.source, "ingested", len(ids), detail)


def ingest_all(
    entries: list[SourceEntry],
    settings: Settings,
    embedder: Embedder,
    *,
    rebuild: bool = False,
    refresh: bool = False,
    prune: bool = False,
    client: httpx.Client | None = None,
    now: datetime | None = None,
) -> list[DocResult]:
    col = open_collection(settings, embedder.name, create=True, rebuild=rebuild)
    ingest_date = (now or datetime.now(UTC)).date().isoformat()
    own_client = client is None
    http = client or httpx.Client(
        follow_redirects=True,
        timeout=60.0,
        headers={
            "User-Agent": f"agri-advisor-ingest/0.1 ({settings.contact_email or 'no-contact'})"
        },
    )
    results: list[DocResult] = []
    try:
        for e in entries:
            try:
                r = _ingest_one(e, col, embedder, settings, http, refresh=refresh,
                                ingest_date=ingest_date)  # fmt: skip
            except Exception as exc:  # one bad document must not stop the run
                log.exception("failed: %s", e.source)
                r = DocResult(e.source, "failed", detail=f"{type(exc).__name__}: {exc}")
            if r.status == "skipped":
                log.warning("skipped %s: %s", e.source, r.detail)
            results.append(r)
        if prune:
            keep = {e.doc_id for e in entries}
            stale = {m["doc_id"] for m in col.get(include=["metadatas"])["metadatas"]} - keep
            for doc_id in stale:
                col.delete(where={"doc_id": doc_id})
                results.append(DocResult(f"(doc_id {doc_id})", "skipped", detail="pruned"))
    finally:
        if own_client:
            http.close()
    return results
