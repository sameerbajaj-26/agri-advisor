"""RAG tests. Fixture documents contain deliberately meaningless filler words, never agronomy:
the real knowledge base only ever comes from documents listed in urls.txt."""

import hashlib
import math
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.deps import get_app_settings, get_retriever
from app.errors import NotConfigured
from app.main import app
from app.rag.chunk import chunk_text
from app.rag.ingest import (
    SourceEntry,
    UrlsFileError,
    ingest_all,
    parse_urls_file,
)
from app.rag.parse import UnusableDocument, parse_document
from app.rag.retriever import Retriever
from app.rag.store import open_collection
from tests.helpers import make_settings


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: no model download needed in tests."""

    def __init__(self, name: str = "fake-model") -> None:
        self.name = name

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * 256
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 550, 800), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return bytes(data)


def make_scanned_pdf() -> bytes:
    doc = pymupdf.open()
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.clear_with(200)
    for _ in range(3):
        doc.new_page().insert_image(pymupdf.Rect(50, 50, 300, 300), pixmap=pix)
    data = doc.tobytes()
    doc.close()
    return bytes(data)


PAGE1 = "Zebra quartz lantern. Marble orbit whisper. " * 6
PAGE2 = "Violet harbor compass. Ember glacier tundra. Plinth saffron monsoon-free ledger. " * 5


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(
        chroma_dir=str(tmp_path / "chroma"), kb_raw_dir=str(tmp_path / "raw"),
    )  # fmt: skip


def _write(tmp_path: Path, name: str, data: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


def _count(settings: Settings) -> int:
    return int(open_collection(settings, "fake-model").count())


# ---- chunking / parsing ----------------------------------------------------------------------


def test_chunks_respect_max_and_overlap() -> None:
    text = " ".join(f"Sentence number {i} has some filler words." for i in range(60))
    chunks = chunk_text(text, max_chars=300, overlap=80)
    assert len(chunks) > 3 and all(len(c) <= 300 for c in chunks)
    for a, b in zip(chunks, chunks[1:], strict=False):
        assert a.split(". ")[-1].rstrip(".") in b  # last sentence of a is carried into b


def test_chunk_hard_slices_unbroken_text() -> None:
    chunks = chunk_text("x" * 2500, max_chars=1000, overlap=100)
    assert all(len(c) <= 1000 for c in chunks) and len(chunks) == 3


def test_chunk_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", max_chars=10, overlap=10)


def test_parse_pdf_keeps_page_numbers() -> None:
    parsed = parse_document(make_pdf([PAGE1, PAGE2]))
    assert [p for p, _ in parsed.pages] == [1, 2] and parsed.doc_type == "pdf"


def test_scanned_pdf_is_rejected_not_guessed() -> None:
    with pytest.raises(UnusableDocument, match="OCR is not enabled"):
        parse_document(make_scanned_pdf())


def test_html_and_unsupported() -> None:
    html = (
        b"<html><title>T</title><nav>menu</nav><body><p>" + PAGE1.encode() + b"</p></body></html>"
    )
    parsed = parse_document(html)
    assert (
        parsed.doc_type == "html" and parsed.meta_title == "T" and "menu" not in parsed.pages[0][1]
    )
    with pytest.raises(UnusableDocument):
        parse_document(b"\x89PNG....")


# ---- urls.txt --------------------------------------------------------------------------------


def test_urls_file_parsing(tmp_path: Path) -> None:
    f = tmp_path / "urls.txt"
    f.write_text("# c\n\n", encoding="utf-8")
    assert parse_urls_file(f) == []  # comments only => empty (script stops and explains)
    f.write_text(
        "https://a/x.pdf | Wheat | Title | MP\nhttps://a/y.pdf|general\n", encoding="utf-8"
    )
    e = parse_urls_file(f)
    assert (e[0].crop, e[0].title, e[0].region) == ("wheat", "Title", "MP") and e[1].title is None
    f.write_text("https://a/x.pdf\n", encoding="utf-8")
    with pytest.raises(UrlsFileError):
        parse_urls_file(f)
    f.write_text("https://a | w\nhttps://a | w\n", encoding="utf-8")
    with pytest.raises(UrlsFileError, match="duplicate"):
        parse_urls_file(f)


# ---- ingestion -------------------------------------------------------------------------------


def test_ingest_is_idempotent_and_replaces_changed_docs(tmp_path: Path, settings: Settings) -> None:
    src = _write(tmp_path, "a.pdf", make_pdf([PAGE1, PAGE2]))
    entries = [SourceEntry(src, "cropx", "Doc A", "Region R", 1)]
    emb = FakeEmbedder()
    r1 = ingest_all(entries, settings, emb)
    n = _count(settings)
    assert r1[0].status == "ingested" and n == r1[0].chunks > 0
    r2 = ingest_all(entries, settings, emb)
    assert r2[0].status == "unchanged" and _count(settings) == n  # no duplicates
    Path(src).write_bytes(make_pdf(["Entirely different filler words appear here now. " * 8]))
    r3 = ingest_all(entries, settings, emb)
    col = open_collection(settings, "fake-model")
    assert r3[0].status == "ingested"
    assert not any("Zebra" in d for d in col.get()["documents"])  # old chunks gone
    r4 = ingest_all(entries, settings, emb, rebuild=True)
    assert r4[0].status == "ingested" and _count(settings) == r3[0].chunks


def test_metadata_change_triggers_reingest(tmp_path: Path, settings: Settings) -> None:
    src = _write(tmp_path, "a.pdf", make_pdf([PAGE1]))
    ingest_all([SourceEntry(src, "cropx", "T", None, 1)], settings, FakeEmbedder())
    r = ingest_all([SourceEntry(src, "cropy", "T", None, 1)], settings, FakeEmbedder())
    assert r[0].status == "ingested"
    assert open_collection(settings, "fake-model").get()["metadatas"][0]["crop"] == "cropy"


def test_bad_documents_are_logged_and_do_not_stop_the_run(
    tmp_path: Path, settings: Settings
) -> None:
    good = _write(tmp_path, "good.pdf", make_pdf([PAGE1]))
    scanned = _write(tmp_path, "scan.pdf", make_scanned_pdf())
    entries = [
        SourceEntry(scanned, "cropx", None, None, 1),
        SourceEntry(str(tmp_path / "missing.pdf"), "cropx", None, None, 2),
        SourceEntry(good, "cropx", None, None, 3),
    ]
    res = ingest_all(entries, settings, FakeEmbedder())
    assert [r.status for r in res] == ["skipped", "failed", "ingested"]
    assert "OCR is not enabled" in res[0].detail


def test_download_is_cached_and_prune_removes_unlisted(tmp_path: Path, settings: Settings) -> None:
    calls: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        calls.append(str(req.url))
        return httpx.Response(200, content=make_pdf([PAGE1]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    e = SourceEntry("https://example.test/doc.pdf", "cropx", None, None, 1)
    emb = FakeEmbedder()
    ingest_all([e], settings, emb, client=client)
    ingest_all([e], settings, emb, client=client)
    assert len(calls) == 1  # second run used the saved copy
    ingest_all([e], settings, emb, client=client, refresh=True)
    assert len(calls) == 2
    ingest_all([], settings, emb, client=client, prune=True)
    assert _count(settings) == 0


def test_http_failure_is_reported(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.rag.ingest.time.sleep", lambda _: None)
    client = httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(404)))
    e = SourceEntry("https://example.test/gone.pdf", "cropx", None, None, 1)
    r = ingest_all([e], settings, FakeEmbedder(), client=client)
    assert r[0].status == "failed" and "download failed" in r[0].detail


# ---- retriever -------------------------------------------------------------------------------


def _build(tmp_path: Path, settings: Settings) -> Retriever:
    a = _write(tmp_path, "a.pdf", make_pdf([PAGE1, PAGE2]))
    b = _write(tmp_path, "b.pdf", make_pdf(["Quasar pebble drizzle lattice. " * 8]))
    c = _write(tmp_path, "c.pdf", make_pdf(["Thimble canyon velvet anchor. " * 8]))
    ingest_all(
        [
            SourceEntry(a, "cropx", "Doc A", "Region R", 1),
            SourceEntry(b, "cropy", "Doc B", None, 2),
            SourceEntry(c, "general", "Doc C", None, 3),
        ],
        settings, FakeEmbedder(),
    )  # fmt: skip
    return Retriever(open_collection(settings, "fake-model"), FakeEmbedder())


def test_retriever_returns_cited_chunks_and_filters_by_crop(
    tmp_path: Path, settings: Settings
) -> None:
    r = _build(tmp_path, settings)
    top = r.search("violet harbor compass", k=3)[0]
    assert top.source_title == "Doc A" and top.page == 2 and "Doc A, p. 2 - " in top.citation
    assert top.region == "Region R" and top.ingest_date
    only_y = r.search("violet harbor compass", crop="CropY", k=10)
    assert {c.crop for c in only_y} == {"cropy", "general"}  # filter (+ general docs), case-insens.
    assert all(c.crop != "cropx" for c in only_y)
    strict = r.search("anything", crop="cropy", include_general=False, k=10)
    assert {c.crop for c in strict} == {"cropy"}
    assert r.search("violet", min_score=0.99, k=3) == []


def test_empty_or_unbuilt_kb_is_an_explicit_error(settings: Settings) -> None:
    with pytest.raises(NotConfigured, match="not built"):
        open_collection(settings, "fake-model")
    col = open_collection(settings, "fake-model", create=True)
    with pytest.raises(NotConfigured, match="no documents"):
        Retriever(col, FakeEmbedder()).search("x")


def test_embedding_model_mismatch_demands_rebuild(tmp_path: Path, settings: Settings) -> None:
    _build(tmp_path, settings)
    with pytest.raises(NotConfigured, match="--rebuild"):
        open_collection(settings, "another-model")


# ---- endpoint --------------------------------------------------------------------------------


@pytest.fixture
def client_clear() -> Iterator[None]:
    yield
    app.dependency_overrides.clear()


def test_kb_search_endpoint(tmp_path: Path, settings: Settings, client_clear: None) -> None:
    retriever = _build(tmp_path, settings)
    app.dependency_overrides[get_retriever] = lambda: retriever
    with TestClient(app) as c:
        r = c.get("/kb/search", params={"q": "violet harbor compass", "crop": "cropx", "k": 2})
    body = r.json()
    assert r.status_code == 200 and body["count"] == 2
    assert body["results"][0]["citation"].startswith("Doc A, p. 2")


def test_kb_search_503_when_not_built(
    tmp_path: Path, settings: Settings, client_clear: None
) -> None:
    app.dependency_overrides[get_app_settings] = lambda: settings
    with TestClient(app) as c:
        r = c.get("/kb/search", params={"q": "anything"})
    assert r.status_code == 503 and "ingest_docs.py" in r.json()["message"]


def test_api_sees_documents_ingested_by_another_process(
    tmp_path: Path, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Chroma's in-process index goes stale when scripts/ingest_docs.py writes; get_retriever()
    must reopen so the running API needs no restart."""
    import subprocess
    import sys

    import app.deps as deps

    monkeypatch.setattr(deps, "_RETRIEVER", None)
    monkeypatch.setattr(
        Retriever, "from_settings",
        classmethod(lambda cls, st: Retriever(open_collection(st, "fake-model"), FakeEmbedder())),
    )  # fmt: skip
    a = _write(tmp_path, "a.pdf", make_pdf([PAGE1]))
    b = _write(tmp_path, "b.pdf", make_pdf(["Quasar pebble drizzle lattice. " * 8]))
    ingest_all([SourceEntry(a, "cropx", "A", None, 1)], settings, FakeEmbedder())
    assert {c.source_title for c in deps.get_retriever(settings).search("quasar", k=5)} == {"A"}
    code = (
        "from tests.test_rag import *\n"
        f"s = make_settings(chroma_dir={settings.chroma_dir!r},"
        f" kb_raw_dir={settings.kb_raw_dir!r})\n"
        f"ingest_all([SourceEntry({a!r},'cropx','A',None,1),"
        f" SourceEntry({b!r},'cropy','B',None,2)],"
        " s, FakeEmbedder())\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1])
    titles = {c.source_title for c in deps.get_retriever(settings).search("quasar", k=5)}
    assert titles == {"A", "B"}
