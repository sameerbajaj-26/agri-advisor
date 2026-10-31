"""Downloaded bytes (PDF or HTML) -> clean per-page text. No OCR: scanned PDFs are rejected."""

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Literal

MIN_PAGE_CHARS = 25  # fewer extractable characters than this => page treated as image-only
MIN_USABLE_PAGE_FRACTION = 0.3  # below this the whole document is treated as scanned
MAX_BAD_CHAR_RATIO = 0.2  # U+FFFD / private-use glyphs => unmapped (legacy) font, text is junk


class UnusableDocument(Exception):
    """The document cannot be turned into trustworthy text. Logged and skipped, never guessed."""


@dataclass(frozen=True)
class ParsedDocument:
    doc_type: Literal["pdf", "html"]
    pages: list[tuple[int, str]]  # (1-based page number, text); HTML is a single page 0
    total_pages: int
    meta_title: str | None


def clean_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\x00", "")
    text = re.sub(r"([a-z])-\n([a-z])", r"\1\2", text)  # un-hyphenate line-end splits
    paras = (" ".join(p.split()) for p in re.split(r"\n\s*\n", text))
    return "\n\n".join(p for p in paras if p)


def _usable(text: str) -> bool:
    if len(text) < MIN_PAGE_CHARS:
        return False
    bad = sum(1 for c in text if c == "\ufffd" or "\ue000" <= c <= "\uf8ff")
    return bad / len(text) <= MAX_BAD_CHAR_RATIO


def parse_document(data: bytes) -> ParsedDocument:
    if data[:1024].lstrip().startswith(b"%PDF") or b"%PDF-" in data[:1024]:
        return _parse_pdf(data)
    head = data[:4096].lower()
    if b"<html" in head or b"<!doctype html" in head or b"<body" in head:
        return _parse_html(data)
    raise UnusableDocument("unsupported content type (expected PDF or HTML)")


def _parse_pdf(data: bytes) -> ParsedDocument:
    import pymupdf  # lazy: only the ingest path and tests need it

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # corrupt / truncated download
        raise UnusableDocument(f"cannot open PDF: {exc}") from exc
    with doc:
        if doc.needs_pass:
            raise UnusableDocument("PDF is password-protected")
        pages: list[tuple[int, str]] = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            text = clean_text(page.get_text("text", sort=True))
            if _usable(text):
                pages.append((page.number + 1, text))
        total = doc.page_count
        title = (doc.metadata or {}).get("title") or None
    if total == 0 or len(pages) / total < MIN_USABLE_PAGE_FRACTION:
        raise UnusableDocument(
            f"scanned/image-only or unreadable text ({len(pages)}/{total} pages have usable "
            "text); OCR is not enabled"
        )
    return ParsedDocument("pdf", pages, total, title.strip() if title else None)


_SKIP = {"script", "style", "noscript", "nav", "header", "footer", "form", "svg", "aside"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "section"}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK:
            self.parts.append("\n\n" if tag != "br" else "\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif not self._skip:
            self.parts.append(data)


def _parse_html(data: bytes) -> ParsedDocument:
    html = data.decode("utf-8", errors="replace")
    extractor = _TextExtractor()
    extractor.feed(html)
    text = clean_text("".join(extractor.parts))
    if not _usable(text):
        raise UnusableDocument("HTML page has no usable text")
    title = " ".join(extractor.title.split()) or None
    return ParsedDocument("html", [(0, text)], 1, title)
