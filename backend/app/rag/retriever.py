"""query -> top-k chunks with citations. Reads only what was ingested; invents nothing."""

from typing import Any

from pydantic import BaseModel

from app.config import Settings
from app.errors import NotConfigured
from app.rag.embed import Embedder, get_embedder
from app.rag.store import open_collection

GENERAL_CROP = "general"  # multi-crop documents are ingested with crop = general


class RetrievedChunk(BaseModel):
    rank: int
    score: float  # cosine similarity, 1.0 = identical direction
    text: str
    citation: str  # "<title>, p. <n> - <url>" (page omitted for web pages)
    source_title: str
    url: str
    page: int | None  # None for HTML pages
    crop: str
    region: str | None
    ingest_date: str
    chunk_id: str


def format_citation(title: str, page: int | None, url: str) -> str:
    return f"{title}, p. {page} - {url}" if page else f"{title} - {url}"


class Retriever:
    def __init__(self, collection: Any, embedder: Embedder) -> None:
        self._col = collection
        self._embedder = embedder

    @classmethod
    def from_settings(cls, settings: Settings) -> "Retriever":
        col = open_collection(settings, settings.embedding_model)
        return cls(col, get_embedder(settings.embedding_model))

    @property
    def embedding_model(self) -> str:
        return self._embedder.name

    def search(
        self,
        query: str,
        *,
        crop: str | None = None,
        k: int = 5,
        include_general: bool = True,
        min_score: float | None = None,
    ) -> list[RetrievedChunk]:
        total = self._col.count()
        if total == 0:
            raise NotConfigured(
                "Knowledge base", "no documents ingested - run: python scripts/ingest_docs.py"
            )
        where: dict[str, Any] | None = None
        if crop and crop.strip():
            c = crop.strip().lower()
            where = {"crop": {"$in": [c, GENERAL_CROP]}} if include_general else {"crop": c}
        res = self._col.query(
            query_embeddings=[self._embedder.embed_query(query)],
            n_results=min(max(k, 1), total),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        out: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0],
            res["documents"][0],
            res["metadatas"][0],
            res["distances"][0],
            strict=True,
        ):
            score = 1.0 - float(dist)
            if min_score is not None and score < min_score:
                continue
            page = int(meta["page"]) or None
            out.append(
                RetrievedChunk(
                    rank=len(out) + 1,
                    score=round(score, 4),
                    text=doc,
                    citation=format_citation(meta["source_title"], page, meta["url"]),
                    source_title=meta["source_title"],
                    url=meta["url"],
                    page=page,
                    crop=meta["crop"],
                    region=meta.get("region") or None,
                    ingest_date=meta["ingest_date"],
                    chunk_id=cid,
                )
            )
        return out
