"""Multilingual embeddings via sentence-transformers (multilingual-e5-*, BAAI/bge-m3, ...)."""

from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol

from app.errors import NotConfigured


class Embedder(Protocol):
    name: str

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbedder:
    """Loads the model on first use (slow, ~0.5-2 GB download the first time)."""

    def __init__(self, name: str) -> None:
        self.name = name
        is_e5 = "e5" in name.lower()  # e5 was trained with these prefixes; bge-m3 uses none
        self._passage_prefix = "passage: " if is_e5 else ""
        self._query_prefix = "query: " if is_e5 else ""
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise NotConfigured(
                    "Knowledge base", 'sentence-transformers missing: pip install -e ".[rag]"'
                ) from exc
            model = SentenceTransformer(self.name)
            model.max_seq_length = min(model.max_seq_length or 512, 512)
            self._model = model
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(  # type: ignore[attr-defined]
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vecs]

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return self._encode([self._passage_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_prefix + text])[0]


@lru_cache(maxsize=2)
def get_embedder(name: str) -> Embedder:
    return SentenceTransformerEmbedder(name)
