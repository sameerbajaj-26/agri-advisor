"""ChromaDB access. One persistent collection, cosine space, embedding model recorded."""

import contextlib
from typing import Any

from app.config import Settings
from app.errors import NotConfigured


def _chromadb() -> Any:
    try:
        import chromadb
    except ImportError as exc:
        raise NotConfigured("Knowledge base", 'chromadb missing: pip install -e ".[rag]"') from exc
    return chromadb


def open_collection(
    settings: Settings, embedding_model: str, *, create: bool = False, rebuild: bool = False
) -> Any:
    chromadb = _chromadb()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(settings.chroma_path),
        settings=chromadb.config.Settings(anonymized_telemetry=False),
    )
    name = settings.kb_collection
    if rebuild:
        with contextlib.suppress(Exception):  # nothing to delete on the first build
            client.delete_collection(name)
    if create:
        col = client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine", "embedding_model": embedding_model}
        )
    else:
        try:
            col = client.get_collection(name)
        except Exception as exc:
            raise NotConfigured(
                "Knowledge base", "not built yet - run: python scripts/ingest_docs.py"
            ) from exc
    stored = (col.metadata or {}).get("embedding_model")
    if stored and stored != embedding_model:
        raise NotConfigured(
            "Knowledge base",
            f"built with embedding model '{stored}' but EMBEDDING_MODEL is '{embedding_model}'; "
            "re-run scripts/ingest_docs.py --rebuild (or set EMBEDDING_MODEL back)",
        )
    return col


def kb_version(settings: Settings) -> int:
    """Changes whenever another process (scripts/ingest_docs.py) writes to the store."""
    files = settings.chroma_path.glob("chroma.sqlite3*")
    return max((f.stat().st_mtime_ns for f in files), default=0)


def drop_client_cache() -> None:
    """Chroma keeps one in-process client per path, whose vector index does not see writes made
    by other processes. Clearing it makes the next open_collection() see the new data."""
    try:
        from chromadb.api.client import SharedSystemClient  # type: ignore[attr-defined]

        SharedSystemClient.clear_system_cache()
    except Exception:  # helper missing in this chromadb version: restart API after ingest
        pass
