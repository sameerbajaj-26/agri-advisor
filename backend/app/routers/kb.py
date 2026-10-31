"""Debug endpoint for the crop-knowledge RAG. Returns stored document text verbatim + citations."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from app.deps import get_retriever
from app.rag.retriever import RetrievedChunk, Retriever

router = APIRouter(tags=["debug"])


class KBSearchResponse(BaseModel):
    query: str
    crop: str | None
    embedding_model: str
    count: int
    results: list[RetrievedChunk]


@router.get("/kb/search", response_model=KBSearchResponse)
def kb_search(
    *,
    q: Annotated[str, Query(min_length=2, max_length=500)],
    crop: Annotated[str | None, Query(max_length=60)] = None,
    k: Annotated[int, Query(ge=1, le=20)] = 5,
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> KBSearchResponse:
    # plain `def`: embedding is CPU-bound, FastAPI runs it in the threadpool
    results = retriever.search(q, crop=crop, k=k)
    return KBSearchResponse(
        query=q, crop=crop, embedding_model=retriever.embedding_model,
        count=len(results), results=results,
    )  # fmt: skip
