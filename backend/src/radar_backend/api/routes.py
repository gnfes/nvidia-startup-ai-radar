from fastapi import APIRouter

from radar_backend.rag.retrieve import hybrid_search

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/rag/search")
def rag_search(q: str) -> list[dict]:
    """Manual smoke-test endpoint for the Phase 3/4 hybrid retrieval pipeline
    — not consumed by any agent yet (that's Phase 5's NVIDIA RAG Agent). Run
    with `uv run fastapi dev src/radar_backend/main.py`, then e.g.
    `curl 'localhost:8000/rag/search?q=How+does+NVIDIA+NIM+work%3F'`.
    """
    return [
        {
            "fonte_titulo": chunk.fonte_titulo,
            "secao": chunk.secao,
            "url_fonte": chunk.url_fonte,
            "categoria": chunk.categoria,
            "tipo": chunk.tipo,
            "relevance_score": chunk.relevance_score,
            "conteudo_chunk": chunk.conteudo_chunk,
        }
        for chunk in hybrid_search(q)
    ]
