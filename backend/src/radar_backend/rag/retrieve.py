"""Hybrid retrieval (vector + lexical) over nvidia_kb_chunks, with Cohere
rerank on top (brief §5.3). Written before either nvidia_kb_chunks had data
in Supabase or COHERE_API_KEY was available — see the two "NOT YET
LIVE-VERIFIED" notes below before trusting this against real traffic.

Run after: 0002_nvidia_kb.sql applied + nvidia_kb_seed.sql loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import cohere
from openai import OpenAI

from radar_backend.core.config import get_settings
from radar_backend.db.session import get_connection

VECTOR_CANDIDATES = 20
LEXICAL_CANDIDATES = 20
RRF_K = 60  # standard reciprocal-rank-fusion constant
RERANK_TOP_N = 8


@dataclass
class RetrievedChunk:
    id: str
    categoria: str
    tipo: str
    fonte_titulo: str
    url_fonte: str
    secao: str | None
    conteudo_chunk: str
    relevance_score: float | None = None


def embed_query(text: str) -> list[float]:
    """input_type="query" (as opposed to "passage" in nvidia_kb.py) — same
    asymmetric-retrieval-model reasoning as the ingestion side.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.nvidia_api_key, base_url=settings.nvidia_base_url)
    response = client.embeddings.create(
        input=[text],
        model=settings.nvidia_embedding_model,
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return response.data[0].embedding


def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in vector) + "]"


def _vector_search(query_embedding: list[float], top_k: int) -> list[str]:
    """Returns chunk ids ordered by cosine distance (pgvector's <=> operator;
    ascending = most similar first). NOT YET LIVE-VERIFIED — nvidia_kb_chunks
    has no rows in Supabase yet at the time this was written.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select id
            from nvidia_kb_chunks
            order by embedding <=> %s::vector
            limit %s
            """,
            (_vector_literal(query_embedding), top_k),
        )
        return [row[0] for row in cur.fetchall()]


def _lexical_search(query_text: str, top_k: int) -> list[str]:
    """Postgres full-text search (ts_rank) over the generated tsvector column
    — the lexical half of hybrid search, catches exact terms (product names,
    acronyms like "CUDA" or "NIM") that a pure embedding match can miss.
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select id
            from nvidia_kb_chunks
            where tsv @@ plainto_tsquery('english', %s)
            order by ts_rank(tsv, plainto_tsquery('english', %s)) desc
            limit %s
            """,
            (query_text, query_text, top_k),
        )
        return [row[0] for row in cur.fetchall()]


def _reciprocal_rank_fusion(*ranked_id_lists: list[str]) -> list[str]:
    """Merges independently-ranked id lists into one ranking without needing
    the two scales (cosine distance vs. ts_rank) to be comparable — each list
    only contributes rank position, not its raw score.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, chunk_id in enumerate(ranked_ids):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def _fetch_chunks(chunk_ids: list[str]) -> dict[str, RetrievedChunk]:
    if not chunk_ids:
        return {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            select id, categoria, tipo, fonte_titulo, url_fonte, secao, conteudo_chunk
            from nvidia_kb_chunks
            where id = any(%s)
            """,
            (chunk_ids,),
        )
        return {
            row[0]: RetrievedChunk(
                id=row[0], categoria=row[1], tipo=row[2], fonte_titulo=row[3], url_fonte=row[4], secao=row[5], conteudo_chunk=row[6]
            )
            for row in cur.fetchall()
        }


def rerank(query_text: str, candidates: list[RetrievedChunk], top_n: int = RERANK_TOP_N) -> list[RetrievedChunk]:
    """NOT YET LIVE-VERIFIED — COHERE_API_KEY wasn't available at the time
    this was written. Falls back to returning candidates unranked (already in
    RRF order) if the key is missing, so the retrieval pipeline still works
    end-to-end without it.
    """
    settings = get_settings()
    if not settings.cohere_api_key or not candidates:
        return candidates[:top_n]
    client = cohere.ClientV2(api_key=settings.cohere_api_key)
    response = client.rerank(
        model=settings.cohere_rerank_model,
        query=query_text,
        documents=[c.conteudo_chunk for c in candidates],
        top_n=top_n,
    )
    reranked = []
    for result in response.results:
        chunk = candidates[result.index]
        chunk.relevance_score = result.relevance_score
        reranked.append(chunk)
    return reranked


def hybrid_search(query_text: str, top_n: int = RERANK_TOP_N) -> list[RetrievedChunk]:
    """Vector + lexical candidates -> RRF merge -> Cohere rerank -> top_n
    chunks with full citation metadata (fonte_titulo, secao, url_fonte) —
    the "precise, traceable citations" the league president flagged as a
    credibility differentiator (CLAUDE.md).
    """
    query_embedding = embed_query(query_text)
    vector_ids = _vector_search(query_embedding, VECTOR_CANDIDATES)
    lexical_ids = _lexical_search(query_text, LEXICAL_CANDIDATES)
    merged_ids = _reciprocal_rank_fusion(vector_ids, lexical_ids)

    chunks_by_id = _fetch_chunks(merged_ids)
    candidates = [chunks_by_id[chunk_id] for chunk_id in merged_ids if chunk_id in chunks_by_id]
    return rerank(query_text, candidates, top_n=top_n)
