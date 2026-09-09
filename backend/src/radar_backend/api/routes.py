from cohere.errors import TooManyRequestsError
from fastapi import APIRouter, HTTPException
from openai import APIError, APITimeoutError
from pydantic import BaseModel

from radar_backend.agents.graph import build_graph
from radar_backend.db.session import get_connection
from radar_backend.rag.retrieve import hybrid_search

router = APIRouter()

_STARTUP_COLUMNS = (
    "id, nome, site, setor, estagio, localizacao, descricao_curta, ano_fundacao, tamanho_time"
)


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


@router.get("/startups")
def list_startups(
    q: str | None = None,
    setor: str | None = None,
    estagio: str | None = None,
    porte: str | None = None,
) -> list[dict]:
    """Browse/filter view for the dashboard (Phase 7) — plain substring
    filters over `startups`, AND'd together (unlike the Retriever's OR
    search in agents/graph.py, which optimizes for recall during analysis;
    here the user is narrowing an already-visible list, so AND is the
    expected browsing behavior). `q` free-texts across nome/descricao_curta.
    """
    conditions: list[str] = []
    params: list = []

    if q:
        conditions.append("(nome ilike %s or descricao_curta ilike %s)")
        params.extend([f"%{q}%", f"%{q}%"])
    if setor:
        conditions.append("setor ilike %s")
        params.append(f"%{setor}%")
    if estagio:
        conditions.append("estagio ilike %s")
        params.append(f"%{estagio}%")
    if porte:
        conditions.append("tamanho_time ilike %s")
        params.append(f"%{porte}%")

    where_clause = " and ".join(conditions) if conditions else "true"

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"select {_STARTUP_COLUMNS} from startups where {where_clause} order by nome",
            params,
        )
        columns = [col.name for col in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


class AnalysisRequest(BaseModel):
    query: str


@router.post("/analysis")
def run_analysis(request: AnalysisRequest) -> dict:
    """Triggers a full run of the Phase 5 LangGraph pipeline for a
    natural-language query and returns the final state. Blocking on purpose
    — a run over the whole dataset takes low minutes, and building an async
    job/polling flow wasn't worth the time against a 5-point UI weight (see
    CLAUDE.md's "Interface web" barema line) for a submission due in a day.
    The frontend shows a loading state while this request is in flight.
    """
    graph = build_graph()
    try:
        result = graph.invoke(
            {
                "user_query": request.query,
                "search_criteria": None,
                "analysis_strategy": None,
                "startups": [],
                "final_briefing": None,
            }
        )
    except (APITimeoutError, APIError, ValueError, TooManyRequestsError) as exc:
        # chat_json() (agents/llm.py) already retries transient NVIDIA NIM
        # errors 3x before giving up (APITimeoutError/APIError for timeouts
        # and bad responses, ValueError for empty/non-JSON content), and
        # rerank() (rag/retrieve.py) retries Cohere 429s up to 8x — this
        # only triggers once one of those is exhausted, i.e. a real
        # outage/degradation, not a single blip. Without this handler the
        # exception propagates unhandled, which Starlette's default error
        # path doesn't reliably attach CORS headers to (observed live: the
        # browser reports a "CORS blocked" network error instead of
        # surfacing any status or message) — an explicit HTTPException goes
        # through the normal response path instead, so the frontend gets a
        # real status + message to show.
        raise HTTPException(
            status_code=503,
            detail="A análise falhou porque um serviço externo (NVIDIA NIM ou Cohere) "
            "não respondeu a tempo (free tier, instabilidade intermitente já observada). "
            "Tente novamente.",
        ) from exc
    return {
        "startups": result["startups"],
        "final_briefing": result["final_briefing"],
    }
