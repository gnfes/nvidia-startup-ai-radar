"""LangGraph multi-agent pipeline (brief §5.1).

  Query Planner -> Retriever -> Extractor -> Startup Classifier
  -> Evidence Validator -(retry)-> Retriever
                         -(continue)-> NVIDIA RAG -> Recommendation -> Briefing

Nodes are filled in one at a time. Real so far: Retriever (Postgres search
over startups/documentos) and NVIDIA RAG (rag/retrieve.py's hybrid_search).
The rest are still stubs — each is wired into the graph with the right
state in/out shape and a docstring naming what it will actually do, but no
LLM call yet (they all need NVIDIA NIM, not yet integrated in this package).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from radar_backend.agents.state import AgentState, NvidiaChunkMatch, SearchCriteria, new_startup_analysis
from radar_backend.db.session import get_connection
from radar_backend.rag.retrieve import RetrievedChunk, hybrid_search

MAX_RETRIES = 1  # bounded single retry — see project decision notes
RETRIEVER_TOP_K_STARTUPS = 10
# Initial pass ranks each startup's documentos by relevance and caps them;
# a retry pass drops the cap and fetches everything that startup has —
# the "broadened query" the Evidence Validator's retry loop relies on.
RETRIEVER_DOCS_PER_STARTUP = 5
NVIDIA_RAG_TOP_N = 5  # chunks per startup


def query_planner(state: AgentState) -> dict:
    """Turns `user_query` into `SearchCriteria` (setor, porte, estagio,
    palavras-chave, sinais de IA) via an LLM call. Stub: no criteria yet.
    """
    return {"search_criteria": {}, "analysis_strategy": None}


def _search_startups(criteria: SearchCriteria) -> list[dict]:
    """Structured + lexical search over `startups` (no embeddings on this
    table — that's only nvidia_kb_chunks; here it's plain SQL filters plus
    full-text search over descricao_curta/documentos for palavras_chave and
    sinais_ia). Empty criteria matches everything, capped at the top-k.
    """
    conditions: list[str] = []
    params: list = []

    if criteria.get("setor"):
        conditions.append("setor ilike %s")
        params.append(f"%{criteria['setor']}%")
    if criteria.get("estagio"):
        conditions.append("estagio ilike %s")
        params.append(f"%{criteria['estagio']}%")
    if criteria.get("porte"):
        # no dedicated "porte" column — tamanho_time (team size) is the
        # closest proxy the schema has.
        conditions.append("tamanho_time ilike %s")
        params.append(f"%{criteria['porte']}%")

    keywords = [*(criteria.get("palavras_chave") or []), *(criteria.get("sinais_ia") or [])]
    if keywords:
        query_text = " or ".join(keywords)  # websearch_to_tsquery honors "or" as boolean OR
        conditions.append(
            """(
                to_tsvector('portuguese', coalesce(descricao_curta, ''))
                    @@ websearch_to_tsquery('portuguese', %s)
                or id in (
                    select startup_id from documentos
                    where to_tsvector('portuguese', titulo || ' ' || conteudo_texto)
                        @@ websearch_to_tsquery('portuguese', %s)
                )
            )"""
        )
        params.extend([query_text, query_text])

    where_clause = " and ".join(conditions) if conditions else "true"
    params.append(RETRIEVER_TOP_K_STARTUPS)

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            select id, nome, site, setor, estagio, localizacao, descricao_curta,
                   ano_fundacao, tamanho_time
            from startups
            where {where_clause}
            limit %s
            """,
            params,
        )
        columns = [col.name for col in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def _fetch_documentos(startup_id: str, keywords: list[str], *, all_docs: bool) -> list[dict]:
    """Evidence docs for one startup. Ranked-and-capped on the initial pass;
    `all_docs=True` (the retry pass) drops both the ranking and the cap.
    """
    with get_connection() as conn, conn.cursor() as cur:
        if all_docs or not keywords:
            cur.execute(
                """
                select id, tipo, titulo, conteudo_texto, url_fonte, data_publicacao
                from documentos
                where startup_id = %s
                order by data_publicacao desc nulls last
                """,
                (startup_id,),
            )
        else:
            query_text = " or ".join(keywords)
            cur.execute(
                """
                select id, tipo, titulo, conteudo_texto, url_fonte, data_publicacao
                from documentos
                where startup_id = %s
                order by ts_rank(
                    to_tsvector('portuguese', titulo || ' ' || conteudo_texto),
                    websearch_to_tsquery('portuguese', %s)
                ) desc
                limit %s
                """,
                (startup_id, query_text, RETRIEVER_DOCS_PER_STARTUP),
            )
        columns = [col.name for col in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]


def retriever(state: AgentState) -> dict:
    """Selects candidate startups + evidence `documentos` from Postgres per
    `search_criteria`. On the initial pass, searches `startups` and pulls
    each match's top-ranked docs. On a retry pass (any startup left with
    `evidence_validated=False` by the Validator), re-fetches only those
    startups' docs — uncapped and unranked this time — and bumps their
    `retry_count`; this node owns that counter since it's the one spending
    the retry attempt (`evidence_validator` only ever reads it).
    """
    criteria: SearchCriteria = state.get("search_criteria") or {}
    keywords = [*(criteria.get("palavras_chave") or []), *(criteria.get("sinais_ia") or [])]

    if not state.get("startups"):
        rows = _search_startups(criteria)
        startups = [
            new_startup_analysis(
                startup_id=row["id"],
                startup_row=row,
                documentos=_fetch_documentos(row["id"], keywords, all_docs=False),
            )
            for row in rows
        ]
        return {"startups": startups}

    updated = []
    for startup in state["startups"]:
        if not startup.get("evidence_validated"):
            startup = dict(startup)
            startup["documentos"] = _fetch_documentos(startup["startup_id"], keywords, all_docs=True)
            startup["retry_count"] = startup.get("retry_count", 0) + 1
        updated.append(startup)
    return {"startups": updated}


def extractor(state: AgentState) -> dict:
    """Turns each startup's raw `documentos` text into a `structured_profile`
    (stack, use of AI, team signals) via an LLM call. Stub: no-op per
    startup.
    """
    return {"startups": state["startups"]}


def startup_classifier(state: AgentState) -> dict:
    """Classifies each startup as ai_native / ai_enabled / non_ai from its
    `structured_profile`. Stub: no-op per startup.
    """
    return {"startups": state["startups"]}


def evidence_validator(state: AgentState) -> dict:
    """Checks each startup's extracted claims have sufficient evidence/
    sources in `documentos`; sets `evidence_validated` (does not touch
    `retry_count` — that's `retriever`'s job, see its docstring). Stub:
    marks every startup validated, so the graph never actually loops until
    this is filled in.
    """
    updated = [dict(sa, evidence_validated=True) for sa in state["startups"]]
    return {"startups": updated}


def needs_retry(state: AgentState) -> str:
    """Conditional edge out of Evidence Validator: if any startup is still
    unvalidated and has retry budget left, loop the whole batch back to the
    Retriever for one broadened pass; otherwise continue downstream. Bounded
    by MAX_RETRIES so a genuinely evidence-poor startup can't loop forever —
    it just proceeds flagged as low-confidence (Briefing surfaces that).
    """
    for startup in state["startups"]:
        if not startup.get("evidence_validated") and startup.get("retry_count", 0) < MAX_RETRIES:
            return "retry"
    return "continue"


def _build_nvidia_query(state: AgentState, startup: dict) -> str:
    """Query text for the NVIDIA KB search. Prefers the Extractor's
    `structured_profile` (once that node is filled in) since it's the most
    specific signal about the startup's actual stack/gaps; until then, falls
    back to the startup's own descricao_curta/setor plus the raw user query.
    """
    profile = startup.get("structured_profile")
    if profile:
        return " ".join(str(value) for value in profile.values() if value)
    row = startup.get("startup_row") or {}
    parts = [row.get("descricao_curta"), row.get("setor"), state.get("user_query")]
    return " ".join(part for part in parts if part)


def _chunk_to_match(chunk: RetrievedChunk) -> NvidiaChunkMatch:
    return NvidiaChunkMatch(
        id=chunk.id,
        fonte_titulo=chunk.fonte_titulo,
        url_fonte=chunk.url_fonte,
        secao=chunk.secao,
        conteudo_chunk=chunk.conteudo_chunk,
        relevance_score=chunk.relevance_score,
    )


def nvidia_rag(state: AgentState) -> dict:
    """Runs rag/retrieve.py's hybrid_search per startup, scoped to whatever
    profile signal is available for it, to find matching NVIDIA KB chunks.
    """
    updated = []
    for startup in state["startups"]:
        query_text = _build_nvidia_query(state, startup)
        chunks = hybrid_search(query_text, top_n=NVIDIA_RAG_TOP_N) if query_text else []
        startup = dict(startup)
        startup["nvidia_chunks"] = [_chunk_to_match(chunk) for chunk in chunks]
        updated.append(startup)
    return {"startups": updated}


def recommendation(state: AgentState) -> dict:
    """Cross-references each startup's profile + classification with its
    `nvidia_chunks` to build a `Recommendation` (brief §5.5 shape). Stub:
    no-op per startup.
    """
    return {"startups": state["startups"]}


def briefing(state: AgentState) -> dict:
    """Aggregates every startup's recommendation into the final report for
    the Startups & VCs manager, flagging any that stayed low-confidence
    after the retry. Stub: empty report.
    """
    return {"final_briefing": ""}


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("query_planner", query_planner)
    graph.add_node("retriever", retriever)
    graph.add_node("extractor", extractor)
    graph.add_node("startup_classifier", startup_classifier)
    graph.add_node("evidence_validator", evidence_validator)
    graph.add_node("nvidia_rag", nvidia_rag)
    graph.add_node("recommendation", recommendation)
    graph.add_node("briefing", briefing)

    graph.add_edge(START, "query_planner")
    graph.add_edge("query_planner", "retriever")
    graph.add_edge("retriever", "extractor")
    graph.add_edge("extractor", "startup_classifier")
    graph.add_edge("startup_classifier", "evidence_validator")
    graph.add_conditional_edges(
        "evidence_validator",
        needs_retry,
        {"retry": "retriever", "continue": "nvidia_rag"},
    )
    graph.add_edge("nvidia_rag", "recommendation")
    graph.add_edge("recommendation", "briefing")
    graph.add_edge("briefing", END)

    return graph.compile()
