"""LangGraph multi-agent pipeline (brief §5.1).

  Query Planner -> Retriever -> Extractor -> Startup Classifier
  -> Evidence Validator -(retry)-> Retriever
                         -(continue)-> NVIDIA RAG -> Recommendation -> Briefing

All 8 nodes are real. Retriever and NVIDIA RAG hit Postgres/rag/retrieve.py
directly; Query Planner, Extractor, Startup Classifier, Evidence Validator,
and Recommendation call the shared NVIDIA NIM chat client in agents/llm.py;
Briefing is a plain aggregation of the final state, no LLM call.
"""

from __future__ import annotations

import json

from langgraph.graph import END, START, StateGraph

from radar_backend.agents.llm import chat_json
from radar_backend.agents.state import (
    AgentState,
    NvidiaChunkMatch,
    Recommendation,
    SearchCriteria,
    new_startup_analysis,
)
from radar_backend.db.session import get_connection
from radar_backend.rag.retrieve import RetrievedChunk, hybrid_search

MAX_RETRIES = 1  # bounded single retry — see project decision notes
RETRIEVER_TOP_K_STARTUPS = 10
# Initial pass ranks each startup's documentos by relevance and caps them;
# a retry pass drops the cap and fetches everything that startup has —
# the "broadened query" the Evidence Validator's retry loop relies on.
RETRIEVER_DOCS_PER_STARTUP = 5
NVIDIA_RAG_TOP_N = 5  # chunks per startup


QUERY_PLANNER_SYSTEM_PROMPT = """Você é o Query Planner Agent de um sistema de análise de startups \
brasileiras de IA. Transforme a consulta em linguagem natural do usuário em critérios de busca \
estruturados sobre a base de startups, e defina uma breve estratégia de análise.

Responda APENAS com um objeto JSON válido, sem markdown, sem texto fora do JSON. Formato exato:
{
  "search_criteria": {
    "setor": string ou null,
    "estagio": string ou null,
    "porte": string ou null,
    "palavras_chave": [string],
    "sinais_ia": [string]
  },
  "analysis_strategy": string (1-2 frases em português explicando o que será buscado e por quê)
}

Regras:
- Preencha apenas os campos que a consulta realmente sugere; deixe null/[] quando não houver sinal claro.
- "sinais_ia" são termos que indicam uso de IA (ex: "IA conversacional", "visão computacional").
- "palavras_chave" são termos gerais de busca (setor, produto, dor do usuário)."""


def query_planner(state: AgentState) -> dict:
    """Turns `user_query` into `SearchCriteria` (setor, porte, estagio,
    palavras-chave, sinais de IA) via an LLM call.
    """
    result = chat_json(QUERY_PLANNER_SYSTEM_PROMPT, state["user_query"])
    return {
        "search_criteria": result.get("search_criteria") or {},
        "analysis_strategy": result.get("analysis_strategy"),
    }


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

    # Any criterion that matches contributes candidates — OR, not AND. A
    # free-text "setor" guessed by Query Planner (e.g. "Atendimento por
    # Voz") is unlikely to substring-match the DB's own freeform sector text
    # (e.g. "IA conversacional por voz (atendimento, vendas, cobrança)")
    # even for a startup that's an exact conceptual fit — requiring every
    # field to align via AND turns real matches into false negatives, which
    # directly contradicts the project's "avoid false negatives" priority
    # (found live: an AND version returned 0 results for a query that
    # should have surfaced Fintalk).
    where_clause = " or ".join(conditions) if conditions else "true"
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


def _format_documentos(documentos: list[dict], *, char_limit: int = 1500) -> str:
    if not documentos:
        return "(nenhum documento disponível)"
    parts = []
    for doc in documentos:
        conteudo = (doc.get("conteudo_texto") or "")[:char_limit]
        parts.append(f"[{doc.get('tipo')}] {doc.get('titulo')}\n{conteudo}")
    return "\n\n---\n\n".join(parts)


EXTRACTOR_SYSTEM_PROMPT = """Você é o Extractor Agent de um sistema de análise de startups \
brasileiras de IA. Dado o nome, setor e documentos brutos (site institucional, vagas, notícias, \
releases) de uma startup, extraia um perfil estruturado.

Responda APENAS com JSON válido, sem markdown. Formato exato:
{
  "resumo": string (2-3 frases),
  "stack_tecnologico": [string],
  "uso_de_ia": string (como a IA é usada no produto, ou "não identificado"),
  "depende_apenas_de_apis_externas": bool ou null
}

Baseie-se SOMENTE no conteúdo fornecido; não invente informações que não estejam nos documentos. \
Se a informação não estiver disponível, use null ou lista vazia."""


def extractor(state: AgentState) -> dict:
    """Turns each startup's raw `documentos` text into a `structured_profile`
    (stack, use of AI, team signals) via an LLM call.
    """
    updated = []
    for startup in state["startups"]:
        row = startup.get("startup_row") or {}
        user_prompt = (
            f"Startup: {row.get('nome')}\n"
            f"Setor: {row.get('setor')}\n"
            f"Descrição curta: {row.get('descricao_curta')}\n\n"
            f"Documentos:\n{_format_documentos(startup.get('documentos') or [])}"
        )
        profile = chat_json(EXTRACTOR_SYSTEM_PROMPT, user_prompt)
        startup = dict(startup)
        startup["structured_profile"] = profile
        updated.append(startup)
    return {"startups": updated}


CLASSIFIER_SYSTEM_PROMPT = """Você é o Startup Classifier Agent de um sistema de análise de \
startups brasileiras de IA. Classifique a startup como "ai_native", "ai_enabled" ou "non_ai" \
com base no perfil estruturado fornecido:
- ai_native: IA é o núcleo do produto/diferencial competitivo, com stack técnico proprietário \
ou uso profundo de modelos.
- ai_enabled: usa IA de forma auxiliar/via APIs externas, mas não é o diferencial central.
- non_ai: não há evidência de uso relevante de IA.

Responda APENAS com JSON: {"classification": "ai_native"|"ai_enabled"|"non_ai", \
"classification_reasoning": string em português, 1-2 frases}"""


def startup_classifier(state: AgentState) -> dict:
    """Classifies each startup as ai_native / ai_enabled / non_ai from its
    `structured_profile`.
    """
    updated = []
    for startup in state["startups"]:
        row = startup.get("startup_row") or {}
        profile = startup.get("structured_profile") or {}
        user_prompt = (
            f"Startup: {row.get('nome')}\nSetor: {row.get('setor')}\n"
            f"Perfil extraído: {json.dumps(profile, ensure_ascii=False)}"
        )
        result = chat_json(CLASSIFIER_SYSTEM_PROMPT, user_prompt)
        startup = dict(startup)
        startup["classification"] = result.get("classification")
        startup["classification_reasoning"] = result.get("classification_reasoning")
        updated.append(startup)
    return {"startups": updated}


VALIDATOR_SYSTEM_PROMPT = """Você é o Evidence Validator Agent de um sistema de análise de \
startups brasileiras de IA. Verifique se o perfil estruturado extraído de uma startup é \
adequadamente sustentado pelos documentos brutos fornecidos como evidência: as afirmações do \
perfil realmente batem com o conteúdo dos documentos, e há documentos suficientes (não confie em \
um perfil baseado em zero ou um documento muito curto)?

Responda APENAS com JSON: {"evidence_validated": bool, "evidence_notes": string em português, \
1 frase explicando o motivo}"""


def evidence_validator(state: AgentState) -> dict:
    """Checks each startup's extracted claims have sufficient evidence/
    sources in `documentos`; sets `evidence_validated` (does not touch
    `retry_count` — that's `retriever`'s job, see its docstring).
    """
    updated = []
    for startup in state["startups"]:
        row = startup.get("startup_row") or {}
        profile = startup.get("structured_profile") or {}
        docs = startup.get("documentos") or []
        user_prompt = (
            f"Startup: {row.get('nome')}\n"
            f"Perfil extraído: {json.dumps(profile, ensure_ascii=False)}\n\n"
            f"Documentos disponíveis ({len(docs)}):\n{_format_documentos(docs)}"
        )
        result = chat_json(VALIDATOR_SYSTEM_PROMPT, user_prompt)
        startup = dict(startup)
        startup["evidence_validated"] = bool(result.get("evidence_validated"))
        startup["evidence_notes"] = result.get("evidence_notes")
        updated.append(startup)
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


def _format_nvidia_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(nenhum trecho relevante encontrado na base de conhecimento NVIDIA)"
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk.get('fonte_titulo')} | {chunk.get('secao') or ''}]\n{chunk.get('conteudo_chunk')[:600]}")
    return "\n\n---\n\n".join(parts)


RECOMMENDATION_SYSTEM_PROMPT = """Você é o Recommendation Agent de um sistema que cruza o \
perfil de startups brasileiras de IA com a base de conhecimento de tecnologias NVIDIA (brief §5.5). \
Recomende tecnologias NVIDIA adequadas ao perfil da startup, usando APENAS os trechos da base de \
conhecimento fornecidos como base para a recomendação — não invente tecnologias que não estejam \
nesses trechos.

Responda APENAS com JSON válido, sem markdown. Formato exato:
{
  "tecnologias_recomendadas": [string],
  "justificativa_tecnica": string,
  "justificativa_negocio": string,
  "nivel_prioridade": "alta" | "media" | "baixa",
  "complexidade_implementacao": "baixa" | "media" | "alta",
  "proxima_acao": string (ação sugerida para o time NVIDIA)
}"""


def recommendation(state: AgentState) -> dict:
    """Cross-references each startup's profile + classification with its
    `nvidia_chunks` to build a `Recommendation` (brief §5.5 shape). The LLM
    only judges the textual fields — `evidencias` is set directly from the
    real `nvidia_chunks` retrieved earlier, not regenerated by the model, so
    citations stay traceable rather than risking a hallucinated url/id.
    """
    updated = []
    for startup in state["startups"]:
        row = startup.get("startup_row") or {}
        profile = startup.get("structured_profile") or {}
        chunks = startup.get("nvidia_chunks") or []
        user_prompt = (
            f"Startup: {row.get('nome')}\n"
            f"Classificação: {startup.get('classification')}\n"
            f"Perfil: {json.dumps(profile, ensure_ascii=False)}\n\n"
            f"Trechos da base de conhecimento NVIDIA:\n{_format_nvidia_chunks(chunks)}"
        )
        result = chat_json(RECOMMENDATION_SYSTEM_PROMPT, user_prompt, max_tokens=700)
        startup = dict(startup)
        startup["recommendation"] = Recommendation(
            tecnologias_recomendadas=result.get("tecnologias_recomendadas", []),
            justificativa_tecnica=result.get("justificativa_tecnica", ""),
            justificativa_negocio=result.get("justificativa_negocio", ""),
            nivel_prioridade=result.get("nivel_prioridade", "media"),
            complexidade_implementacao=result.get("complexidade_implementacao", "media"),
            proxima_acao=result.get("proxima_acao", ""),
            evidencias=chunks,
        )
        updated.append(startup)
    return {"startups": updated}


def briefing(state: AgentState) -> dict:
    """Aggregates every startup's recommendation into the final report for
    the Startups & VCs manager, flagging any that stayed low-confidence
    after the retry. Built directly from state (no LLM call) so the report
    can't drift from the actual underlying data.
    """
    lines = [
        "# Relatório de Análise — NVIDIA Startup AI Radar",
        "",
        f"**Consulta:** {state.get('user_query')}",
    ]
    if state.get("analysis_strategy"):
        lines += ["", f"**Estratégia de análise:** {state['analysis_strategy']}"]
    lines.append("")

    for startup in state["startups"]:
        row = startup.get("startup_row") or {}
        rec = startup.get("recommendation") or {}
        low_confidence = not startup.get("evidence_validated")

        heading = row.get("nome", "Startup desconhecida")
        if low_confidence:
            heading += " (evidência limitada — retomar com mais fontes)"
        lines.append(f"## {heading}")
        lines.append(
            f"- Classificação: {startup.get('classification') or 'não classificado'}"
            f" — {startup.get('classification_reasoning') or ''}"
        )
        if rec:
            tecnologias = ", ".join(rec.get("tecnologias_recomendadas") or []) or "nenhuma"
            lines.append(f"- Tecnologias NVIDIA recomendadas: {tecnologias}")
            lines.append(
                f"- Prioridade: {rec.get('nivel_prioridade')} | "
                f"Complexidade de implementação: {rec.get('complexidade_implementacao')}"
            )
            lines.append(f"- Justificativa técnica: {rec.get('justificativa_tecnica')}")
            lines.append(f"- Justificativa de negócio: {rec.get('justificativa_negocio')}")
            lines.append(f"- Próxima ação sugerida: {rec.get('proxima_acao')}")
            evidencias = rec.get("evidencias") or []
            if evidencias:
                fontes = "; ".join(f"{e.get('fonte_titulo')} ({e.get('url_fonte')})" for e in evidencias)
                lines.append(f"- Evidências (base NVIDIA): {fontes}")
        lines.append("")

    return {"final_briefing": "\n".join(lines)}


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
