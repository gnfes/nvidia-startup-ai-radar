"""LangGraph multi-agent pipeline (brief §5.1).

  Query Planner -> Retriever -> Extractor -> Startup Classifier
  -> Evidence Validator -(retry)-> Retriever
                         -(continue)-> NVIDIA RAG -> Recommendation -> Briefing

Node functions here are stubs — each is wired into the graph with the right
state in/out shape and a docstring naming what it will actually do, but no
real DB/LLM calls yet. Filled in one node at a time in later commits, built
on top of the retrieval work already in rag/retrieve.py.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from radar_backend.agents.state import AgentState

MAX_RETRIES = 1  # bounded single retry — see project decision notes


def query_planner(state: AgentState) -> dict:
    """Turns `user_query` into `SearchCriteria` (setor, porte, estagio,
    palavras-chave, sinais de IA) via an LLM call. Stub: no criteria yet.
    """
    return {"search_criteria": {}, "analysis_strategy": None}


def retriever(state: AgentState) -> dict:
    """Selects candidate startups + evidence `documentos` from Postgres per
    `search_criteria` (hybrid_search-style, mirroring rag/retrieve.py but
    over `startups`/`documentos` instead of `nvidia_kb_chunks`). On a retry
    pass (any startup with `evidence_validated=False`), only re-searches for
    those startups, with a broadened query (drop threshold / raise top-k),
    and increments their `retry_count` — this node owns that counter, since
    it's the one spending a retry attempt; `evidence_validator` only ever
    reads it. Stub: passes the batch through unchanged.
    """
    return {"startups": state.get("startups", [])}


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


def nvidia_rag(state: AgentState) -> dict:
    """Runs rag/retrieve.py's hybrid_search per startup, scoped to the
    startup's profile/classification, to find matching NVIDIA KB chunks.
    Stub: no-op per startup.
    """
    return {"startups": state["startups"]}


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
