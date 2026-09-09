"""Shared state schema for the Phase 5 LangGraph pipeline (brief §5.1).

One `AgentState` flows through all 8 nodes:
  Query Planner -> Retriever -> Extractor -> Startup Classifier
  -> Evidence Validator -> NVIDIA RAG -> Recommendation -> Briefing

A single query can match multiple startups (§5.1: Retriever "seleciona as
*empresas*" — plural). Each matched startup gets its own `StartupAnalysis`
record; nodes update the list with a plain loop (no LangGraph parallel
branching) and Briefing aggregates all of them into one report.
"""

from __future__ import annotations

from typing import Literal, TypedDict

ClassificationLabel = Literal["ai_native", "ai_enabled", "non_ai"]
PriorityLevel = Literal["alta", "media", "baixa"]
ComplexityLevel = Literal["baixa", "media", "alta"]


class SearchCriteria(TypedDict, total=False):
    """Query Planner output — filters over `startups` (brief §5.1)."""

    setor: str | None
    porte: str | None
    estagio: str | None
    palavras_chave: list[str]
    sinais_ia: list[str]


class NvidiaChunkMatch(TypedDict):
    """One retrieved NVIDIA KB chunk (mirrors rag.retrieve.RetrievedChunk)."""

    id: str
    fonte_titulo: str
    url_fonte: str
    secao: str | None
    conteudo_chunk: str
    relevance_score: float | None


class RuleMatch(TypedDict):
    """One deterministic rule hit from agents.rules.match_rules — a NVIDIA
    product whose keyword signals appeared in the startup's profile text.
    """

    produto: str
    sinais_correspondentes: list[str]


class Recommendation(TypedDict):
    """Recommendation Agent output — brief §5.5's required fields, plus two
    Phase 8 additions from the deterministic rule cross-check (additive,
    doesn't change or remove any §5.5 field): `concordancia_regras` (products
    both the LLM and the rule table agreed on) and `alertas_regras`
    (products the rule table flagged that the LLM's recommendation missed —
    a candidate false negative to review manually).
    """

    tecnologias_recomendadas: list[str]
    justificativa_tecnica: str
    justificativa_negocio: str
    nivel_prioridade: PriorityLevel
    complexidade_implementacao: ComplexityLevel
    proxima_acao: str
    evidencias: list[NvidiaChunkMatch]
    concordancia_regras: list[str]
    alertas_regras: list[str]


class StartupAnalysis(TypedDict, total=False):
    """Per-startup record threaded through the pipeline after Retriever fans
    a query out into candidate startups.
    """

    # Retriever
    startup_id: str
    startup_row: dict  # raw `startups` table row
    documentos: list[dict]  # raw `documentos` rows selected as evidence

    # Extractor
    structured_profile: dict | None

    # Startup Classifier
    classification: ClassificationLabel | None
    classification_reasoning: str | None

    # Evidence Validator
    evidence_validated: bool
    evidence_notes: str | None
    retry_count: int  # bounded at 1 — see graph.py's Validator routing

    # Rule Matcher (Phase 8 diferencial — deterministic cross-check)
    regras_correspondentes: list[RuleMatch]

    # NVIDIA RAG
    nvidia_chunks: list[NvidiaChunkMatch]

    # Recommendation
    recommendation: Recommendation | None


class AgentState(TypedDict):
    """Top-level state threaded through the LangGraph pipeline."""

    user_query: str
    search_criteria: SearchCriteria | None
    analysis_strategy: str | None
    startups: list[StartupAnalysis]
    final_briefing: str | None


def new_startup_analysis(startup_id: str, startup_row: dict, documentos: list[dict]) -> StartupAnalysis:
    """Factory for a `StartupAnalysis` with the defaults every downstream
    node expects to already be set (avoids repeating this boilerplate at
    every call site in graph.py).
    """
    return StartupAnalysis(
        startup_id=startup_id,
        startup_row=startup_row,
        documentos=documentos,
        structured_profile=None,
        classification=None,
        classification_reasoning=None,
        evidence_validated=False,
        evidence_notes=None,
        retry_count=0,
        regras_correspondentes=[],
        nvidia_chunks=[],
        recommendation=None,
    )
