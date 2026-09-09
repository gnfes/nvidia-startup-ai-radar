"""Deterministic rule-mapping cross-check (Phase 8 diferencial).

A static, hand-curated table of NVIDIA product -> keyword signals, matched
against each startup's profile text with plain substring search — no LLM,
no embeddings. Runs as its own graph node (`rule_cross_check` in graph.py)
alongside the LLM-driven Recommendation agent, as a second, independent
opinion:

- when both agree on a product, that agreement is surfaced as
  `concordancia_regras` (two independent methods reached the same
  conclusion);
- when this table flags a product the LLM's Recommendation missed, that's
  surfaced as `alertas_regras` — a candidate false negative, which is
  exactly the failure mode the league president said to avoid (see
  project_league_president_guidance.md).

Deliberately excludes catalog/program entries from kb_sources.py that
aren't actual technologies to recommend (NVIDIA Inception, NVIDIA API
Catalog, the case-study articles, the video sources) — only real products a
startup could adopt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProductRule:
    produto: str  # matches the corresponding fonte_titulo in nvidia_kb_chunks
    sinais: tuple[str, ...]  # lowercase keywords; substring match against the profile text


PRODUCT_RULES: tuple[ProductRule, ...] = (
    ProductRule(
        "NVIDIA Riva",
        (
            "voz", "fala", "reconhecimento de voz", "speech to text", "text to speech",
            "atendimento por voz", "transcrição", "call center", "asr", "tts",
        ),
    ),
    ProductRule(
        "NVIDIA NeMo",
        (
            "nlp", "processamento de linguagem natural", "fine-tuning", "modelo customizado",
            "ia conversacional", "chatbot", "atendimento automatizado", "assistente virtual",
        ),
    ),
    ProductRule(
        "NeMo Guardrails",
        (
            "guardrails", "moderação de conteúdo", "ia responsável", "compliance de conteúdo",
            "controle de respostas", "segurança de conteúdo",
        ),
    ),
    ProductRule(
        "NVIDIA NIM",
        (
            "llm", "modelo de linguagem", "inferência de modelo", "microserviço de ia",
            "deploy de modelo", "ia generativa", "gpt",
        ),
    ),
    ProductRule(
        "NVIDIA Triton Inference Server",
        (
            "serving de modelos", "inferência em produção", "mlops", "produção de modelos",
            "escalar modelos",
        ),
    ),
    ProductRule(
        "TensorRT-LLM",
        ("latência de inferência", "otimização de inferência", "performance de modelo", "custo de inferência"),
    ),
    ProductRule(
        "NVIDIA RAPIDS",
        ("big data", "etl", "ciência de dados", "data science", "análise de dados em larga escala"),
    ),
    ProductRule(
        "cuDF",
        ("pandas", "dataframe", "processamento de dados tabulares", "análise tabular"),
    ),
    ProductRule(
        "cuML",
        (
            "scikit-learn", "modelos preditivos", "clustering", "regressão",
            "classificação de dados", "machine learning tradicional",
        ),
    ),
    ProductRule(
        "CUDA Toolkit",
        ("computação gpu", "cuda", "kernel customizado", "processamento paralelo", "computação de alto desempenho"),
    ),
    ProductRule(
        "NVIDIA Omniverse",
        ("simulação", "gêmeo digital", "digital twin", "metaverso", "realidade virtual", "design industrial"),
    ),
    ProductRule(
        "NVIDIA Isaac",
        ("robótica", "robô", "automação física", "manipulação robótica", "veículo autônomo", "logística automatizada"),
    ),
    ProductRule(
        "NVIDIA Clara",
        ("saúde", "healthtech", "imagem médica", "diagnóstico médico", "hospital", "clínica", "medicina"),
    ),
    ProductRule(
        "NVIDIA Morpheus",
        ("segurança cibernética", "cybersecurity", "detecção de fraude", "antifraude", "anomalia", "compliance financeiro"),
    ),
    ProductRule(
        "NVIDIA AI Enterprise",
        ("plataforma de ia empresarial", "governança de ia", "escala enterprise", "mlops corporativo"),
    ),
)


def _profile_haystack(startup_row: dict, structured_profile: dict | None) -> str:
    """Concatenates every text signal available about a specific startup
    into one lowercase string to substring-match rule keywords against.

    Deliberately does NOT include the run's `search_criteria` (the user's
    query, translated into keywords by Query Planner) — that's query intent
    shared by every candidate startup in the batch, not evidence about this
    particular one. Mixing it in was tried and reverted after a live run
    showed it: a "startups de atendimento por voz" query gave every
    retrieved startup an identical "NVIDIA Riva" rule hit, including ones
    with nothing to do with voice (a real-estate marketplace, a BI tool) —
    the opposite of what a credible independent cross-check should do.
    """
    parts: list[str] = [
        startup_row.get("nome") or "",
        startup_row.get("setor") or "",
        startup_row.get("descricao_curta") or "",
    ]

    profile = structured_profile or {}
    parts.append(profile.get("resumo") or "")
    parts.append(profile.get("uso_de_ia") or "")
    parts.extend(str(item) for item in (profile.get("stack_tecnologico") or []))

    return " ".join(parts).lower()


def match_rules(startup_row: dict, structured_profile: dict | None) -> list[dict]:
    """Returns every `ProductRule` whose signals appear in the startup's own
    profile text, as `{"produto": str, "sinais_correspondentes": [str]}`.
    """
    haystack = _profile_haystack(startup_row, structured_profile)
    matches = []
    for rule in PRODUCT_RULES:
        hits = [sinal for sinal in rule.sinais if sinal in haystack]
        if hits:
            matches.append({"produto": rule.produto, "sinais_correspondentes": hits})
    return matches
