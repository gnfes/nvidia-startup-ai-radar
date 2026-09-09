"""Deterministic rule-mapping cross-check (Phase 8 diferencial).

A static, hand-curated table of NVIDIA product -> keyword signals, matched
against each startup's own profile text — no LLM, no embeddings. Consumed by
the Recommendation node (graph.py) as a second, independent opinion on top
of the LLM's own recommendation:

- when both agree on a product, that agreement is surfaced as
  `concordancia_regras` (two independent methods reached the same
  conclusion);
- when this table flags a product the LLM's recommendation didn't
  specifically confirm, that's surfaced as `alertas_regras` — a candidate
  false negative, which is exactly the failure mode the league president
  said to avoid (see project_league_president_guidance.md). This is a
  keyword heuristic, not a semantic check: an empty `alertas_regras` means
  "the table found nothing to flag," not "there is definitely no gap" — a
  signal in English or an unlisted paraphrase can still slip past both this
  table and the LLM. Bilingual (PT/EN) keywords narrow that gap but don't
  close it.

Deliberately excludes catalog/program entries from kb_sources.py that
aren't actual technologies to recommend (NVIDIA Inception, NVIDIA API
Catalog, the case-study articles, the video sources) — only real products a
startup could adopt.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class ProductRule:
    produto: str  # matches the corresponding fonte_titulo in nvidia_kb_chunks
    sinais: tuple[str, ...]  # lowercase PT/EN keywords; substring match against the profile text


PRODUCT_RULES: tuple[ProductRule, ...] = (
    ProductRule(
        "NVIDIA Riva",
        (
            "voz", "fala", "reconhecimento de voz", "speech to text", "text to speech",
            "atendimento por voz", "transcrição", "call center", "asr", "tts",
            "voice", "speech recognition", "speech ai", "transcription",
        ),
    ),
    ProductRule(
        "NVIDIA NeMo",
        (
            "nlp", "processamento de linguagem natural", "fine-tuning", "modelo customizado",
            "ia conversacional", "chatbot", "atendimento automatizado", "assistente virtual",
            "natural language processing", "conversational ai", "custom model", "virtual assistant",
        ),
    ),
    ProductRule(
        "NeMo Guardrails",
        (
            "guardrails", "moderação de conteúdo", "ia responsável", "compliance de conteúdo",
            "controle de respostas", "segurança de conteúdo",
            "content moderation", "responsible ai", "content safety",
        ),
    ),
    ProductRule(
        "NVIDIA NIM",
        (
            "llm", "modelo de linguagem", "inferência de modelo", "microserviço de ia",
            "deploy de modelo", "ia generativa", "gpt",
            "language model", "model inference", "generative ai", "model deployment",
        ),
    ),
    ProductRule(
        "NVIDIA Triton Inference Server",
        (
            "serving de modelos", "inferência em produção", "mlops", "produção de modelos",
            "escalar modelos",
            "model serving", "production inference", "scaling models",
        ),
    ),
    ProductRule(
        "TensorRT-LLM",
        (
            "latência de inferência", "otimização de inferência", "performance de modelo", "custo de inferência",
            "inference latency", "inference optimization", "model performance",
        ),
    ),
    ProductRule(
        "NVIDIA RAPIDS",
        (
            "big data", "etl", "ciência de dados", "data science", "análise de dados em larga escala",
            "large-scale data analysis",
        ),
    ),
    ProductRule(
        "cuDF",
        ("pandas", "dataframe", "processamento de dados tabulares", "análise tabular", "tabular data"),
    ),
    ProductRule(
        "cuML",
        (
            "scikit-learn", "modelos preditivos", "clustering", "regressão",
            "classificação de dados", "machine learning tradicional",
            "predictive models", "regression", "data classification", "classical machine learning",
        ),
    ),
    ProductRule(
        "CUDA Toolkit",
        (
            "computação gpu", "cuda", "kernel customizado", "processamento paralelo", "computação de alto desempenho",
            "gpu computing", "custom kernel", "parallel processing", "high performance computing",
        ),
    ),
    ProductRule(
        "NVIDIA Omniverse",
        (
            "simulação", "gêmeo digital", "digital twin", "metaverso", "realidade virtual", "design industrial",
            "simulation", "virtual reality", "industrial design",
        ),
    ),
    ProductRule(
        "NVIDIA Isaac",
        (
            "robótica", "robô", "automação física", "manipulação robótica", "veículo autônomo", "logística automatizada",
            "robotics", "robot", "physical automation", "autonomous vehicle", "automated logistics",
        ),
    ),
    ProductRule(
        "NVIDIA Clara",
        (
            "saúde", "healthtech", "imagem médica", "diagnóstico médico", "hospital", "clínica", "medicina",
            "healthcare", "medical imaging", "medical diagnosis", "clinic", "medicine",
        ),
    ),
    ProductRule(
        "NVIDIA Morpheus",
        (
            "segurança cibernética", "cybersecurity", "detecção de fraude", "antifraude", "anomalia", "compliance financeiro",
            "fraud detection", "anti-fraud", "anomaly detection", "financial compliance",
        ),
    ),
    ProductRule(
        "NVIDIA AI Enterprise",
        (
            "plataforma de ia empresarial", "governança de ia", "escala enterprise", "mlops corporativo",
            "enterprise ai platform", "ai governance", "enterprise scale",
        ),
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


def _normalize_words(nome: str) -> str:
    """Lowercases, strips accents, drops "nvidia", and collapses the name to
    space-separated alphanumeric words — e.g. "NVIDIA NeMo" -> "nemo",
    "TensorRT-LLM" -> "tensorrt llm". Mirrors the accent-stripping technique
    in ingestion/resolve.py's `_slugify`, but keeps word boundaries: that
    one concatenates words for domain-name comparison (fine when comparing
    against a URL, which has no spaces), which here would make "cuda" a raw
    substring of "cudatoolkit" with no way to tell a whole-word match from a
    partial one inside an unrelated word.
    """
    ascii_only = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    words = [w for w in _WORD_RE.findall(ascii_only.lower()) if w != "nvidia"]
    return " ".join(words)


def _best_matching_produto(tecnologia: str, produtos: list[str]) -> str | None:
    """Which rule-table product (if any) an LLM-recommended technology
    string most specifically refers to.

    Matches as a whole-word phrase in either direction — so "CUDA" matches
    "CUDA Toolkit" and "Triton" matches "NVIDIA Triton Inference Server" —
    An exact match (after normalization) always wins outright — this matters
    because whole-word partial containment alone can't disambiguate: if the
    LLM writes precisely "NVIDIA NeMo", that must resolve to NVIDIA NeMo
    even though the word "nemo" is also a whole word inside the separate,
    longer "NeMo Guardrails" entry. Only once nothing matches exactly does
    this fall back to whole-word containment in either direction — fixing
    the plain-substring version's blind spot where the LLM's own
    abbreviation was never a substring of the (longer) rule name, e.g.
    "CUDA" still matches "CUDA Toolkit" and "Triton" still matches "NVIDIA
    Triton Inference Server" — picking the longer/more specific name when
    more than one partially matches (e.g. an LLM response of just "NeMo
    Guardrails" partially contains the word "nemo" too, so it technically
    also partially matches "NVIDIA NeMo"; the longer name wins since
    Guardrails genuinely is a distinct product built on NeMo, and a
    recommendation naming it shouldn't also credit the separate, broader
    NeMo entry as agreed-upon).
    """
    tecnologia_norm = _normalize_words(tecnologia)
    if not tecnologia_norm:
        return None

    exact_produto: str | None = None
    best_produto: str | None = None
    best_specificity = 0
    for produto in produtos:
        produto_norm = _normalize_words(produto)
        if not produto_norm:
            continue
        if produto_norm == tecnologia_norm:
            exact_produto = produto
            continue
        contained = re.search(rf"\b{re.escape(produto_norm)}\b", tecnologia_norm) or re.search(
            rf"\b{re.escape(tecnologia_norm)}\b", produto_norm
        )
        if contained:
            specificity = len(produto_norm.split())
            if specificity > best_specificity:
                best_produto, best_specificity = produto, specificity
    return exact_produto or best_produto


def cross_check(rule_matches: list[dict], tecnologias: list) -> tuple[list[str], list[str]]:
    """Cross-references this startup's deterministic rule matches against
    the LLM's recommended technologies. Returns `(concordancia, alertas)`:
    product names both methods point to, and rule matches the LLM's list
    didn't specifically confirm — a candidate false negative to flag.

    Defensively filters `tecnologias` to strings only: `chat_json` does no
    schema validation, so a flaky LLM response (this project's free-tier
    model is documented elsewhere as occasionally unreliable) could return
    `null` or a non-string item for this field, which previously crashed
    this comparison outright instead of just degrading gracefully.
    """
    tecnologias_str = [t for t in (tecnologias or []) if isinstance(t, str)]
    produtos = [m["produto"] for m in rule_matches]

    confirmed = {_best_matching_produto(t, produtos) for t in tecnologias_str}
    confirmed.discard(None)

    concordancia = [p for p in produtos if p in confirmed]
    alertas = [p for p in produtos if p not in confirmed]
    return concordancia, alertas
