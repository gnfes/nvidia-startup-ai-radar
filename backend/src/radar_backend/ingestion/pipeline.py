"""Orchestrates discovery + domain resolution + document collection into a
ready-to-run SQL seed file, mirroring the hand-written db/seed/manual_seed.sql
format (fixed uuid literals so documentos can FK into startups without a
round trip through the database).

Run with: uv run python -m radar_backend.ingestion.pipeline
"""

from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from radar_backend.ingestion.collect import RawDocument, fetch_one_secondary_document, fetch_site_institucional
from radar_backend.ingestion.resolve import resolve_site
from radar_backend.ingestion.sources import fetch_distrito_ai_candidates, fetch_wikipedia_company

TARGET_NEW_STARTUPS = 32
MIN_DOCUMENTS = 3

ALREADY_SEEDED_NAMES = {
    "neuralmind", "visioai", "fintalk", "blip", "contabilizei", "loggi",
    "madeiramadeira", "printi",
}

# pt.wikipedia.org article titles curated by hand for AI-enabled / non-AI
# diversity (the Distrito listicle only covers AI-native companies). Verified
# to exist and have a usable infobox before adding here.
WIKIPEDIA_TITLES = [
    "QuintoAndar", "Ebanx", "Creditas", "Méliuz", "Hotmart", "Sympla",
    "ContaAzul", "Wildlife_Studios", "Movile", "Stone_Pagamentos",
    "PagSeguro", "C6_Bank", "Banco_Inter", "Facily", "Petlove", "iFood",
    "Buser", "Netshoes", "CI%26T",
]


@dataclass
class ScrapedStartup:
    nome: str
    site: str
    setor: str | None
    localizacao: str | None
    ano_fundacao: int | None
    descricao_curta: str
    documentos: list[RawDocument] = field(default_factory=list)


def _slug_key(nome: str) -> str:
    return "".join(ch for ch in nome.lower() if ch.isalnum())


def _from_wikipedia() -> list[ScrapedStartup]:
    results: list[ScrapedStartup] = []
    for title in WIKIPEDIA_TITLES:
        company = fetch_wikipedia_company(title)
        if company is None:
            print(f"  [wikipedia] {title}: no usable page, skipping", file=sys.stderr)
            continue
        if _slug_key(company.nome) in ALREADY_SEEDED_NAMES:
            continue

        site = company.site or resolve_site(company.nome)
        if not site:
            print(f"  [wikipedia] {company.nome}: no resolvable site, skipping", file=sys.stderr)
            continue

        docs = [RawDocument(tipo="noticia", titulo=f"{company.nome} (verbete)", conteudo_texto=company.resumo, url_fonte=company.url_fonte)]
        institucional = fetch_site_institucional(site, company.nome)
        if institucional:
            docs.append(institucional)
        secondary = fetch_one_secondary_document(
            site,
            institucional.conteudo_texto if institucional else None,
            institucional.url_fonte if institucional else None,
        )
        if secondary:
            docs.append(secondary)

        if len(docs) < MIN_DOCUMENTS:
            print(f"  [wikipedia] {company.nome}: only {len(docs)} document(s), skipping", file=sys.stderr)
            continue

        results.append(
            ScrapedStartup(
                nome=company.nome,
                site=site,
                setor=company.setor,
                localizacao=company.localizacao,
                ano_fundacao=company.ano_fundacao,
                descricao_curta=company.resumo[:400],
                documentos=docs,
            )
        )
        print(f"  [wikipedia] {company.nome}: OK ({len(docs)} docs)", file=sys.stderr)
    return results


def _from_distrito(limit: int) -> list[ScrapedStartup]:
    results: list[ScrapedStartup] = []
    candidates = fetch_distrito_ai_candidates()
    print(f"  [distrito] {len(candidates)} raw candidates found", file=sys.stderr)

    seen_names: set[str] = set()
    for candidate in candidates:
        if len(results) >= limit:
            break
        key = _slug_key(candidate.nome)
        if key in ALREADY_SEEDED_NAMES or key in seen_names:
            continue
        seen_names.add(key)

        site = resolve_site(candidate.nome)
        if not site:
            print(f"  [distrito] {candidate.nome}: no resolvable site, skipping", file=sys.stderr)
            continue

        docs = [
            RawDocument(
                tipo="noticia",
                titulo="Startups de IA: o que são e principais nomes no Brasil",
                conteudo_texto=f"{candidate.nome} ({candidate.setor}): {candidate.descricao_raw}",
                url_fonte=candidate.url_fonte,
            )
        ]
        institucional = fetch_site_institucional(site, candidate.nome)
        if institucional:
            docs.append(institucional)
        secondary = fetch_one_secondary_document(
            site,
            institucional.conteudo_texto if institucional else None,
            institucional.url_fonte if institucional else None,
        )
        if secondary:
            docs.append(secondary)

        if len(docs) < MIN_DOCUMENTS:
            print(f"  [distrito] {candidate.nome}: only {len(docs)} document(s), skipping", file=sys.stderr)
            continue

        results.append(
            ScrapedStartup(
                nome=candidate.nome,
                site=site,
                setor=candidate.setor,
                localizacao=None,
                ano_fundacao=None,
                descricao_curta=candidate.descricao_raw[:400],
                documentos=docs,
            )
        )
        print(f"  [distrito] {candidate.nome}: OK ({len(docs)} docs) -> {site}", file=sys.stderr)
    return results


def _sql_literal(value: str | int | None) -> str:
    if value is None:
        return "null"
    if isinstance(value, int):
        return str(value)
    return "'" + value.replace("'", "''") + "'"


def render_sql(startups: list[ScrapedStartup]) -> str:
    lines = [
        "-- Seed: scraped startups (discovery: Distrito AI listicle + curated Wikipedia",
        "-- pages for AI-enabled/non-AI diversity). Generated by",
        "-- radar_backend.ingestion.pipeline — see that module for the scraping rules",
        "-- (robots.txt respected, no bot-detection bypass, domain-guess + on-page name",
        "-- verification only, no fabricated data).",
        "-- Run after backend/db/seed/manual_seed.sql.",
        "",
    ]
    for startup in startups:
        startup_id = str(uuid.uuid4())
        lines.append(f"-- {startup.nome} (scraped)")
        lines.append("insert into startups (id, nome, site, setor, localizacao, descricao_curta, ano_fundacao) values")
        lines.append(
            "("
            + ", ".join(
                [
                    _sql_literal(startup_id),
                    _sql_literal(startup.nome),
                    _sql_literal(startup.site),
                    _sql_literal(startup.setor),
                    _sql_literal(startup.localizacao),
                    _sql_literal(startup.descricao_curta),
                    _sql_literal(startup.ano_fundacao),
                ]
            )
            + ");"
        )
        lines.append("")
        lines.append("insert into documentos (startup_id, tipo, titulo, conteudo_texto, url_fonte) values")
        doc_rows = [
            "("
            + ", ".join(
                [
                    _sql_literal(startup_id),
                    _sql_literal(doc.tipo),
                    _sql_literal(doc.titulo),
                    _sql_literal(doc.conteudo_texto),
                    _sql_literal(doc.url_fonte),
                ]
            )
            + ")"
            for doc in startup.documentos
        ]
        lines.append(",\n".join(doc_rows) + ";")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    print("Discovering from Wikipedia (AI-enabled / non-AI diversity)...", file=sys.stderr)
    wiki_results = _from_wikipedia()

    remaining = TARGET_NEW_STARTUPS - len(wiki_results)
    print(f"\nDiscovering from Distrito listicle (AI-native, up to {remaining})...", file=sys.stderr)
    distrito_results = _from_distrito(limit=max(remaining, 0))

    all_results = wiki_results + distrito_results
    print(f"\nTotal scraped startups: {len(all_results)}", file=sys.stderr)

    output_path = Path(__file__).parents[3] / "db" / "seed" / "scraped_batch.sql"
    output_path.write_text(render_sql(all_results), encoding="utf-8")
    print(f"Wrote {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
