"""Discovery sources: pages we scrape to find candidate startup names.

Both sources are static, server-rendered HTML (verified by hand before writing
this), so plain requests + BeautifulSoup is enough — no headless browser needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from radar_backend.ingestion.http import polite_get

DISTRITO_AI_LIST_URL = "https://www.distrito.me/blog/startups-de-ia-o-que-sao-e-principais-nomes-no-brasil"
WIKIPEDIA_BASE_URL = "https://pt.wikipedia.org/wiki/"

_CITATION_RE = re.compile(r"\[\s*\d+\s*\]")
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _clean_text(text: str) -> str:
    return _CITATION_RE.sub("", text).strip()


def _clean_domain(text: str) -> str:
    # Wikipedia sometimes renders external-link domains with spaces around dots
    # (e.g. "business .ebanx .com") because of the little arrow icon markup.
    return re.sub(r"\s*\.\s*", ".", text.strip())


@dataclass
class Candidate:
    nome: str
    setor: str
    descricao_raw: str
    url_fonte: str


def fetch_distrito_ai_candidates() -> list[Candidate]:
    """Parses Distrito's "50 principais startups de IA brasileiras" listicle.

    Structure: one <h2> section titled "Lista das 50 principais startups de IA
    brasileiras" containing <h3> sector headings, each followed by a <ul> of
    <li><strong>Nome:</strong> descrição</li> items.
    """
    response = polite_get(DISTRITO_AI_LIST_URL)
    if response is None:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    article = soup.find("article") or soup

    start_heading = None
    for h2 in article.find_all("h2"):
        if "principais startups de ia" in h2.get_text(strip=True).lower():
            start_heading = h2
            break
    if start_heading is None:
        return []

    candidates: list[Candidate] = []
    setor = "Não especificado"
    for element in start_heading.find_next_siblings():
        if element.name == "h2":
            break
        if element.name == "h3":
            setor = element.get_text(strip=True)
            continue
        for li in element.find_all("li") if element.name == "ul" else []:
            label = li.find(["strong", "b"])
            if label is None:
                continue
            nome = label.get_text(strip=True).rstrip(":").strip()
            full_text = li.get_text(" ", strip=True)
            descricao = full_text[len(label.get_text(strip=True)):].lstrip(": ").strip()
            if not nome or not descricao:
                continue
            candidates.append(
                Candidate(nome=nome, setor=setor, descricao_raw=descricao, url_fonte=DISTRITO_AI_LIST_URL)
            )
    return candidates


@dataclass
class WikipediaCompany:
    nome: str
    site: str | None
    setor: str | None
    localizacao: str | None
    ano_fundacao: int | None
    resumo: str
    url_fonte: str


def fetch_wikipedia_company(title: str) -> WikipediaCompany | None:
    """Parses a pt.wikipedia.org company article: infobox (site, sector,
    headquarters, founding year) plus the lead paragraph as raw summary text.
    Returns None if the page doesn't exist or has no usable infobox.
    """
    url = f"{WIKIPEDIA_BASE_URL}{title}"
    response = polite_get(url)
    if response is None:
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    heading = soup.find(id="firstHeading")
    nome = _clean_text(heading.get_text(strip=True)) if heading else title.replace("_", " ")

    infobox = soup.find("table", class_="infobox")
    site = setor = localizacao = None
    ano_fundacao = None
    if infobox is not None:
        for row in infobox.find_all("tr"):
            label_cell = row.find("th")
            value_cell = row.find("td")
            if label_cell is None or value_cell is None:
                continue
            label = _clean_text(label_cell.get_text(" ", strip=True)).lower()
            value = _clean_text(value_cell.get_text(" ", strip=True))
            if "website" in label or "página oficial" in label or "site oficial" in label:
                link = value_cell.find("a")
                href = link.get("href") if link else None
                # Prefer the href (the actual external URL) over the link's
                # visible text — pt.wikipedia usually shows the domain as the
                # label, but that's a display convention, not guaranteed.
                raw = href if href and href.startswith(("http://", "https://")) else (
                    link.get_text(strip=True) if link else value
                )
                site = "https://" + _clean_domain(raw).removeprefix("https://").removeprefix("http://")
            elif label.startswith("sede") or label.startswith("localidade"):
                localizacao = value
            elif "atividade" in label or "indústria" in label:
                setor = value
            elif setor is None and label.startswith("produtos"):
                setor = value
            elif label.startswith("fundação") or label.startswith("fundada"):
                match = _YEAR_RE.search(value)
                if match:
                    ano_fundacao = int(match.group())

    content = soup.find("div", class_="mw-parser-output")
    resumo = ""
    if content is not None:
        for paragraph in content.find_all("p"):
            text = _clean_text(paragraph.get_text(" ", strip=True))
            if len(text) > 60:
                resumo = text
                break

    if infobox is None and not resumo:
        return None

    return WikipediaCompany(
        nome=nome,
        site=site,
        setor=setor,
        localizacao=localizacao,
        ano_fundacao=ano_fundacao,
        resumo=resumo,
        url_fonte=url,
    )
