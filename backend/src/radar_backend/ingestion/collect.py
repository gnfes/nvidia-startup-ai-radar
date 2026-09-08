"""Collects the raw documentos for a startup once we have a confirmed site.

Kept deliberately dumb per the schema's own design: conteudo_texto is meant to
be unstructured raw text for the (future, Phase 5) Extractor Agent to work on,
so we just grab clean-ish page text, not parsed fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from radar_backend.ingestion.http import polite_get

MAX_TEXT_CHARS = 1800

_SECONDARY_PATHS: list[tuple[str, str]] = [
    ("/blog", "blog"),
    ("/carreiras", "vaga"),
    ("/vagas", "vaga"),
    ("/trabalhe-conosco", "vaga"),
    ("/imprensa", "noticia"),
    ("/sobre", "site_institucional"),
    ("/sobre-nos", "site_institucional"),
    ("/about", "site_institucional"),
]


@dataclass
class RawDocument:
    tipo: str
    titulo: str
    conteudo_texto: str
    url_fonte: str


def _page_title(soup: BeautifulSoup, fallback: str) -> str:
    if soup.title and soup.title.get_text(strip=True):
        return soup.title.get_text(strip=True)[:200]
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"][:200]
    return fallback


def _extract_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    text = soup.get_text(" ", strip=True)
    return text[:MAX_TEXT_CHARS]


def fetch_site_institucional(site: str, nome: str) -> RawDocument | None:
    response = polite_get(site)
    if response is None:
        return None
    soup = BeautifulSoup(response.text, "html.parser")
    text = _extract_text(soup)
    if len(text) < 80:
        return None
    return RawDocument(
        tipo="site_institucional",
        titulo=_page_title(soup, nome),
        conteudo_texto=text,
        url_fonte=site,
    )


def fetch_one_secondary_document(
    site: str, institucional_text: str | None = None, institucional_url: str | None = None
) -> RawDocument | None:
    """`institucional_text`/`institucional_url` should come from the already-
    fetched site_institucional doc, if any — some sites (SPAs, catch-all
    routing) return a 200 with the homepage for any unknown path instead of a
    404, which without this check silently stores a mislabeled duplicate
    document. We check both the resolved URL (catches a redirect straight
    back to the homepage even if the page has slightly different dynamic
    content each load) and the extracted text (catches a same-URL homepage
    serve without a redirect).
    """
    institucional_url_norm = institucional_url.rstrip("/") if institucional_url else None
    for path, tipo in _SECONDARY_PATHS:
        url = urljoin(site, path)
        response = polite_get(url)
        if response is None:
            continue
        if institucional_url_norm is not None and response.url.rstrip("/") == institucional_url_norm:
            continue
        soup = BeautifulSoup(response.text, "html.parser")
        text = _extract_text(soup)
        if len(text) < 80:
            continue
        if institucional_text is not None and text == institucional_text:
            continue
        return RawDocument(
            tipo=tipo,
            titulo=_page_title(soup, path.strip("/")),
            conteudo_texto=text,
            url_fonte=response.url,
        )
    return None
