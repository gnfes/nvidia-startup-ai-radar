"""Fetches and extracts content for the 24 §8 sources.

Text sources (articles/blog/docs) get full heading-aware extraction — unlike
ingestion/collect.py's truncated startup-page text, these pages ARE the
knowledge base, so nothing is cut short. Video sources get title+description
only via yt-dlp (no API key, no download) — full transcription was scoped out
given the project timeline; see CLAUDE.md for that decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import yt_dlp
from bs4 import BeautifulSoup
from bs4.element import Tag

from radar_backend.ingestion.http import polite_get
from radar_backend.rag.kb_sources import KbSource

MIN_SECTION_CHARS = 40
_HEADING_TAGS = ["h1", "h2", "h3", "h4"]
_CONTENT_TAGS = ["p", "li", "pre", "blockquote"]


@dataclass
class KbSection:
    secao: str | None
    texto: str


def _content_root(soup: BeautifulSoup) -> Tag:
    # Deliberately not preferring <article>/<main>: NVIDIA's blog template
    # uses <article> for "related posts" teaser tiles that appear before the
    # real post body in document order, so find("article") grabbed the wrong
    # element and returned zero content. Body + chrome-stripping + the
    # MIN_SECTION_CHARS filter in extract_sections is more robust across
    # unknown page templates than trusting semantic tags to be used correctly.
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside"]):
        tag.decompose()
    return soup.body or soup


def extract_sections(html: str) -> list[KbSection]:
    """Walks the page in document order, grouping text under the nearest
    preceding heading so a chunk can later cite a specific section instead of
    just the page URL. Doesn't guard against a content tag nested inside
    another matched content tag (e.g. <p> inside <li>) double-counting text —
    acceptable occasional duplication for the time budget here.
    """
    soup = BeautifulSoup(html, "html.parser")
    root = _content_root(soup)

    sections: list[KbSection] = []
    current_heading: str | None = None
    buffer: list[str] = []

    def flush() -> None:
        text = " ".join(buffer).strip()
        if len(text) >= MIN_SECTION_CHARS:
            sections.append(KbSection(secao=current_heading, texto=text))
        buffer.clear()

    for element in root.find_all(_HEADING_TAGS + _CONTENT_TAGS):
        if element.name in _HEADING_TAGS:
            flush()
            current_heading = element.get_text(" ", strip=True)[:200] or current_heading
        else:
            text = element.get_text(" ", strip=True)
            if text:
                buffer.append(text)
    flush()
    return sections


def fetch_video_metadata(url: str) -> KbSection | None:
    opts = {"quiet": True, "skip_download": True, "extract_flat": False}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        # yt-dlp raises many distinct exception types for network/geo/
        # availability issues — an unreachable video is simply skipped,
        # same as a blocked page in ingestion/http.py.
        return None
    text = f"{info.get('title') or ''}\n\n{info.get('description') or ''}".strip()
    return KbSection(secao=None, texto=text) if len(text) >= MIN_SECTION_CHARS else None


def fetch_playlist_metadata(url: str) -> KbSection | None:
    opts = {"quiet": True, "skip_download": True, "extract_flat": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        return None
    entry_titles = [e.get("title", "") for e in (info.get("entries") or []) if e.get("title")]
    parts = [
        info.get("title") or "",
        info.get("description") or "",
        ("Vídeos: " + "; ".join(entry_titles)) if entry_titles else "",
    ]
    text = "\n\n".join(part for part in parts if part).strip()
    return KbSection(secao=None, texto=text) if len(text) >= MIN_SECTION_CHARS else None


def fetch_source(source: KbSource) -> list[KbSection]:
    if source.tipo == "video":
        section = fetch_playlist_metadata(source.url) if "playlist" in source.url else fetch_video_metadata(source.url)
        return [section] if section else []
    response = polite_get(source.url)
    if response is None:
        return []
    return extract_sections(response.text)
