"""Domain resolution for startups discovered without an explicit site link
(the Distrito listicle names companies but never links to them).

We deliberately do NOT use a search engine here: the obvious no-API-key
option (DuckDuckGo's HTML endpoint) turned out to require solving a CAPTCHA,
and bypassing bot-detection is off the table. Instead we guess the domain
from the company name and verify it actually resolves — lower hit rate, but
we only ever keep a startup whose domain we've confirmed responds, never a
guess we didn't check.
"""

from __future__ import annotations

import re
import unicodedata

from radar_backend.ingestion.http import polite_get, url_exists

_NAME_TLD_RE = re.compile(r"\.(ai|law|io|co|app)$", re.IGNORECASE)


def _slugify(nome: str) -> str:
    normalized = unicodedata.normalize("NFKD", nome).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", normalized.lower())


def _mentions_name(url: str, nome: str) -> bool:
    """Guards against a guessed domain that resolves but belongs to an
    unrelated business (a real risk of pure name-to-domain guessing) — the
    homepage has to actually mention the company name somewhere.
    """
    response = polite_get(url)
    if response is None:
        return False
    haystack = _slugify(response.text[:20000])
    needle = _slugify(nome)
    return bool(needle) and needle in haystack


def resolve_site(nome: str) -> str | None:
    candidates: list[str] = []

    compact = nome.lower().replace(" ", "")
    if _NAME_TLD_RE.search(compact):
        candidates += [f"https://www.{compact}", f"https://{compact}"]

    slug = _slugify(nome)
    if slug:
        for domain in (f"{slug}.com.br", f"{slug}.ai", f"{slug}.com", f"{slug}.io"):
            candidates += [f"https://www.{domain}", f"https://{domain}"]

    seen: set[str] = set()
    for url in candidates:
        if url in seen:
            continue
        seen.add(url)
        if url_exists(url) and _mentions_name(url, nome):
            return url
    return None
