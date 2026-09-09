"""Polite HTTP fetching for the ingestion pipeline: identifies itself, checks
robots.txt before every request, and rate-limits per host. No CAPTCHA/bot-
detection bypass of any kind — a host that blocks us is simply skipped.
"""

from __future__ import annotations

import time
from urllib import robotparser
from urllib.parse import urlparse

import requests

USER_AGENT = "nvidia-startup-ai-radar-bot/0.1 (+educational project; contact: gnfsemp@gmail.com)"
REQUEST_TIMEOUT = 10
MIN_DELAY_SECONDS = 1.5

_robots_cache: dict[str, robotparser.RobotFileParser] = {}
_last_request_at: dict[str, float] = {}


def _host(url: str) -> str:
    return urlparse(url).netloc


def _robots_for(url: str) -> robotparser.RobotFileParser:
    # Deliberately not using RobotFileParser.read(): it fetches robots.txt via
    # urllib with Python's default User-Agent, which some hosts (e.g.
    # Wikimedia) reject with a 403 — robotparser then treats that as
    # "disallow everything". Fetching with our own identified UA and handing
    # the text to parse() avoids that false negative.
    host = _host(url)
    if host not in _robots_cache:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{host}/robots.txt"
        rp = robotparser.RobotFileParser()
        rp.set_url(robots_url)
        try:
            response = requests.get(robots_url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT)
            if response.status_code >= 400:
                rp.allow_all = True
            else:
                rp.parse(response.text.splitlines())
        except requests.RequestException:
            rp.allow_all = True
        _robots_cache[host] = rp
    return _robots_cache[host]


def _throttle(host: str) -> None:
    last = _last_request_at.get(host)
    if last is not None:
        elapsed = time.monotonic() - last
        if elapsed < MIN_DELAY_SECONDS:
            time.sleep(MIN_DELAY_SECONDS - elapsed)
    _last_request_at[host] = time.monotonic()


def polite_get(url: str) -> requests.Response | None:
    """GETs url if robots.txt allows it, throttled per host. Returns None on
    any failure (blocked by robots, non-2xx, network error, etc.) rather than
    raising — callers just treat a missing page as "skip this source"."""
    try:
        if not _robots_for(url).can_fetch(USER_AGENT, url):
            return None
        _throttle(_host(url))
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.5"},
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code >= 400:
            return None
        # requests defaults to Latin-1 for text/* responses that omit a
        # charset in Content-Type, which mangles accented pt-BR text; most
        # sites here are actually UTF-8, so sniff it when unspecified.
        if "charset" not in (response.headers.get("Content-Type") or "").lower():
            response.encoding = response.apparent_encoding
        return response
    except requests.RequestException:
        return None


def url_exists(url: str) -> bool:
    try:
        if not _robots_for(url).can_fetch(USER_AGENT, url):
            return False
        _throttle(_host(url))
        response = requests.head(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )
        if response.status_code in (405, 403):
            return polite_get(url) is not None
        return response.status_code < 400
    except requests.RequestException:
        return False
