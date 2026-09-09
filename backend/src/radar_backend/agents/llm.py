"""Shared NVIDIA NIM chat client for the LangGraph agent nodes (brief §5.1).

Mirrors rag/retrieve.py's embed_query: same OpenAI-compatible client pointed
at NVIDIA's base_url, but for chat completions instead of embeddings.
"""

from __future__ import annotations

import json
import time

from openai import APIError, APITimeoutError, OpenAI

from radar_backend.core.config import get_settings

REQUEST_TIMEOUT_SECONDS = 30.0
# This account's free-tier NIM endpoint has shown the same kind of
# intermittent flakiness (timeouts, occasional 500s) the Cohere trial key
# did in rag/retrieve.py's rerank() — retry through it rather than let one
# blip fail an entire node's pass over a batch of startups.
MAX_CHAT_ATTEMPTS = 3
CHAT_RETRY_DELAY_SECONDS = 3


def _client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.nvidia_api_key,
        base_url=settings.nvidia_base_url,
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=0,  # we do our own retry loop, with our own logging/backoff
    )


def _strip_markdown_fence(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.lower().startswith("json"):
            content = content[4:]
        content = content.strip()
    return content


def chat_json(system_prompt: str, user_prompt: str, *, temperature: float = 0.0, max_tokens: int = 500) -> dict:
    """Calls the configured NVIDIA NIM chat model and parses its response as
    JSON. The system prompt is responsible for instructing the model to
    return only JSON — NIM model support for a strict json_object response
    format varies by model, so this parses defensively (strips ```json
    fences) rather than relying on that.
    """
    settings = get_settings()
    client = _client()

    for attempt in range(1, MAX_CHAT_ATTEMPTS + 1):
        try:
            response = client.chat.completions.create(
                model=settings.nvidia_llm_model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("NVIDIA NIM chat response had empty content")
            break
        except (APITimeoutError, APIError, ValueError):
            if attempt == MAX_CHAT_ATTEMPTS:
                raise
            time.sleep(CHAT_RETRY_DELAY_SECONDS)

    cleaned = _strip_markdown_fence(content)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"NVIDIA NIM chat response was not valid JSON: {cleaned!r}") from exc
