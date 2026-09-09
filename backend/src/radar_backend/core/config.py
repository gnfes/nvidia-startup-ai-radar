from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Supabase / Postgres (with pgvector extension enabled)
    database_url: str = ""

    # NVIDIA NIM (OpenAI-compatible API) — https://build.nvidia.com
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    # meta/llama-3.1-70b-instruct (named in Phase 1/2 scoping) reached EOL by
    # Phase 5 — same pattern as the embedding model below. Re-verified live
    # against this account's actual catalog: several NVIDIA-branded
    # candidates either 404'd or were unstable (one Nemotron variant always
    # emits an unsuppressable chain-of-thought preamble that ate the whole
    # token budget before reaching JSON; a couple of others intermittently
    # timed out or 500'd on this free tier). This one responds in ~1-2s and
    # follows JSON-only system prompts cleanly — see agents/llm.py, which
    # wraps it with the same kind of retry the Cohere rerank call needed for
    # its own free-tier flakiness.
    nvidia_llm_model: str = "meta/llama-3.2-11b-vision-instruct"
    # nv-embedqa-e5-v5 reached end-of-life 2026-08-25; nemotron-3-embed-1b is
    # its current replacement on this account (verified against the live API
    # during Phase 3 — several other embedding models listed by build.nvidia.com
    # 404'd for this account). 2048-dim output.
    nvidia_embedding_model: str = "nvidia/nemotron-3-embed-1b"

    # Cohere Rerank — https://docs.cohere.com/docs/rerank-overview
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-v3.5"


@lru_cache
def get_settings() -> Settings:
    return Settings()
