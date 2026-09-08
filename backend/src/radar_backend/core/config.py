from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Supabase / Postgres (with pgvector extension enabled)
    database_url: str = ""

    # NVIDIA NIM (OpenAI-compatible API) — https://build.nvidia.com
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_llm_model: str = "meta/llama-3.1-70b-instruct"
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
