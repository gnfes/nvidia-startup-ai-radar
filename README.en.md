🇧🇷 [Leia em Português](README.md) | 🇺🇸 English

# NVIDIA Startup AI Radar

Multi-agent platform that analyzes Brazilian startups from a pre-populated
database, diagnoses AI-native maturity, and recommends NVIDIA technologies
via a RAG pipeline with reranking. Built for the Inteli Academy x NVIDIA TAPI
selection process.

## Stack

- Backend: Python, FastAPI, LangGraph, Postgres (Supabase) + pgvector
- LLM / embeddings: NVIDIA NIM (build.nvidia.com)
- Reranking: Cohere Rerank
- Frontend: Next.js, TypeScript, Tailwind, shadcn/ui

## Project structure

```
backend/    FastAPI app, LangGraph agents, RAG pipeline (uv-managed)
frontend/   Next.js dashboard (npm-managed)
```

## Getting started

### Backend

```bash
cd backend
cp .env.example .env   # fill in DATABASE_URL, NVIDIA_API_KEY, COHERE_API_KEY
uv sync
uv run uvicorn radar_backend.main:app --reload
```

API available at `http://localhost:8000` (health check at `/health`).

### Frontend

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

App available at `http://localhost:3000`.
