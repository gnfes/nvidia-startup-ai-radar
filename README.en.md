🇧🇷 [Leia em Português](README.md) | 🇺🇸 English

# NVIDIA Startup AI Radar

Multi-agent platform that analyzes Brazilian startups from a pre-populated
database, diagnoses AI-native maturity, and recommends NVIDIA technologies
via a RAG pipeline with reranking. Built for the Inteli Academy x NVIDIA TAPI
selection process.

📐 Architecture documentation: [ARCHITECTURE.en.md](ARCHITECTURE.en.md) — how
the agent pipeline and the hybrid RAG work, and why.

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

### Database (Supabase)

1. Create a project at [supabase.com](https://supabase.com) (São Paulo/`sa-east-1` region recommended).
2. In the project's **SQL Editor**, run the files below **in this order** (paste each file's full contents and click Run):
   - `backend/db/migrations/0001_startups_documentos.sql` — creates the `startups`/`documentos` tables and sets up RLS.
   - `backend/db/seed/manual_seed.sql` — 8 hand-curated startups.
   - `backend/db/seed/scraped_batch.sql` — 23 more startups, collected via the pipeline in `backend/src/radar_backend/ingestion/`.
   - `backend/db/seed/new_startups_seed.sql` — 19 more startups (50 total), each individually verified for founding-in-Brazil and backed by at least 3 real, citable documentos.
   - `backend/db/migrations/0002_nvidia_kb.sql` — creates the `nvidia_kb_chunks` table (pgvector + tsvector) for the NVIDIA knowledge base.
   - `backend/db/seed/nvidia_kb_seed.sql` — 520 chunks (24 sources from brief §8), with embeddings. **~10MB**: the web SQL Editor may choke on that size — run it with the script below instead of pasting it in (works for any of the migration/seed files above too, if you'd rather):
     ```bash
     cd backend
     uv run python -m radar_backend.db.apply_sql db/seed/nvidia_kb_seed.sql
     ```
3. Get the connection string from **Connect → Direct connection/Session pooler → URI**. **Use the Session pooler** (port 5432, host `aws-0-<region>.pooler.supabase.com`), not "Direct connection": that hostname only has an IPv6 DNS record and fails to resolve on networks without IPv6 (common on many home networks).

### Backend

```bash
cd backend
cp .env.example .env   # fill in DATABASE_URL (connection string from above), NVIDIA_API_KEY, COHERE_API_KEY
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
