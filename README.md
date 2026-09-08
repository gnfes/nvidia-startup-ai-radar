🇧🇷 Português | 🇺🇸 [Read in English](README.en.md)

# NVIDIA Startup AI Radar

Plataforma multi-agente que analisa startups brasileiras a partir de uma
base de dados pré-populada, diagnostica a maturidade AI-native de cada uma
e recomenda tecnologias NVIDIA por meio de um pipeline de RAG com
reranking. Desenvolvido para o processo seletivo do TAPI Inteli Academy x
NVIDIA.

## Stack

- Backend: Python, FastAPI, LangGraph, Postgres (Supabase) + pgvector
- LLM / embeddings: NVIDIA NIM (build.nvidia.com)
- Reranking: Cohere Rerank
- Frontend: Next.js, TypeScript, Tailwind, shadcn/ui

## Estrutura do projeto

```
backend/    App FastAPI, agentes LangGraph, pipeline de RAG (gerenciado com uv)
frontend/   Dashboard em Next.js (gerenciado com npm)
```

## Como rodar

### Banco de dados (Supabase)

1. Crie um projeto em [supabase.com](https://supabase.com) (região São Paulo/`sa-east-1` recomendada).
2. No **SQL Editor** do projeto, rode os arquivos abaixo **nesta ordem** (cole o conteúdo completo de cada um e clique em Run):
   - `backend/db/migrations/0001_startups_documentos.sql` — cria as tabelas `startups`/`documentos` e configura RLS.
   - `backend/db/seed/manual_seed.sql` — 8 startups curadas manualmente.
   - `backend/db/seed/scraped_batch.sql` — 23 startups adicionais, coletadas pelo pipeline em `backend/src/radar_backend/ingestion/`.
3. Pegue a connection string em **Connect → Direct connection/Session pooler → URI**. **Use a Session pooler** (porta 5432, host `aws-0-<região>.pooler.supabase.com`), não a "Direct connection": o hostname da Direct connection só tem registro DNS IPv6, e falha em redes sem IPv6 (comum em conexões domésticas no Brasil).

### Backend

```bash
cd backend
cp .env.example .env   # preencha DATABASE_URL (connection string do passo acima), NVIDIA_API_KEY, COHERE_API_KEY
uv sync
uv run uvicorn radar_backend.main:app --reload
```

API disponível em `http://localhost:8000` (health check em `/health`).

### Frontend

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

App disponível em `http://localhost:3000`.
