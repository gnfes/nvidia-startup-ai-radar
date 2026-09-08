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

### Backend

```bash
cd backend
cp .env.example .env   # preencha DATABASE_URL, NVIDIA_API_KEY, COHERE_API_KEY
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
