-- Fase 3: base de conhecimento NVIDIA para RAG (brief §8, §5.3).
-- Rodar manualmente no SQL editor do Supabase, apos a 0001.

create extension if not exists vector;

create table nvidia_kb_chunks (
    id uuid primary key default gen_random_uuid(),
    categoria text not null check (categoria in ('material_apoio', 'documentacao_oficial')),
    tipo text not null check (tipo in ('artigo', 'blog', 'documentacao', 'video')),
    fonte_titulo text not null,
    url_fonte text not null,
    -- Heading (h1-h4) the chunk was extracted from, when the source page has
    -- one — lets citations point at a specific section instead of just the
    -- source URL. Null for sources with no heading structure (e.g. video
    -- title/description metadata).
    secao text,
    chunk_index int not null,
    conteudo_chunk text not null,
    -- nvidia/nemotron-3-embed-1b, 2048-dim output (its predecessor,
    -- nv-embedqa-e5-v5, reached end-of-life 2026-08-25 mid-Phase-3).
    embedding vector(2048),
    -- The embedding model and the source material are both English;
    -- 'english' gives proper stemming for the lexical half of hybrid search,
    -- unlike the 'portuguese' config used for documentos elsewhere in this
    -- project.
    tsv tsvector generated always as (to_tsvector('english', conteudo_chunk)) stored,
    created_at timestamptz not null default now()
);

create index idx_nvidia_kb_chunks_tsv on nvidia_kb_chunks using gin (tsv);

-- No vector index (HNSW/ivfflat): pgvector caps indexed columns at 2000
-- dimensions and nemotron-3-embed-1b outputs 2048 — discovered when this
-- migration was first run. At this scale (hundreds of chunks) a sequential
-- scan for `order by embedding <=> ...` is fast enough that an ANN index
-- isn't worth the complexity of dimensionality reduction just to fit under
-- the limit.

-- Same rationale as 0001: the backend connects as the postgres role and
-- bypasses RLS regardless; this only matters for Supabase's auto-generated
-- REST/GraphQL API, and this data (public NVIDIA docs/marketing content) is
-- fine to expose read-only there too.
alter table nvidia_kb_chunks enable row level security;

create policy "Public read access" on nvidia_kb_chunks for select using (true);
