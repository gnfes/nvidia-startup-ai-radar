-- Fase 2: modelo de dados base (brief §5.2).
-- Rodar manualmente no SQL editor do Supabase, na ordem numerica dos arquivos.

create extension if not exists pgcrypto;

create table startups (
    id uuid primary key default gen_random_uuid(),
    nome text not null,
    site text,
    setor text,
    estagio text,
    localizacao text,
    descricao_curta text,
    ano_fundacao smallint,
    tamanho_time text,
    created_at timestamptz not null default now()
);

create table documentos (
    id uuid primary key default gen_random_uuid(),
    startup_id uuid not null references startups(id) on delete cascade,
    tipo text not null check (tipo in (
        'site_institucional', 'blog', 'noticia', 'vaga', 'perfil_founder', 'release'
    )),
    titulo text not null,
    conteudo_texto text not null,
    url_fonte text not null,
    data_publicacao date,
    created_at timestamptz not null default now()
);

create index idx_documentos_startup_id on documentos(startup_id);

-- RLS: our backend connects with the postgres role (bypasses RLS regardless),
-- so this only matters if Supabase's auto-generated REST/GraphQL API is ever
-- queried directly. This data is public startup info, so allow public reads
-- and leave writes unpoliced (i.e. impossible) through that API.
alter table startups enable row level security;
alter table documentos enable row level security;

create policy "Public read access" on startups for select using (true);
create policy "Public read access" on documentos for select using (true);
