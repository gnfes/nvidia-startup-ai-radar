const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Startup = {
  id: string;
  nome: string;
  site: string | null;
  setor: string | null;
  estagio: string | null;
  localizacao: string | null;
  descricao_curta: string | null;
  ano_fundacao: number | null;
  tamanho_time: string | null;
};

export type StartupFilters = {
  q?: string;
  setor?: string;
  estagio?: string;
  porte?: string;
};

export async function fetchStartups(filters: StartupFilters): Promise<Startup[]> {
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(filters)) {
    if (value) params.set(key, value);
  }
  const res = await fetch(`${API_URL}/startups?${params.toString()}`);
  if (!res.ok) throw new Error(`Falha ao buscar startups (${res.status})`);
  return res.json();
}

export type NvidiaChunkMatch = {
  id: string;
  fonte_titulo: string;
  url_fonte: string;
  secao: string | null;
  conteudo_chunk: string;
  relevance_score: number | null;
};

export type Recommendation = {
  tecnologias_recomendadas: string[];
  justificativa_tecnica: string;
  justificativa_negocio: string;
  nivel_prioridade: "alta" | "media" | "baixa";
  complexidade_implementacao: "baixa" | "media" | "alta";
  proxima_acao: string;
  evidencias: NvidiaChunkMatch[];
};

export type StartupAnalysis = {
  startup_id: string;
  startup_row: Startup;
  classification: "ai_native" | "ai_enabled" | "non_ai" | null;
  classification_reasoning: string | null;
  evidence_validated: boolean;
  evidence_notes: string | null;
  recommendation: Recommendation | null;
};

export type AnalysisResult = {
  startups: StartupAnalysis[];
  final_briefing: string;
};

export async function runAnalysis(query: string): Promise<AnalysisResult> {
  const res = await fetch(`${API_URL}/analysis`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });
  if (!res.ok) throw new Error(`Falha ao rodar a análise (${res.status})`);
  return res.json();
}
