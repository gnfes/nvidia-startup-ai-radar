"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { runAnalysis, type AnalysisResult, type StartupAnalysis } from "@/lib/api";

const CLASSIFICATION_LABEL: Record<string, string> = {
  ai_native: "AI-native",
  ai_enabled: "AI-enabled",
  non_ai: "Não-IA",
};

const PRIORITY_VARIANT: Record<string, "default" | "secondary" | "outline"> = {
  alta: "default",
  media: "secondary",
  baixa: "outline",
};

function StartupResultCard({ analysis }: { analysis: StartupAnalysis }) {
  const { startup_row: startup, recommendation: rec } = analysis;
  const sinaisPorProduto = new Map(
    analysis.regras_correspondentes.map((m) => [m.produto, m.sinais_correspondentes])
  );

  return (
    <Card className="border-t-2 border-t-primary">
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>{startup.nome}</CardTitle>
          {analysis.classification && (
            <Badge variant="outline" className="border-primary text-primary bg-primary/10">
              {CLASSIFICATION_LABEL[analysis.classification] ?? analysis.classification}
            </Badge>
          )}
          {rec && (
            <Badge variant={PRIORITY_VARIANT[rec.nivel_prioridade] ?? "outline"}>
              prioridade {rec.nivel_prioridade}
            </Badge>
          )}
          {!analysis.evidence_validated && (
            <Badge variant="destructive">evidência limitada — retomar com mais fontes</Badge>
          )}
        </div>
        {analysis.classification_reasoning && (
          <CardDescription>{analysis.classification_reasoning}</CardDescription>
        )}
      </CardHeader>
      <CardContent className="flex flex-col gap-3 text-sm">
        {rec ? (
          <>
            <div>
              <div className="mb-1 font-mono text-[10px] tracking-wider text-primary/80 uppercase">Tecnologias NVIDIA recomendadas</div>
              {rec.tecnologias_recomendadas.join(", ") || "nenhuma"}
            </div>
            <div>
              <div className="mb-1 font-mono text-[10px] tracking-wider text-primary/80 uppercase">Justificativa técnica</div>
              {rec.justificativa_tecnica}
            </div>
            <div>
              <div className="mb-1 font-mono text-[10px] tracking-wider text-primary/80 uppercase">Justificativa de negócio</div>
              {rec.justificativa_negocio}
            </div>
            <div>
              <div className="mb-1 font-mono text-[10px] tracking-wider text-primary/80 uppercase">Próxima ação sugerida</div>
              {rec.proxima_acao}
            </div>
            <div>
              <div className="mb-1 font-mono text-[10px] tracking-wider text-primary/80 uppercase">Complexidade de implementação</div>
              {rec.complexidade_implementacao}
            </div>
            {rec.evidencias.length > 0 && (
              <div>
                <div className="mb-1 font-mono text-[10px] tracking-wider text-primary/80 uppercase">Evidências (base NVIDIA)</div>
                <ul className="mt-1 list-inside list-disc">
                  {rec.evidencias.map((ev) => (
                    <li key={ev.id}>
                      <a
                        href={ev.url_fonte}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="hover:underline"
                      >
                        {ev.fonte_titulo}
                        {ev.secao ? ` — ${ev.secao}` : ""}
                      </a>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {(rec.concordancia_regras.length > 0 || rec.alertas_regras.length > 0) && (
              <div>
                <div className="mb-1 font-mono text-[10px] tracking-wider text-primary/80 uppercase">
                  Verificação cruzada (regras determinísticas)
                </div>
                <div className="flex flex-col gap-1.5">
                  {rec.concordancia_regras.map((produto) => (
                    <div key={produto} className="flex flex-wrap items-center gap-1.5">
                      <Badge variant="outline" className="border-primary/60 text-primary">
                        ✓ confirmado: {produto}
                      </Badge>
                      <span className="text-xs text-muted-foreground">
                        sinais: {(sinaisPorProduto.get(produto) ?? []).join(", ")}
                      </span>
                    </div>
                  ))}
                  {rec.alertas_regras.map((produto) => (
                    <div key={produto} className="flex flex-wrap items-center gap-1.5">
                      <Badge variant="destructive">⚠ sinal não confirmado: {produto}</Badge>
                      <span className="text-xs text-muted-foreground">
                        sinais: {(sinaisPorProduto.get(produto) ?? []).join(", ")} — pode ser uma
                        lacuna real ou um sinal genérico; vale conferir
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : (
          <p className="text-muted-foreground">Sem recomendação gerada para esta startup.</p>
        )}
      </CardContent>
    </Card>
  );
}

function downloadMarkdown(briefing: string) {
  const blob = new Blob([briefing], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `briefing-${new Date().toISOString().slice(0, 10)}.md`;
  link.click();
  URL.revokeObjectURL(url);
}

export function AnalysisPanel() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      setResult(await runAnalysis(query));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Erro desconhecido");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="flex flex-col gap-4">
      <form onSubmit={handleSubmit} className="flex flex-col gap-3 sm:flex-row">
        <Input
          placeholder="Ex.: startups de atendimento por voz que poderiam usar IA generativa"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          disabled={loading}
        />
        <Button type="submit" disabled={loading || !query.trim()} className="clip-corner rounded-none">
          {loading ? "Analisando..." : "Rodar análise"}
        </Button>
      </form>

      {loading && (
        <p className="text-sm text-muted-foreground">
          Rodando o pipeline de agentes (Query Planner → Retriever → Extractor →
          Classifier → Validator → NVIDIA RAG → Recommendation → Briefing) — isso
          pode levar alguns minutos para várias startups.
        </p>
      )}

      {error && <p className="text-sm text-destructive">{error}</p>}

      {result && (
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              {result.startups.length} startup(s) analisada(s).
            </p>
            <Button variant="outline" size="sm" onClick={() => downloadMarkdown(result.final_briefing)}>
              Exportar briefing (.md)
            </Button>
          </div>
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {result.startups.map((analysis) => (
              <StartupResultCard key={analysis.startup_id} analysis={analysis} />
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
