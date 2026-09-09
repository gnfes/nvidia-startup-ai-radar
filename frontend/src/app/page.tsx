import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { StartupBrowser } from "@/components/startup-browser";
import { AnalysisPanel } from "@/components/analysis-panel";

export default function Home() {
  return (
    <div className="mx-auto flex w-full max-w-5xl flex-1 flex-col gap-6 px-6 py-10">
      <header className="flex flex-col gap-2 border-b border-border pb-5">
        <div className="flex items-center gap-3">
          <svg
            width="28"
            height="28"
            viewBox="0 0 28 28"
            fill="none"
            className="text-primary shrink-0"
            aria-hidden="true"
          >
            <rect x="8" y="8" width="12" height="12" rx="2" stroke="currentColor" strokeWidth="2" />
            <rect x="12" y="12" width="4" height="4" fill="currentColor" />
            <line x1="8" y1="11" x2="4" y2="11" stroke="currentColor" strokeWidth="2" />
            <line x1="8" y1="17" x2="4" y2="17" stroke="currentColor" strokeWidth="2" />
            <line x1="20" y1="11" x2="24" y2="11" stroke="currentColor" strokeWidth="2" />
            <line x1="20" y1="17" x2="24" y2="17" stroke="currentColor" strokeWidth="2" />
            <line x1="11" y1="8" x2="11" y2="4" stroke="currentColor" strokeWidth="2" />
            <line x1="17" y1="8" x2="17" y2="4" stroke="currentColor" strokeWidth="2" />
            <line x1="11" y1="20" x2="11" y2="24" stroke="currentColor" strokeWidth="2" />
            <line x1="17" y1="20" x2="17" y2="24" stroke="currentColor" strokeWidth="2" />
          </svg>
          <h1 className="font-brand text-xl font-bold uppercase tracking-wide">
            <span className="text-primary">NVIDIA</span> Startup AI Radar
          </h1>
        </div>
        <p className="text-sm text-muted-foreground">
          Diagnóstico de maturidade em IA de startups brasileiras e recomendação de
          tecnologias NVIDIA via RAG.
        </p>
      </header>

      <Tabs defaultValue="startups">
        <TabsList>
          <TabsTrigger value="startups">Startups</TabsTrigger>
          <TabsTrigger value="analysis">Análise</TabsTrigger>
        </TabsList>
        <TabsContent value="startups" className="mt-4">
          <StartupBrowser />
        </TabsContent>
        <TabsContent value="analysis" className="mt-4">
          <AnalysisPanel />
        </TabsContent>
      </Tabs>
    </div>
  );
}
