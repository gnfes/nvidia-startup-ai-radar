"use client";

import { useEffect, useState } from "react";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { fetchStartups, type Startup } from "@/lib/api";

export function StartupBrowser() {
  const [q, setQ] = useState("");
  const [setor, setSetor] = useState("");
  const [estagio, setEstagio] = useState("");
  const [startups, setStartups] = useState<Startup[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const timeout = setTimeout(() => {
      setLoading(true);
      setError(null);
      fetchStartups({ q, setor, estagio })
        .then(setStartups)
        .catch((err) => setError(err.message))
        .finally(() => setLoading(false));
    }, 300);
    return () => clearTimeout(timeout);
  }, [q, setor, estagio]);

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Input
          placeholder="Buscar por nome ou descrição..."
          value={q}
          onChange={(e) => setQ(e.target.value)}
        />
        <Input
          placeholder="Filtrar por setor..."
          value={setor}
          onChange={(e) => setSetor(e.target.value)}
        />
        <Input
          placeholder="Filtrar por estágio..."
          value={estagio}
          onChange={(e) => setEstagio(e.target.value)}
        />
      </div>

      {error && <p className="text-sm text-destructive">{error}</p>}

      <div className="overflow-x-auto rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="font-mono text-[10px] tracking-wider text-primary/80 uppercase">Nome</TableHead>
              <TableHead className="font-mono text-[10px] tracking-wider text-primary/80 uppercase">Setor</TableHead>
              <TableHead className="font-mono text-[10px] tracking-wider text-primary/80 uppercase">Estágio</TableHead>
              <TableHead className="font-mono text-[10px] tracking-wider text-primary/80 uppercase">Localização</TableHead>
              <TableHead className="font-mono text-[10px] tracking-wider text-primary/80 uppercase">Fundação</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {loading ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted-foreground">
                  Carregando...
                </TableCell>
              </TableRow>
            ) : startups.length === 0 ? (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted-foreground">
                  Nenhuma startup encontrada.
                </TableCell>
              </TableRow>
            ) : (
              startups.map((s) => (
                <TableRow key={s.id}>
                  <TableCell className="font-medium">
                    {s.site ? (
                      <a href={s.site} target="_blank" rel="noopener noreferrer" className="hover:underline">
                        {s.nome}
                      </a>
                    ) : (
                      s.nome
                    )}
                  </TableCell>
                  <TableCell className="max-w-xs truncate">
                    <Badge variant="secondary">{s.setor ?? "—"}</Badge>
                  </TableCell>
                  <TableCell className="max-w-xs truncate">{s.estagio ?? "—"}</TableCell>
                  <TableCell>{s.localizacao ?? "—"}</TableCell>
                  <TableCell>{s.ano_fundacao ?? "—"}</TableCell>
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}
