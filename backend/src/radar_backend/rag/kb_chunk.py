"""Chunks extracted sections with LangChain's RecursiveCharacterTextSplitter —
the standard tool for "chunking semântico" per the brief, and flagged by the
league president as a credibility differentiator for the RAG work (see
CLAUDE.md's "Guidance from the league president").
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from radar_backend.rag.kb_collect import KbSection

# Sized for retrieval granularity, not a model context limit (nemotron-3-embed-1b
# handles up to 32k tokens) — smaller chunks make citations more precise and
# keep unrelated topics within a section from being embedded as one vector.
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@dataclass
class KbChunk:
    secao: str | None
    chunk_index: int
    texto: str


def chunk_sections(sections: list[KbSection]) -> list[KbChunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks: list[KbChunk] = []
    index = 0
    for section in sections:
        for piece in splitter.split_text(section.texto):
            chunks.append(KbChunk(secao=section.secao, chunk_index=index, texto=piece))
            index += 1
    return chunks
