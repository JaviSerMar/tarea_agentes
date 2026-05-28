"""Carga y segmentación del corpus oficial DNI.

Los documentos narrativos se segmentan con RecursiveCharacterTextSplitter.
Los documentos en formato Q:/A: se tratan de forma especial para no separar
una pregunta de su respuesta, ya que forman una unidad semántica natural.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int


def load_corpus(corpus_dir: Path) -> list[dict]:
    """Carga todos los documentos .txt disponibles en el corpus."""
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Corpus no encontrado en {corpus_dir}")

    docs = [
        {"name": path.name, "text": path.read_text(encoding="utf-8")}
        for path in sorted(corpus_dir.glob("*.txt"))
    ]

    if not docs:
        raise RuntimeError(f"No hay .txt en {corpus_dir}")

    return docs


def _has_qa_format(text: str) -> bool:
    """Indica si el documento contiene pares explícitos Q:/A:."""
    return bool(re.search(r"(?m)^Q:\s+.+", text) and re.search(r"(?m)^A:\s+.+", text))


def _split_qa_pairs(text: str) -> list[str]:
    """Extrae cada par Q:/A: completo como una unidad semántica."""
    pattern = re.compile(r"(?ms)^Q:\s*.*?(?=^Q:\s*|\Z)")
    return [match.group(0).strip() for match in pattern.finditer(text)]


def split_documents(
    docs: list[dict],
    *,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Chunk]:
    """Segmenta documentos conservando siempre la trazabilidad de la fuente."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Chunk] = []

    for doc in docs:
        if _has_qa_format(doc["text"]):
            pieces = _split_qa_pairs(doc["text"])
        else:
            pieces = splitter.split_text(doc["text"])

        for index, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{doc['name']}__chunk_{index:04d}",
                    text=piece,
                    source=doc["name"],
                    chunk_index=index,
                )
            )

    return chunks