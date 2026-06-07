"""Construye el índice vectorial usando los adapters configurados.

Permite indexar el corpus oficial DNI tanto con ChromaDB como con FAISS,
seleccionando la implementación mediante las variables del archivo .env.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agente_rag.chunker import load_corpus, split_documents
from agente_rag.composition import build_vector_store_adapter
from agente_rag.config import SETTINGS
from agente_rag.domain.entities import Chunk


def main() -> int:
    """Carga el corpus, construye chunks de dominio e indexa evidencias."""
    print(f"[hex_index] corpus_dir            = {SETTINGS.corpus_dir}")
    print(f"[hex_index] embedder_provider     = {SETTINGS.embedder_provider}")
    print(f"[hex_index] vector_store_provider = {SETTINGS.vector_store_provider}")
    print(f"[hex_index] collection_name       = {SETTINGS.collection_name}")
    print(f"[hex_index] faiss_path            = {SETTINGS.faiss_path}")

    documents = load_corpus(SETTINGS.corpus_dir)
    print(f"[hex_index] {len(documents)} documentos cargados.")

    source_chunks = split_documents(documents)
    chunks = [
        Chunk(
            source=chunk.source,
            text=chunk.text,
            score=0.0,
            chunk_id=chunk.id,
        )
        for chunk in source_chunks
    ]
    print(f"[hex_index] {len(chunks)} chunks generados.")

    vector_store = build_vector_store_adapter()

    started_at = time.time()
    indexed_count = vector_store.index(chunks)
    elapsed_seconds = time.time() - started_at

    print(
        f"[hex_index] {indexed_count} chunks indexados "
        f"en {elapsed_seconds:.1f}s."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())