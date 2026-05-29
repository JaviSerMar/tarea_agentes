"""Construye el índice vectorial desde base_conocimiento/.

Ejecución desde la raíz del repo:
    python scripts/build_index.py

Tiempo estimado: 30-90 s para el corpus GTI (depende del modelo de
embeddings y de la latencia con Ollama).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agente_rag.chunker import load_corpus, split_documents
from agente_rag.config import SETTINGS
from agente_rag.retriever import build_index


def main() -> int:
    print(f"[build_index] corpus_dir   = {SETTINGS.corpus_dir}")
    print(f"[build_index] chroma_path  = {SETTINGS.chroma_path}")
    print(f"[build_index] collection   = {SETTINGS.collection_name}")
    print(f"[build_index] ollama_url   = {SETTINGS.ollama_url}")
    print(f"[build_index] embed_model  = {SETTINGS.embed_model}")

    docs = load_corpus(SETTINGS.corpus_dir)
    print(f"[build_index] {len(docs)} documentos cargados.")

    chunks = split_documents(docs)
    print(f"[build_index] {len(chunks)} chunks generados.")

    t0 = time.time()
    n = build_index(chunks)
    print(f"[build_index] {n} chunks indexados en {time.time() - t0:.1f}s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
