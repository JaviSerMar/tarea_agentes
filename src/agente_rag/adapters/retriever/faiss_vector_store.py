"""Adapter de almacenamiento vectorial en memoria basado en FAISS.

FAISS actúa como alternativa intercambiable a ChromaDB. Este adapter utiliza
similitud coseno mediante vectores normalizados y un índice de producto interno.
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from agente_rag.domain.entities import Chunk
from agente_rag.domain.ports import EmbedderPort


class FaissVectorStoreAdapter:
    """Almacén vectorial FAISS implementado sobre memoria y fichero local."""

    def __init__(
        self,
        *,
        path: Path,
        embedder: EmbedderPort,
    ) -> None:
        self._path = path
        self._embedder = embedder
        self._index: faiss.Index | None = None
        self._chunks: list[Chunk] = []

    def index(self, chunks: list[Chunk]) -> int:
        """Construye el índice FAISS a partir de los chunks recibidos."""
        if not chunks:
            self._index = None
            self._chunks = []
            return 0

        vectors = np.asarray(
            [self._embedder.embed(chunk.text) for chunk in chunks],
            dtype="float32",
        )

        faiss.normalize_L2(vectors)

        dimension = vectors.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(vectors)
        self._chunks = list(chunks)

        self._save_index()

        return len(self._chunks)

    def search(self, query_embedding: list[float], *, k: int = 5) -> list[Chunk]:
        """Busca los chunks más similares a un embedding de consulta."""
        self._ensure_loaded()

        if self._index is None or not self._chunks:
            return []

        query_vector = np.asarray([query_embedding], dtype="float32")
        faiss.normalize_L2(query_vector)

        result_count = min(k, len(self._chunks))
        scores, indexes = self._index.search(query_vector, result_count)

        results: list[Chunk] = []

        for score, index in zip(scores[0], indexes[0]):
            if index < 0:
                continue

            stored_chunk = self._chunks[int(index)]
            results.append(
                Chunk(
                    source=stored_chunk.source,
                    text=stored_chunk.text,
                    score=round(float(score), 4),
                    chunk_id=stored_chunk.chunk_id,
                )
            )

        return results

    def _save_index(self) -> None:
        """Persiste el índice y los metadatos mínimos necesarios."""
        if self._index is None:
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._path))

        metadata_path = self._path.with_suffix(".chunks.npz")
        np.savez(
            metadata_path,
            sources=np.asarray([chunk.source for chunk in self._chunks]),
            texts=np.asarray([chunk.text for chunk in self._chunks]),
            chunk_ids=np.asarray([chunk.chunk_id for chunk in self._chunks]),
        )

    def _ensure_loaded(self) -> None:
        """Carga el índice desde disco cuando aún no existe en memoria."""
        if self._index is not None:
            return

        metadata_path = self._path.with_suffix(".chunks.npz")

        if not self._path.exists() or not metadata_path.exists():
            return

        self._index = faiss.read_index(str(self._path))
        metadata = np.load(metadata_path)

        self._chunks = [
            Chunk(
                source=str(source),
                text=str(text),
                score=0.0,
                chunk_id=str(chunk_id),
            )
            for source, text, chunk_id in zip(
                metadata["sources"],
                metadata["texts"],
                metadata["chunk_ids"],
            )
        ]