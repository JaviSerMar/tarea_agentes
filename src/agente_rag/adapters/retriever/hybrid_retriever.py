"""Adapter de recuperación híbrida para el dominio hexagonal.

Reutiliza temporalmente la recuperación híbrida ya validada del proyecto
y traduce sus resultados a las entidades puras del dominio.
"""

from __future__ import annotations

from collections.abc import Callable

from agente_rag.domain.entities import Chunk
from agente_rag.retriever import RetrievedChunk, retrieve


class HybridRetrieverAdapter:
    """Implementación de RetrieverPort usando retrieval semántico + BM25."""

    def __init__(
        self,
        retrieve_function: Callable[..., list[RetrievedChunk]] = retrieve,
    ) -> None:
        self._retrieve_function = retrieve_function

    def retrieve(self, query: str, *, k: int = 5) -> list[Chunk]:
        """Recupera evidencias y las convierte a entidades del dominio."""
        retrieved_chunks = self._retrieve_function(query, k=k)

        return [
            Chunk(
                source=chunk.source,
                text=chunk.text,
                score=chunk.score,
                chunk_id=chunk.chunk_id,
            )
            for chunk in retrieved_chunks
        ]