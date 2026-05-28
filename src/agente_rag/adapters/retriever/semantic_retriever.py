"""Adapter de retrieval semántico desacoplado del almacenamiento vectorial.

Este adapter utiliza un EmbedderPort para vectorizar la consulta y un
VectorStorePort para recuperar evidencias, sin conocer si el almacenamiento
concreto es ChromaDB o FAISS.
"""

from __future__ import annotations

from agente_rag.domain.entities import Chunk
from agente_rag.domain.ports import EmbedderPort, VectorStorePort


class SemanticRetrieverAdapter:
    """Implementación de RetrieverPort mediante búsqueda vectorial semántica."""

    def __init__(
        self,
        *,
        embedder: EmbedderPort,
        vector_store: VectorStorePort,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(self, query: str, *, k: int = 5) -> list[Chunk]:
        """Vectoriza una consulta y recupera sus chunks más próximos."""
        query_embedding = self._embedder.embed(query)
        return self._vector_store.search(query_embedding, k=k)