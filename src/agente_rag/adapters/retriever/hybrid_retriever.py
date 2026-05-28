"""Adapter de recuperación híbrida para el dominio hexagonal.

Combina un canal semántico y un canal léxico mediante Reciprocal Rank Fusion
(RRF). Además, prioriza respuestas FAQ exactas para evitar ruido y conserva
varias coincidencias exactas para que el dominio detecte contradicciones.
"""

from __future__ import annotations

import re
import unicodedata

from agente_rag.domain.entities import Chunk
from agente_rag.domain.ports import RetrieverPort


class HybridRetrieverAdapter:
    """Combina dos implementaciones de RetrieverPort."""

    def __init__(
        self,
        *,
        semantic_retriever: RetrieverPort,
        lexical_retriever: RetrieverPort,
    ) -> None:
        self._semantic_retriever = semantic_retriever
        self._lexical_retriever = lexical_retriever

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normaliza texto para detectar coincidencias FAQ exactas."""
        normalized = unicodedata.normalize("NFD", text.casefold())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        return re.sub(r"\s+", " ", without_accents).strip()

    def retrieve(self, query: str, *, k: int = 5) -> list[Chunk]:
        """Recupera chunks fusionando resultados semánticos y léxicos."""
        candidate_k = max(k * 4, 20)

        semantic_chunks = self._semantic_retriever.retrieve(query, k=candidate_k)
        lexical_chunks = self._lexical_retriever.retrieve(query, k=candidate_k)

        chunks_by_id = {
            chunk.chunk_id: chunk
            for chunk in semantic_chunks + lexical_chunks
        }

        normalized_query = self._normalize_text(query)
        expected_qa_start = f"q: {normalized_query}"

        exact_matches = [
            chunk
            for chunk in chunks_by_id.values()
            if self._normalize_text(chunk.text).startswith(expected_qa_start)
        ]

        if exact_matches:
            return [
                Chunk(
                    source=chunk.source,
                    text=chunk.text,
                    score=1.0,
                    chunk_id=chunk.chunk_id,
                )
                for chunk in exact_matches[:k]
            ]

        fused_scores: dict[str, float] = {}

        for rank, chunk in enumerate(semantic_chunks, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (60 + rank)
            )

        for rank, chunk in enumerate(lexical_chunks, start=1):
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (60 + rank)
            )

        ranked_ids = sorted(
            fused_scores,
            key=lambda chunk_id: fused_scores[chunk_id],
            reverse=True,
        )[:k]

        return [
            Chunk(
                source=chunks_by_id[chunk_id].source,
                text=chunks_by_id[chunk_id].text,
                score=round(fused_scores[chunk_id], 4),
                chunk_id=chunk_id,
            )
            for chunk_id in ranked_ids
        ]