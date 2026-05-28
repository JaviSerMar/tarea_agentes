"""Adapter de recuperación léxica basado en BM25.

Este adapter opera sobre una colección de chunks del dominio y resulta
especialmente útil cuando la consulta coincide con preguntas FAQ del corpus.
"""

from __future__ import annotations

import re
import unicodedata

from rank_bm25 import BM25Okapi

from agente_rag.domain.entities import Chunk


class BM25RetrieverAdapter:
    """Implementación de RetrieverPort basada en coincidencia textual."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        self._tokenized_documents = [
            self._tokenize(chunk.text) for chunk in self._chunks
        ]
        self._bm25 = (
            BM25Okapi(self._tokenized_documents)
            if self._tokenized_documents
            else None
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Normaliza y tokeniza un texto ignorando mayúsculas y tildes."""
        normalized = unicodedata.normalize("NFD", text.casefold())
        without_accents = "".join(
            character
            for character in normalized
            if unicodedata.category(character) != "Mn"
        )
        return re.findall(r"\b\w+\b", without_accents)

    def retrieve(self, query: str, *, k: int = 5) -> list[Chunk]:
        """Devuelve los chunks con mayor coincidencia léxica."""
        if self._bm25 is None:
            return []

        scores = self._bm25.get_scores(self._tokenize(query))
        ranked_indexes = sorted(
            range(len(self._chunks)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results: list[Chunk] = []

        for index in ranked_indexes:
            score = float(scores[index])

            if score <= 0:
                continue

            chunk = self._chunks[index]
            results.append(
                Chunk(
                    source=chunk.source,
                    text=chunk.text,
                    score=round(score, 4),
                    chunk_id=chunk.chunk_id,
                )
            )

            if len(results) == k:
                break

        return results