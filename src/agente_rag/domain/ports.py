"""Puertos de salida del dominio del agente RAG de DNI.

Los Protocol definen las capacidades que necesita el dominio sin depender
de implementaciones concretas como Ollama, PoliGPT, ChromaDB o FAISS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .entities import Chunk, GenerationMetrics


@dataclass(frozen=True)
class GeneratedText:
    """Texto generado por un LLM junto con sus métricas."""

    text: str
    metrics: GenerationMetrics


class LLMPort(Protocol):
    """Contrato para generar respuestas con un modelo de lenguaje."""

    def generate(self, prompt: str, *, temperature: float = 0.2) -> GeneratedText:
        """Genera un texto fundamentado a partir de un prompt."""
        ...


class RetrieverPort(Protocol):
    """Contrato para recuperar evidencias relevantes del corpus."""

    def retrieve(self, query: str, *, k: int = 5) -> list[Chunk]:
        """Recupera los chunks más pertinentes para la consulta."""
        ...


class EmbedderPort(Protocol):
    """Contrato para vectorizar textos."""

    def embed(self, text: str) -> list[float]:
        """Transforma un texto en un vector numérico."""
        ...


class VectorStorePort(Protocol):
    """Contrato para indexar y consultar un almacén vectorial."""

    def index(self, chunks: list[Chunk]) -> int:
        """Almacena los chunks y devuelve cuántos se han indexado."""
        ...

    def search(self, query_embedding: list[float], *, k: int = 5) -> list[Chunk]:
        """Busca los chunks más cercanos a un embedding de consulta."""
        ...