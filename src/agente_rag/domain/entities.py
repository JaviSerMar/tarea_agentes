"""Entidades puras del dominio del agente RAG de DNI.

Este módulo contiene únicamente estructuras de datos del negocio. No depende
de Ollama, ChromaDB, PoliGPT ni de ningún framework de infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Question:
    """Pregunta formulada por una persona usuaria."""

    text: str
    conversation_id: str | None = None


@dataclass(frozen=True)
class Chunk:
    """Fragmento del corpus recuperado como evidencia para una respuesta."""

    source: str
    text: str
    score: float
    chunk_id: str


@dataclass(frozen=True)
class GenerationMetrics:
    """Métricas devueltas por el modelo de lenguaje."""

    prompt_tokens: int = 0
    output_tokens: int = 0
    tokens_per_sec: float = 0.0
    latency_s: float = 0.0
    model: str = ""


@dataclass(frozen=True)
class Answer:
    """Respuesta final generada por el agente con sus evidencias."""

    text: str
    sources: list[str] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    metrics: GenerationMetrics | None = None
    traces: list[dict] | None = None
    conversation_id: str | None = None