"""Orquestador del flujo RAG: percibir, recuperar, responder y validar.

Además de generar respuestas con el LLM, este módulo incorpora una salvaguarda
para preguntas que tienen varias respuestas exactas en el corpus. En esos
casos se muestran todas las versiones recuperadas para no ocultar posibles
contradicciones oficiales.
"""

from __future__ import annotations

from .generator import generate
from .prompts import build_prompt
from .retriever import RetrievedChunk, retrieve


def answer(question: str, *, k: int = 5, conversation_id: str | None = None) -> dict:
    """Responde a una pregunta siguiendo el contrato obligatorio."""
    retrieved = retrieve(question, k=k)
    prompt = build_prompt(question, retrieved)
    generation = generate(prompt)

    response_text = _validated_response(generation.text.strip(), retrieved)

    return {
        "respuesta": response_text,
        "fuentes": _unique_preserving_order(chunk.source for chunk in retrieved),
        "chunks": [
            {"source": chunk.source, "text": chunk.text, "score": chunk.score}
            for chunk in retrieved
        ],
        "metricas": {
            "prompt_tokens": generation.prompt_tokens,
            "output_tokens": generation.output_tokens,
            "tokens_per_sec": generation.tokens_per_sec,
            "latencia_s": generation.latency_s,
            "modelo": generation.model,
        },
        "trazas": None,
        "conversation_id": conversation_id,
    }


def _validated_response(generated_text: str, retrieved: list[RetrievedChunk]) -> str:
    """Evita ocultar respuestas exactas múltiples presentes en el corpus."""
    exact_qa_chunks = [
        chunk
        for chunk in retrieved
        if chunk.score == 1.0
        and chunk.text.lstrip().startswith("Q:")
        and "\nA:" in chunk.text
    ]

    if len(exact_qa_chunks) <= 1:
        return generated_text

    versions = [
        f"- Según {chunk.source}: {_extract_answer(chunk.text)}"
        for chunk in exact_qa_chunks
    ]

    return (
        "Las fuentes contienen varias versiones para esta pregunta, "
        "por lo que deben mostrarse todas:\n"
        + "\n".join(versions)
    )


def _extract_answer(chunk_text: str) -> str:
    """Extrae el contenido de la respuesta de un chunk Q:/A:."""
    return chunk_text.split("\nA:", maxsplit=1)[1].strip()


def _unique_preserving_order(items) -> list[str]:
    """Elimina duplicados conservando el orden de recuperación."""
    seen: set[str] = set()
    output: list[str] = []

    for item in items:
        if item not in seen:
            seen.add(item)
            output.append(item)

    return output