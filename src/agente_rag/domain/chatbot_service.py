"""Servicio de dominio del agente RAG de DNI.

Orquesta la recuperación de evidencias, la construcción del prompt y la
generación de la respuesta sin depender de infraestructura concreta.
"""

from __future__ import annotations

from .entities import Answer, Chunk, Question
from .ports import LLMPort, RetrieverPort


REJECTION_PHRASE = "No tengo esa información en mis fuentes"


class ChatbotService:
    """Caso de uso principal: responder preguntas sobre DNI."""

    def __init__(self, llm: LLMPort, retriever: RetrieverPort) -> None:
        self._llm = llm
        self._retriever = retriever

    def answer(self, question: Question, *, k: int = 5) -> Answer:
        """Recupera contexto y produce una respuesta fundamentada."""
        chunks = self._retriever.retrieve(question.text, k=k)
        prompt = self._build_prompt(question.text, chunks)
        generation = self._llm.generate(prompt, temperature=0.2)

        response_text = self._validated_response(generation.text.strip(), chunks)

        return Answer(
            text=response_text,
            sources=self._unique_preserving_order(chunk.source for chunk in chunks),
            chunks=chunks,
            metrics=generation.metrics,
            traces=None,
            conversation_id=question.conversation_id,
        )

    def _build_prompt(self, question: str, chunks: list[Chunk]) -> str:
        """Construye el prompt anti-alucinación a partir de las evidencias."""
        context = "\n\n".join(
            f"[{chunk.source}]\n{chunk.text}" for chunk in chunks
        )

        return f"""Eres un asistente de la asociación juvenil de voluntariado
Damos Nuestra Ilusión (DNI) Valencia.

Tu tarea es responder a la pregunta utilizando únicamente el CONTEXTO.

ORDEN DE DECISIÓN OBLIGATORIO:
1. Busca si algún fragmento del contexto contiene información relevante para responder.
2. Si uno o varios fragmentos contienen la respuesta, responde usando esa información.
3. Si dos fragmentos relevantes ofrecen datos diferentes o incompatibles, NO rechaces la
   pregunta: indica claramente que las fuentes muestran información contradictoria y expón
   cada versión con su archivo correspondiente.
4. Solo si ningún fragmento contiene información relevante, responde literalmente:
   "{REJECTION_PHRASE}".

REGLAS:
- No inventes fechas, horarios, ubicaciones, contactos, cifras ni actividades.
- No elijas una única versión cuando las fuentes sean contradictorias.
- Cita entre paréntesis el nombre del archivo o archivos utilizados.
- Redacta una respuesta clara y breve.

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:"""

    def _validated_response(self, generated_text: str, chunks: list[Chunk]) -> str:
        """Garantiza que varias respuestas exactas no queden ocultas."""
        exact_qa_chunks = [
            chunk
            for chunk in chunks
            if chunk.score == 1.0
            and chunk.text.lstrip().startswith("Q:")
            and "\nA:" in chunk.text
        ]

        if len(exact_qa_chunks) <= 1:
            return generated_text

        versions = [
            f"- Según {chunk.source}: {self._extract_answer(chunk.text)}"
            for chunk in exact_qa_chunks
        ]

        return (
            "Las fuentes contienen varias versiones para esta pregunta, "
            "por lo que deben mostrarse todas:\n"
            + "\n".join(versions)
        )

    @staticmethod
    def _extract_answer(chunk_text: str) -> str:
        """Extrae la respuesta de un chunk en formato Q:/A:."""
        return chunk_text.split("\nA:", maxsplit=1)[1].strip()

    @staticmethod
    def _unique_preserving_order(items) -> list[str]:
        """Elimina fuentes duplicadas conservando su orden original."""
        seen: set[str] = set()
        output: list[str] = []

        for item in items:
            if item not in seen:
                seen.add(item)
                output.append(item)

        return output