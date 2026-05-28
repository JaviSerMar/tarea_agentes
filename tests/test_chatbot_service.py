"""Tests unitarios del dominio hexagonal.

Estos tests no usan Ollama, ChromaDB ni red. Inyectan adapters falsos para
comprobar la lógica central del agente de manera rápida y determinista.
"""

from agente_rag.domain.chatbot_service import ChatbotService
from agente_rag.domain.entities import Chunk, GenerationMetrics, Question
from agente_rag.domain.ports import GeneratedText


class FakeLLM:
    """Adapter falso que devuelve una respuesta predeterminada."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str, *, temperature: float = 0.2) -> GeneratedText:
        self.last_prompt = prompt
        return GeneratedText(
            text=self.response,
            metrics=GenerationMetrics(
                prompt_tokens=10,
                output_tokens=5,
                tokens_per_sec=20.0,
                latency_s=0.01,
                model="fake-llm",
            ),
        )


class FakeRetriever:
    """Adapter falso que devuelve chunks controlados."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.last_query: str | None = None
        self.last_k: int | None = None

    def retrieve(self, query: str, *, k: int = 5) -> list[Chunk]:
        self.last_query = query
        self.last_k = k
        return self.chunks


def test_domain_answers_using_injected_ports():
    chunks = [
        Chunk(
            source="08_preguntas_basicas.txt",
            text=(
                "Q: ¿Qué es DNI?\n"
                "A: DNI es una asociación de jóvenes voluntarios en Valencia."
            ),
            score=1.0,
            chunk_id="dni_definition",
        )
    ]
    llm = FakeLLM("DNI es una asociación de jóvenes voluntarios en Valencia.")
    retriever = FakeRetriever(chunks)
    service = ChatbotService(llm=llm, retriever=retriever)

    answer = service.answer(Question(text="¿Qué es DNI?", conversation_id="conv-1"))

    assert "asociación" in answer.text
    assert answer.sources == ["08_preguntas_basicas.txt"]
    assert answer.chunks == chunks
    assert answer.metrics is not None
    assert answer.metrics.model == "fake-llm"
    assert answer.conversation_id == "conv-1"
    assert retriever.last_query == "¿Qué es DNI?"
    assert "[08_preguntas_basicas.txt]" in llm.last_prompt


def test_domain_preserves_contradictory_exact_answers():
    chunks = [
        Chunk(
            source="01_faq_dni.txt",
            text=(
                "Q: ¿A qué hora son los desayunos solidarios?\n"
                "A: Los desayunos son a las 8 de la mañana."
            ),
            score=1.0,
            chunk_id="faq_horario",
        ),
        Chunk(
            source="11_horarios_ubicaciones.txt",
            text=(
                "Q: ¿A qué hora son los desayunos solidarios?\n"
                "A: Los desayunos suelen realizarse entre las 9:00 y las 12:00h."
            ),
            score=1.0,
            chunk_id="horarios_horario",
        ),
    ]
    llm = FakeLLM("Solo se realizan entre las 9:00 y las 12:00h.")
    service = ChatbotService(llm=llm, retriever=FakeRetriever(chunks))

    answer = service.answer(Question(text="¿A qué hora son los desayunos solidarios?"))

    assert "varias versiones" in answer.text
    assert "8 de la mañana" in answer.text
    assert "9:00" in answer.text
    assert "01_faq_dni.txt" in answer.text
    assert "11_horarios_ubicaciones.txt" in answer.text


def test_domain_keeps_rejection_from_llm_when_context_does_not_answer():
    chunks = [
        Chunk(
            source="09_como_participar.txt",
            text="Información sobre cómo participar en DNI.",
            score=0.3,
            chunk_id="irrelevant_chunk",
        )
    ]
    llm = FakeLLM("No tengo esa información en mis fuentes.")
    service = ChatbotService(llm=llm, retriever=FakeRetriever(chunks))

    answer = service.answer(Question(text="¿Cuánto cuesta un alquiler en Valencia?"))

    assert answer.text == "No tengo esa información en mis fuentes."