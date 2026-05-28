"""Tests del adapter de recuperación híbrida hexagonal."""

from agente_rag.adapters.retriever.hybrid_retriever import HybridRetrieverAdapter
from agente_rag.domain.entities import Chunk


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self.last_query: str | None = None
        self.last_k: int | None = None

    def retrieve(self, query: str, *, k: int = 5) -> list[Chunk]:
        self.last_query = query
        self.last_k = k
        return self._chunks


def test_hybrid_retriever_prioritizes_exact_faq_match():
    exact_chunk = Chunk(
        source="08_preguntas_basicas.txt",
        text="Q: ¿Qué es DNI?\nA: DNI es una asociación juvenil.",
        score=0.7,
        chunk_id="exact",
    )
    irrelevant_chunk = Chunk(
        source="04_filosofia_dni.txt",
        text="¿POR QUÉ DNI? La filosofía de la asociación.",
        score=0.9,
        chunk_id="related",
    )

    semantic = FakeRetriever([irrelevant_chunk, exact_chunk])
    lexical = FakeRetriever([exact_chunk])
    retriever = HybridRetrieverAdapter(
        semantic_retriever=semantic,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve("¿Qué es DNI?", k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "exact"
    assert results[0].score == 1.0
    assert semantic.last_k == 20
    assert lexical.last_k == 20


def test_hybrid_retriever_preserves_multiple_exact_answers():
    first = Chunk(
        source="01_faq_dni.txt",
        text=(
            "Q: ¿A qué hora son los desayunos solidarios?\n"
            "A: Los desayunos son a las 8 de la mañana."
        ),
        score=0.8,
        chunk_id="first",
    )
    second = Chunk(
        source="11_horarios_ubicaciones.txt",
        text=(
            "Q: ¿A qué hora son los desayunos solidarios?\n"
            "A: Los desayunos suelen realizarse entre las 9:00 y las 12:00h."
        ),
        score=0.7,
        chunk_id="second",
    )

    retriever = HybridRetrieverAdapter(
        semantic_retriever=FakeRetriever([first]),
        lexical_retriever=FakeRetriever([first, second]),
    )

    results = retriever.retrieve("¿A qué hora son los desayunos solidarios?")

    assert [chunk.source for chunk in results] == [
        "01_faq_dni.txt",
        "11_horarios_ubicaciones.txt",
    ]
    assert all(chunk.score == 1.0 for chunk in results)


def test_hybrid_retriever_fuses_non_exact_rankings():
    semantic_first = Chunk(
        source="04_filosofia_dni.txt",
        text="Filosofía general de DNI.",
        score=0.9,
        chunk_id="shared",
    )
    lexical_first = Chunk(
        source="04_filosofia_dni.txt",
        text="Filosofía general de DNI.",
        score=2.1,
        chunk_id="shared",
    )
    lexical_second = Chunk(
        source="10_proyectos.txt",
        text="Proyectos de DNI.",
        score=1.4,
        chunk_id="other",
    )

    retriever = HybridRetrieverAdapter(
        semantic_retriever=FakeRetriever([semantic_first]),
        lexical_retriever=FakeRetriever([lexical_first, lexical_second]),
    )

    results = retriever.retrieve("filosofía y proyectos", k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "shared"
    assert results[0].score > results[1].score