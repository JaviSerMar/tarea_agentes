"""Tests del adapter de recuperación léxica BM25."""

from agente_rag.adapters.retriever.bm25_retriever import BM25RetrieverAdapter
from agente_rag.domain.entities import Chunk


def _sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            source="08_preguntas_basicas.txt",
            text=(
                "Q: ¿Qué es DNI?\n"
                "A: DNI es una asociación de jóvenes voluntarios en Valencia."
            ),
            score=0.0,
            chunk_id="dni_definition",
        ),
        Chunk(
            source="01_faq_dni.txt",
            text=(
                "Q: ¿A qué hora son los desayunos solidarios?\n"
                "A: Los desayunos son a las 8 de la mañana."
            ),
            score=0.0,
            chunk_id="breakfast_time",
        ),
        Chunk(
            source="06_coles_refuerzo.txt",
            text="El refuerzo escolar acompaña a niños en sus estudios.",
            score=0.0,
            chunk_id="school_support",
        ),
    ]


def test_bm25_retriever_prioritizes_exact_faq_terms():
    retriever = BM25RetrieverAdapter(_sample_chunks())

    results = retriever.retrieve("¿Qué es DNI?", k=2)

    assert results
    assert results[0].chunk_id == "dni_definition"
    assert results[0].source == "08_preguntas_basicas.txt"
    assert results[0].score > 0


def test_bm25_retriever_ignores_accents_and_case():
    retriever = BM25RetrieverAdapter(_sample_chunks())

    results = retriever.retrieve("DESAYUNOS SOLIDARIOS manana", k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "breakfast_time"


def test_bm25_retriever_returns_empty_list_for_empty_corpus():
    retriever = BM25RetrieverAdapter([])

    assert retriever.retrieve("¿Qué es DNI?") == []