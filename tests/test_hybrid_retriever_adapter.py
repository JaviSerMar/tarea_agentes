"""Tests del adapter de recuperación híbrida.

No utilizan ChromaDB, Ollama ni red: se inyecta una función de recuperación
falsa para comprobar la traducción hacia las entidades del dominio.
"""

from agente_rag.adapters.retriever.hybrid_retriever import HybridRetrieverAdapter
from agente_rag.domain.entities import Chunk
from agente_rag.retriever import RetrievedChunk


def test_hybrid_retriever_adapter_converts_retrieved_chunks_to_domain_chunks():
    received_arguments: dict = {}

    def fake_retrieve(query: str, *, k: int = 5) -> list[RetrievedChunk]:
        received_arguments["query"] = query
        received_arguments["k"] = k

        return [
            RetrievedChunk(
                source="08_preguntas_basicas.txt",
                text="Q: ¿Qué es DNI?\nA: DNI es una asociación juvenil.",
                score=1.0,
                chunk_id="dni_definition",
            ),
            RetrievedChunk(
                source="04_filosofia_dni.txt",
                text="PARA. MIRA. AYUDA.",
                score=0.7,
                chunk_id="dni_philosophy",
            ),
        ]

    adapter = HybridRetrieverAdapter(retrieve_function=fake_retrieve)

    chunks = adapter.retrieve("¿Qué es DNI?", k=2)

    assert received_arguments == {"query": "¿Qué es DNI?", "k": 2}
    assert len(chunks) == 2
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert chunks[0].source == "08_preguntas_basicas.txt"
    assert chunks[0].score == 1.0
    assert chunks[0].chunk_id == "dni_definition"
    assert chunks[1].source == "04_filosofia_dni.txt"


def test_hybrid_retriever_adapter_returns_empty_list_without_results():
    def fake_retrieve(query: str, *, k: int = 5) -> list[RetrievedChunk]:
        return []

    adapter = HybridRetrieverAdapter(retrieve_function=fake_retrieve)

    assert adapter.retrieve("pregunta sin resultados") == []