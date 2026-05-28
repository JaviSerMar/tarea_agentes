"""Tests del retriever semántico desacoplado.

No utilizan embeddings reales ni almacenes vectoriales reales: inyectan
implementaciones falsas de los ports del dominio.
"""

from agente_rag.adapters.retriever.semantic_retriever import (
    SemanticRetrieverAdapter,
)
from agente_rag.domain.entities import Chunk


class FakeEmbedder:
    def __init__(self) -> None:
        self.last_text: str | None = None

    def embed(self, text: str) -> list[float]:
        self.last_text = text
        return [0.1, 0.2, 0.3]


class FakeVectorStore:
    def __init__(self) -> None:
        self.last_embedding: list[float] | None = None
        self.last_k: int | None = None

    def search(self, query_embedding: list[float], *, k: int = 5) -> list[Chunk]:
        self.last_embedding = query_embedding
        self.last_k = k

        return [
            Chunk(
                source="08_preguntas_basicas.txt",
                text="Q: ¿Qué es DNI?\nA: DNI es una asociación juvenil.",
                score=0.95,
                chunk_id="dni_definition",
            )
        ]


def test_semantic_retriever_uses_embedder_and_vector_store():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = SemanticRetrieverAdapter(
        embedder=embedder,
        vector_store=vector_store,
    )

    chunks = retriever.retrieve("¿Qué es DNI?", k=3)

    assert embedder.last_text == "¿Qué es DNI?"
    assert vector_store.last_embedding == [0.1, 0.2, 0.3]
    assert vector_store.last_k == 3
    assert len(chunks) == 1
    assert chunks[0].source == "08_preguntas_basicas.txt"
    assert chunks[0].chunk_id == "dni_definition"