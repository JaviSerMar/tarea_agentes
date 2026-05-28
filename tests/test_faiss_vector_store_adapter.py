"""Tests del adapter de almacenamiento vectorial FAISS.

Utilizan vectores controlados y ficheros temporales para verificar indexación,
búsqueda por similitud y recarga del índice sin afectar al agente principal.
"""

from agente_rag.adapters.retriever.faiss_vector_store import (
    FaissVectorStoreAdapter,
)
from agente_rag.domain.entities import Chunk


class FakeEmbedder:
    """Embedder determinista para pruebas del índice FAISS."""

    def embed(self, text: str) -> list[float]:
        vectors = {
            "Información general de DNI": [1.0, 0.0],
            "Información sobre desayunos": [0.0, 1.0],
        }
        return vectors[text]


def _chunks_for_indexing() -> list[Chunk]:
    return [
        Chunk(
            source="08_preguntas_basicas.txt",
            text="Información general de DNI",
            score=0.0,
            chunk_id="chunk_dni",
        ),
        Chunk(
            source="01_faq_dni.txt",
            text="Información sobre desayunos",
            score=0.0,
            chunk_id="chunk_desayunos",
        ),
    ]


def test_faiss_vector_store_indexes_and_searches_chunks(tmp_path):
    index_path = tmp_path / "dni.index"
    adapter = FaissVectorStoreAdapter(
        path=index_path,
        embedder=FakeEmbedder(),
    )

    count = adapter.index(_chunks_for_indexing())
    results = adapter.search([1.0, 0.0], k=2)

    assert count == 2
    assert index_path.exists()
    assert index_path.with_suffix(".chunks.npz").exists()

    assert len(results) == 2
    assert results[0].source == "08_preguntas_basicas.txt"
    assert results[0].chunk_id == "chunk_dni"
    assert results[0].score == 1.0
    assert results[1].source == "01_faq_dni.txt"


def test_faiss_vector_store_loads_existing_index_from_disk(tmp_path):
    index_path = tmp_path / "dni.index"

    first_adapter = FaissVectorStoreAdapter(
        path=index_path,
        embedder=FakeEmbedder(),
    )
    first_adapter.index(_chunks_for_indexing())

    reloaded_adapter = FaissVectorStoreAdapter(
        path=index_path,
        embedder=FakeEmbedder(),
    )
    results = reloaded_adapter.search([0.0, 1.0], k=1)

    assert len(results) == 1
    assert results[0].source == "01_faq_dni.txt"
    assert results[0].chunk_id == "chunk_desayunos"
    assert results[0].score == 1.0


def test_faiss_vector_store_handles_empty_index(tmp_path):
    adapter = FaissVectorStoreAdapter(
        path=tmp_path / "empty.index",
        embedder=FakeEmbedder(),
    )

    assert adapter.index([]) == 0
    assert adapter.search([1.0, 0.0], k=5) == []