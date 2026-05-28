"""Tests del adapter de almacenamiento vectorial ChromaDB.

No usan una base real ni Ollama: simulan el cliente ChromaDB y el embedder
para comprobar indexación, reemplazo de colecciones y búsqueda semántica.
"""

from unittest.mock import Mock, patch

from agente_rag.adapters.retriever.chroma_vector_store import (
    ChromaVectorStoreAdapter,
)
from agente_rag.domain.entities import Chunk


class FakeEmbedder:
    """Embedder controlado para probar el adapter sin modelos reales."""

    def embed(self, text: str) -> list[float]:
        vectors = {
            "Texto uno": [0.1, 0.2],
            "Texto dos": [0.3, 0.4],
        }
        return vectors[text]


def _chunks_for_indexing() -> list[Chunk]:
    return [
        Chunk(
            source="08_preguntas_basicas.txt",
            text="Texto uno",
            score=0.0,
            chunk_id="chunk_1",
        ),
        Chunk(
            source="04_filosofia_dni.txt",
            text="Texto dos",
            score=0.0,
            chunk_id="chunk_2",
        ),
    ]


def test_chroma_vector_store_indexes_chunks_with_injected_embedder(tmp_path):
    collection = Mock()
    collection.count.return_value = 2

    client = Mock()
    client.list_collections.return_value = []
    client.create_collection.return_value = collection

    adapter = ChromaVectorStoreAdapter(
        path=tmp_path / "chroma",
        collection_name="dni",
        embedder=FakeEmbedder(),
    )

    with patch(
        "agente_rag.adapters.retriever.chroma_vector_store.chromadb.PersistentClient",
        return_value=client,
    ):
        indexed_count = adapter.index(_chunks_for_indexing())

    assert indexed_count == 2

    client.create_collection.assert_called_once_with(
        "dni",
        metadata={"hnsw:space": "cosine"},
    )

    collection.add.assert_called_once_with(
        ids=["chunk_1", "chunk_2"],
        embeddings=[[0.1, 0.2], [0.3, 0.4]],
        documents=["Texto uno", "Texto dos"],
        metadatas=[
            {"source": "08_preguntas_basicas.txt"},
            {"source": "04_filosofia_dni.txt"},
        ],
    )


def test_chroma_vector_store_replaces_existing_collection(tmp_path):
    existing_collection = Mock()
    existing_collection.name = "dni"

    new_collection = Mock()
    new_collection.count.return_value = 2

    client = Mock()
    client.list_collections.return_value = [existing_collection]
    client.create_collection.return_value = new_collection

    adapter = ChromaVectorStoreAdapter(
        path=tmp_path / "chroma",
        collection_name="dni",
        embedder=FakeEmbedder(),
    )

    with patch(
        "agente_rag.adapters.retriever.chroma_vector_store.chromadb.PersistentClient",
        return_value=client,
    ):
        adapter.index(_chunks_for_indexing())

    client.delete_collection.assert_called_once_with("dni")


def test_chroma_vector_store_search_returns_domain_chunks(tmp_path):
    collection = Mock()
    collection.query.return_value = {
        "ids": [["chunk_1", "chunk_2"]],
        "documents": [["Texto uno", "Texto dos"]],
        "metadatas": [[
            {"source": "08_preguntas_basicas.txt"},
            {"source": "04_filosofia_dni.txt"},
        ]],
        "distances": [[0.1, 0.35]],
    }

    client = Mock()
    client.get_collection.return_value = collection

    adapter = ChromaVectorStoreAdapter(
        path=tmp_path / "chroma",
        collection_name="dni",
        embedder=FakeEmbedder(),
    )

    with patch(
        "agente_rag.adapters.retriever.chroma_vector_store.chromadb.PersistentClient",
        return_value=client,
    ):
        results = adapter.search([0.5, 0.6], k=2)

    collection.query.assert_called_once_with(
        query_embeddings=[[0.5, 0.6]],
        n_results=2,
    )

    assert len(results) == 2
    assert all(isinstance(chunk, Chunk) for chunk in results)
    assert results[0].source == "08_preguntas_basicas.txt"
    assert results[0].text == "Texto uno"
    assert results[0].score == 0.9
    assert results[0].chunk_id == "chunk_1"
    assert results[1].score == 0.65