"""Adapter de almacenamiento vectorial basado en ChromaDB.

Este adapter encapsula la persistencia y búsqueda semántica en ChromaDB.
Recibe un EmbedderPort para no depender de un modelo de embeddings concreto.
"""

from __future__ import annotations

from pathlib import Path

import chromadb

from agente_rag.domain.entities import Chunk
from agente_rag.domain.ports import EmbedderPort


class ChromaVectorStoreAdapter:
    """Almacén vectorial persistente implementado con ChromaDB."""

    def __init__(
        self,
        *,
        path: Path,
        collection_name: str,
        embedder: EmbedderPort,
    ) -> None:
        self._path = path
        self._collection_name = collection_name
        self._embedder = embedder

    def _client(self):
        """Obtiene un cliente persistente de ChromaDB."""
        self._path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(self._path))

    def index(self, chunks: list[Chunk]) -> int:
        """Reemplaza la colección actual e indexa los chunks recibidos."""
        client = self._client()

        existing_names = [
            collection.name for collection in client.list_collections()
        ]

        if self._collection_name in existing_names:
            client.delete_collection(self._collection_name)

        collection = client.create_collection(
            self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[self._embedder.embed(chunk.text) for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {"source": chunk.source}
                for chunk in chunks
            ],
        )

        return collection.count()

    def search(self, query_embedding: list[float], *, k: int = 5) -> list[Chunk]:
        """Recupera los chunks más similares a un embedding de consulta."""
        client = self._client()
        collection = client.get_collection(self._collection_name)

        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
        )

        chunks: list[Chunk] = []

        for index, chunk_id in enumerate(result["ids"][0]):
            distance = float(result["distances"][0][index])

            chunks.append(
                Chunk(
                    source=result["metadatas"][0][index]["source"],
                    text=result["documents"][0][index],
                    score=round(1.0 - distance, 4),
                    chunk_id=chunk_id,
                )
            )

        return chunks