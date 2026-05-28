"""Indexación y recuperación híbrida sobre el corpus DNI.

La colección ChromaDB proporciona recuperación semántica mediante embeddings.
BM25 añade recuperación léxica, especialmente útil para preguntas del corpus
redactadas literalmente en formato Q:/A:.

Ambos rankings se combinan mediante Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from .chunker import Chunk
from .config import SETTINGS
from .embedder import embed


@dataclass
class RetrievedChunk:
    source: str
    text: str
    score: float
    chunk_id: str


def _client(path: Path) -> chromadb.api.ClientAPI:
    """Devuelve el cliente persistente de ChromaDB."""
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def build_index(chunks: list[Chunk]) -> int:
    """Construye o reemplaza la colección vectorial con los chunks dados."""
    client = _client(SETTINGS.chroma_path)

    if SETTINGS.collection_name in [collection.name for collection in client.list_collections()]:
        client.delete_collection(SETTINGS.collection_name)

    collection = client.create_collection(
        SETTINGS.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[chunk.id for chunk in chunks],
        embeddings=[embed(chunk.text) for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {"source": chunk.source, "chunk_index": chunk.chunk_index}
            for chunk in chunks
        ],
    )

    return collection.count()


def _open_collection():
    """Abre la colección DNI ya indexada."""
    client = _client(SETTINGS.chroma_path)
    return client.get_collection(SETTINGS.collection_name)


def _tokenize(text: str) -> list[str]:
    """Normaliza y tokeniza texto para BM25 ignorando mayúsculas y tildes."""
    normalized = unicodedata.normalize("NFD", text.casefold())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return re.findall(r"\b\w+\b", without_accents)

def _normalize_text(text: str) -> str:
    """Normaliza texto para detectar coincidencias exactas de preguntas."""
    normalized = unicodedata.normalize("NFD", text.casefold())
    without_accents = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    return re.sub(r"\s+", " ", without_accents).strip()


def _semantic_retrieve(question: str, *, k: int) -> list[RetrievedChunk]:
    """Obtiene candidatos mediante similitud semántica de ChromaDB."""
    collection = _open_collection()
    result = collection.query(query_embeddings=[embed(question)], n_results=k)

    chunks: list[RetrievedChunk] = []

    for index, chunk_id in enumerate(result["ids"][0]):
        distance = float(result["distances"][0][index])
        chunks.append(
            RetrievedChunk(
                source=result["metadatas"][0][index]["source"],
                text=result["documents"][0][index],
                score=round(1.0 - distance, 4),
                chunk_id=chunk_id,
            )
        )

    return chunks


def _lexical_retrieve(question: str, *, k: int) -> list[RetrievedChunk]:
    """Obtiene candidatos mediante coincidencia léxica BM25."""
    collection = _open_collection()
    data = collection.get(include=["documents", "metadatas"])

    documents = data["documents"] or []
    metadatas = data["metadatas"] or []
    ids = data["ids"]

    tokenized_documents = [_tokenize(document) for document in documents]
    bm25 = BM25Okapi(tokenized_documents)
    scores = bm25.get_scores(_tokenize(question))

    ranked_indexes = sorted(
        range(len(documents)),
        key=lambda index: scores[index],
        reverse=True,
    )

    chunks: list[RetrievedChunk] = []

    for index in ranked_indexes:
        score = float(scores[index])

        if score <= 0:
            continue

        chunks.append(
            RetrievedChunk(
                source=metadatas[index]["source"],
                text=documents[index],
                score=round(score, 4),
                chunk_id=ids[index],
            )
        )

        if len(chunks) == k:
            break

    return chunks


def retrieve(question: str, *, k: int = 5) -> list[RetrievedChunk]:
    """Recupera chunks combinando búsqueda semántica, léxica y FAQ exactas.

    Si existe una respuesta Q:/A: exactamente asociada a la pregunta,
    se devuelve directamente ese contexto para evitar ruido innecesario.
    Las consultas que puedan implicar contradicciones se ampliarán en una
    mejora posterior del retriever.
    """
    candidate_k = max(k * 4, 20)

    semantic_chunks = _semantic_retrieve(question, k=candidate_k)
    lexical_chunks = _lexical_retrieve(question, k=candidate_k)

    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in semantic_chunks + lexical_chunks
    }

    normalized_question = _normalize_text(question)
    expected_qa_start = f"q: {normalized_question}"

    exact_matches = [
        chunk
        for chunk in chunks_by_id.values()
        if _normalize_text(chunk.text).startswith(expected_qa_start)
    ]

    if exact_matches:
        return [
            RetrievedChunk(
                source=chunk.source,
                text=chunk.text,
                score=1.0,
                chunk_id=chunk.chunk_id,
            )
            for chunk in exact_matches[:k]
        ]

    fused_scores: dict[str, float] = {}

    for rank, chunk in enumerate(semantic_chunks, start=1):
        fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
            1.0 / (60 + rank)
        )

    for rank, chunk in enumerate(lexical_chunks, start=1):
        fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
            1.0 / (60 + rank)
        )

    ranked_ids = sorted(
        fused_scores,
        key=lambda chunk_id: fused_scores[chunk_id],
        reverse=True,
    )[:k]

    return [
        RetrievedChunk(
            source=chunks_by_id[chunk_id].source,
            text=chunks_by_id[chunk_id].text,
            score=round(fused_scores[chunk_id], 4),
            chunk_id=chunk_id,
        )
        for chunk_id in ranked_ids
    ]