"""Montaje de dependencias del agente RAG de DNI.

Este módulo es el composition root de la arquitectura hexagonal: selecciona
los adapters concretos mediante configuración y construye los componentes
utilizados por la interfaz de entrada ``consultar.py``.
"""

from __future__ import annotations

from agente_rag.adapters.embeddings.ollama_embeddings import (
    OllamaEmbeddingsAdapter,
)
from agente_rag.adapters.embeddings.sentence_transformers_embeddings import (
    SentenceTransformersEmbeddingsAdapter,
)
from agente_rag.adapters.llm.ollama_llm import OllamaLLMAdapter
from agente_rag.adapters.llm.poligpt_llm import PoliGPTLLMAdapter
from agente_rag.adapters.retriever.bm25_retriever import BM25RetrieverAdapter
from agente_rag.adapters.retriever.chroma_vector_store import (
    ChromaVectorStoreAdapter,
)
from agente_rag.adapters.retriever.faiss_vector_store import (
    FaissVectorStoreAdapter,
)
from agente_rag.adapters.retriever.hybrid_retriever import HybridRetrieverAdapter
from agente_rag.adapters.retriever.semantic_retriever import (
    SemanticRetrieverAdapter,
)
from agente_rag.chunker import load_corpus, split_documents
from agente_rag.config import SETTINGS
from agente_rag.domain.chatbot_service import ChatbotService
from agente_rag.domain.entities import Chunk
from agente_rag.domain.ports import EmbedderPort, LLMPort, RetrieverPort, VectorStorePort


def build_llm_adapter() -> LLMPort:
    """Selecciona el adapter LLM indicado en la configuración."""
    if SETTINGS.llm_provider == "ollama":
        return OllamaLLMAdapter(
            base_url=SETTINGS.ollama_url,
            model=SETTINGS.llm_model,
            verify_ssl=SETTINGS.verify_ssl,
        )

    if SETTINGS.llm_provider == "poligpt":
        if not SETTINGS.poligpt_api_key:
            raise RuntimeError(
                "Falta POLIGPT_API_KEY en el archivo .env para usar PoliGPT."
            )

        return PoliGPTLLMAdapter(
            base_url=SETTINGS.poligpt_base_url,
            api_key=SETTINGS.poligpt_api_key,
            model=SETTINGS.poligpt_model,
            verify_ssl=SETTINGS.verify_ssl,
        )

    raise ValueError(
        f"Proveedor LLM no soportado: {SETTINGS.llm_provider!r}. "
        "Usa 'ollama' o 'poligpt'."
    )


def build_embedder_adapter() -> EmbedderPort:
    """Selecciona el adapter de embeddings indicado en la configuración."""
    if SETTINGS.embedder_provider == "ollama":
        return OllamaEmbeddingsAdapter(
            base_url=SETTINGS.ollama_url,
            model=SETTINGS.embed_model,
            verify_ssl=SETTINGS.verify_ssl,
        )

    if SETTINGS.embedder_provider == "sentence_transformers":
        return SentenceTransformersEmbeddingsAdapter(
            model_name=SETTINGS.sentence_transformers_model,
        )

    raise ValueError(
        f"Proveedor de embeddings no soportado: {SETTINGS.embedder_provider!r}. "
        "Usa 'ollama' o 'sentence_transformers'."
    )


def build_vector_store_adapter(
    *,
    embedder: EmbedderPort | None = None,
) -> VectorStorePort:
    """Selecciona el vector store indicado en la configuración."""
    selected_embedder = embedder or build_embedder_adapter()

    if SETTINGS.vector_store_provider == "chroma":
        return ChromaVectorStoreAdapter(
            path=SETTINGS.chroma_path,
            collection_name=SETTINGS.collection_name,
            embedder=selected_embedder,
        )

    if SETTINGS.vector_store_provider == "faiss":
        return FaissVectorStoreAdapter(
            path=SETTINGS.faiss_path,
            embedder=selected_embedder,
        )

    raise ValueError(
        f"Vector store no soportado: {SETTINGS.vector_store_provider!r}. "
        "Usa 'chroma' o 'faiss'."
    )


def build_semantic_retriever() -> RetrieverPort:
    """Construye el canal semántico con adapters configurables."""
    embedder = build_embedder_adapter()
    vector_store = build_vector_store_adapter(embedder=embedder)

    return SemanticRetrieverAdapter(
        embedder=embedder,
        vector_store=vector_store,
    )


def build_lexical_retriever() -> RetrieverPort:
    """Construye el canal BM25 desde el corpus oficial DNI."""
    documents = load_corpus(SETTINGS.corpus_dir)
    source_chunks = split_documents(documents)

    domain_chunks = [
        Chunk(
            source=chunk.source,
            text=chunk.text,
            score=0.0,
            chunk_id=chunk.id,
        )
        for chunk in source_chunks
    ]

    return BM25RetrieverAdapter(domain_chunks)


def build_retriever_adapter() -> RetrieverPort:
    """Construye la recuperación híbrida semántica + léxica."""
    return HybridRetrieverAdapter(
        semantic_retriever=build_semantic_retriever(),
        lexical_retriever=build_lexical_retriever(),
    )


def build_chatbot_service() -> ChatbotService:
    """Construye el agente completo con adapters seleccionables."""
    return ChatbotService(
        llm=build_llm_adapter(),
        retriever=build_retriever_adapter(),
    )