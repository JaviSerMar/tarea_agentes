"""Tests del composition root de la arquitectura hexagonal.

Comprueban que los adapters se seleccionan mediante configuración sin
modificar el dominio ni realizar llamadas reales a servicios externos.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agente_rag.adapters.embeddings.ollama_embeddings import (
    OllamaEmbeddingsAdapter,
)
from agente_rag.adapters.embeddings.sentence_transformers_embeddings import (
    SentenceTransformersEmbeddingsAdapter,
)
from agente_rag.adapters.llm.ollama_llm import OllamaLLMAdapter
from agente_rag.adapters.llm.poligpt_llm import PoliGPTLLMAdapter
from agente_rag.adapters.retriever.chroma_vector_store import (
    ChromaVectorStoreAdapter,
)
from agente_rag.adapters.retriever.faiss_vector_store import (
    FaissVectorStoreAdapter,
)
from agente_rag.adapters.retriever.semantic_retriever import (
    SemanticRetrieverAdapter,
)
from agente_rag.composition import (
    build_embedder_adapter,
    build_llm_adapter,
    build_semantic_retriever,
    build_vector_store_adapter,
)


def _settings(**overrides):
    values = {
        "llm_provider": "ollama",
        "ollama_url": "http://localhost:11434/api",
        "llm_model": "qwen2.5:3b",
        "embed_model": "nomic-embed-text",
        "verify_ssl": True,
        "poligpt_base_url": "https://api.poligpt.upv.es/v1",
        "poligpt_api_key": None,
        "poligpt_model": "poligpt",
        "embedder_provider": "ollama",
        "sentence_transformers_model": "modelo-sentence-transformers",
        "vector_store_provider": "chroma",
        "chroma_path": "data/chroma",
        "collection_name": "dni",
        "faiss_path": "data/dni.index",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_build_llm_adapter_selects_ollama_from_configuration():
    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(llm_provider="ollama"),
    ):
        adapter = build_llm_adapter()

    assert isinstance(adapter, OllamaLLMAdapter)


def test_build_llm_adapter_selects_poligpt_from_configuration():
    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(
            llm_provider="poligpt",
            poligpt_api_key="clave-de-prueba",
            poligpt_model="modelo-poligpt-prueba",
        ),
    ):
        adapter = build_llm_adapter()

    assert isinstance(adapter, PoliGPTLLMAdapter)


def test_build_llm_adapter_requires_key_for_poligpt():
    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(llm_provider="poligpt", poligpt_api_key=None),
    ):
        with pytest.raises(RuntimeError, match="POLIGPT_API_KEY"):
            build_llm_adapter()


def test_build_llm_adapter_rejects_unknown_provider():
    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(llm_provider="desconocido"),
    ):
        with pytest.raises(ValueError, match="Proveedor LLM no soportado"):
            build_llm_adapter()


def test_build_embedder_adapter_selects_ollama():
    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(embedder_provider="ollama"),
    ):
        adapter = build_embedder_adapter()

    assert isinstance(adapter, OllamaEmbeddingsAdapter)


def test_build_embedder_adapter_selects_sentence_transformers():
    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(embedder_provider="sentence_transformers"),
    ):
        adapter = build_embedder_adapter()

    assert isinstance(adapter, SentenceTransformersEmbeddingsAdapter)


def test_build_embedder_adapter_rejects_unknown_provider():
    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(embedder_provider="desconocido"),
    ):
        with pytest.raises(ValueError, match="Proveedor de embeddings no soportado"):
            build_embedder_adapter()


def test_build_vector_store_adapter_selects_chroma():
    embedder = OllamaEmbeddingsAdapter(
        base_url="http://localhost:11434/api",
        model="nomic-embed-text",
    )

    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(vector_store_provider="chroma"),
    ):
        adapter = build_vector_store_adapter(embedder=embedder)

    assert isinstance(adapter, ChromaVectorStoreAdapter)


def test_build_vector_store_adapter_selects_faiss():
    embedder = OllamaEmbeddingsAdapter(
        base_url="http://localhost:11434/api",
        model="nomic-embed-text",
    )

    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(vector_store_provider="faiss"),
    ):
        adapter = build_vector_store_adapter(embedder=embedder)

    assert isinstance(adapter, FaissVectorStoreAdapter)


def test_build_vector_store_adapter_rejects_unknown_provider():
    embedder = OllamaEmbeddingsAdapter(
        base_url="http://localhost:11434/api",
        model="nomic-embed-text",
    )

    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(vector_store_provider="desconocido"),
    ):
        with pytest.raises(ValueError, match="Vector store no soportado"):
            build_vector_store_adapter(embedder=embedder)


def test_build_semantic_retriever_uses_configured_components():
    with patch(
        "agente_rag.composition.SETTINGS",
        _settings(
            embedder_provider="ollama",
            vector_store_provider="faiss",
        ),
    ):
        retriever = build_semantic_retriever()

    assert isinstance(retriever, SemanticRetrieverAdapter)