"""Tests del adapter de embeddings de Ollama.

No realizan llamadas reales a Ollama: se simula la respuesta HTTP para
comprobar la transformación de textos en vectores y el manejo de errores.
"""

from unittest.mock import Mock, patch

import pytest

from agente_rag.adapters.embeddings.ollama_embeddings import (
    OllamaEmbeddingsAdapter,
)


def test_ollama_embeddings_adapter_returns_embedding_vector():
    fake_response = Mock()
    fake_response.json.return_value = {
        "embedding": [0.1, 0.2, 0.3],
    }

    adapter = OllamaEmbeddingsAdapter(
        base_url="http://localhost:11434/api",
        model="nomic-embed-text",
    )

    with patch(
        "agente_rag.adapters.embeddings.ollama_embeddings.requests.post",
        return_value=fake_response,
    ) as post_mock:
        result = adapter.embed("¿Qué es DNI?")

    fake_response.raise_for_status.assert_called_once()
    assert result == [0.1, 0.2, 0.3]

    post_mock.assert_called_once_with(
        "http://localhost:11434/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": "¿Qué es DNI?",
        },
        verify=True,
        timeout=60,
    )


def test_ollama_embeddings_adapter_embed_many_preserves_order():
    adapter = OllamaEmbeddingsAdapter(
        base_url="http://localhost:11434/api",
        model="nomic-embed-text",
    )

    with patch.object(
        adapter,
        "embed",
        side_effect=[[0.1], [0.2], [0.3]],
    ) as embed_mock:
        result = adapter.embed_many(["uno", "dos", "tres"])

    assert result == [[0.1], [0.2], [0.3]]
    assert embed_mock.call_count == 3


def test_ollama_embeddings_adapter_rejects_missing_embedding():
    fake_response = Mock()
    fake_response.json.return_value = {
        "unexpected": "payload",
    }

    adapter = OllamaEmbeddingsAdapter(
        base_url="http://localhost:11434/api",
        model="nomic-embed-text",
    )

    with patch(
        "agente_rag.adapters.embeddings.ollama_embeddings.requests.post",
        return_value=fake_response,
    ):
        with pytest.raises(RuntimeError, match="Respuesta inesperada"):
            adapter.embed("texto")