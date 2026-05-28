"""Tests del adapter Ollama para la arquitectura hexagonal.

No realizan llamadas reales a Ollama: se simula la respuesta HTTP para
comprobar la conversión al contrato del dominio y el envío de parámetros.
"""

from unittest.mock import Mock, patch

from agente_rag.adapters.llm.ollama_llm import OllamaLLMAdapter


def test_ollama_llm_adapter_returns_generated_text_with_metrics():
    fake_response = Mock()
    fake_response.json.return_value = {
        "response": "DNI es una asociación de voluntariado juvenil.",
        "prompt_eval_count": 120,
        "eval_count": 30,
        "eval_duration": 1_000_000_000,
    }

    adapter = OllamaLLMAdapter(
        base_url="http://localhost:11434/api",
        model="qwen2.5:3b",
    )

    with patch(
        "agente_rag.adapters.llm.ollama_llm.requests.post",
        return_value=fake_response,
    ) as post_mock:
        result = adapter.generate("Prompt de prueba", temperature=0.2)

    fake_response.raise_for_status.assert_called_once()

    assert result.text == "DNI es una asociación de voluntariado juvenil."
    assert result.metrics.prompt_tokens == 120
    assert result.metrics.output_tokens == 30
    assert result.metrics.tokens_per_sec == 30.0
    assert result.metrics.model == "qwen2.5:3b"

    post_mock.assert_called_once_with(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:3b",
            "prompt": "Prompt de prueba",
            "stream": False,
            "options": {"temperature": 0.2},
        },
        verify=True,
        timeout=180,
    )


def test_ollama_llm_adapter_allows_custom_ssl_and_timeout():
    fake_response = Mock()
    fake_response.json.return_value = {
        "response": "Respuesta.",
        "prompt_eval_count": 0,
        "eval_count": 0,
        "eval_duration": 0,
    }

    adapter = OllamaLLMAdapter(
        base_url="https://servidor-ejemplo/api/",
        model="modelo-prueba",
        verify_ssl=False,
        timeout=60,
    )

    with patch(
        "agente_rag.adapters.llm.ollama_llm.requests.post",
        return_value=fake_response,
    ) as post_mock:
        result = adapter.generate("Pregunta")

    assert result.metrics.tokens_per_sec == 0.0

    post_mock.assert_called_once_with(
        "https://servidor-ejemplo/api/generate",
        json={
            "model": "modelo-prueba",
            "prompt": "Pregunta",
            "stream": False,
            "options": {"temperature": 0.2},
        },
        verify=False,
        timeout=60,
    )