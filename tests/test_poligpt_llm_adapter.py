"""Tests del adapter PoliGPT para la arquitectura hexagonal.

No realizan llamadas reales a la API: simulan respuestas HTTP para comprobar
el contrato del dominio, el envío seguro de credenciales y el manejo de errores.
"""

from unittest.mock import Mock, patch

import pytest

from agente_rag.adapters.llm.poligpt_llm import PoliGPTLLMAdapter


def test_poligpt_llm_adapter_returns_generated_text_with_metrics():
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "DNI es una asociación de voluntariado juvenil."
                }
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 30,
        },
    }

    adapter = PoliGPTLLMAdapter(
        base_url="https://api.poligpt.upv.es/v1",
        api_key="clave-de-prueba",
        model="modelo-poligpt-prueba",
    )

    with patch(
        "agente_rag.adapters.llm.poligpt_llm.requests.post",
        return_value=fake_response,
    ) as post_mock:
        result = adapter.generate("Prompt de prueba", temperature=0.2)

    fake_response.raise_for_status.assert_called_once()

    assert result.text == "DNI es una asociación de voluntariado juvenil."
    assert result.metrics.prompt_tokens == 120
    assert result.metrics.output_tokens == 30
    assert result.metrics.model == "modelo-poligpt-prueba"
    assert result.metrics.latency_s >= 0.0

    call = post_mock.call_args
    assert call.args[0] == "https://api.poligpt.upv.es/v1/chat/completions"
    assert call.kwargs["headers"]["Authorization"] == "Bearer clave-de-prueba"
    assert call.kwargs["json"] == {
        "model": "modelo-poligpt-prueba",
        "messages": [
            {
                "role": "user",
                "content": "Prompt de prueba",
            }
        ],
        "temperature": 0.2,
    }
    assert call.kwargs["verify"] is True
    assert call.kwargs["timeout"] == 180


def test_poligpt_llm_adapter_supports_custom_ssl_and_timeout():
    fake_response = Mock()
    fake_response.json.return_value = {
        "choices": [{"message": {"content": "Respuesta."}}],
        "usage": {},
    }

    adapter = PoliGPTLLMAdapter(
        base_url="https://servidor-poligpt/v1/",
        api_key="token-test",
        model="modelo",
        verify_ssl=False,
        timeout=60,
    )

    with patch(
        "agente_rag.adapters.llm.poligpt_llm.requests.post",
        return_value=fake_response,
    ) as post_mock:
        result = adapter.generate("Pregunta")

    assert result.metrics.prompt_tokens == 0
    assert result.metrics.output_tokens == 0

    call = post_mock.call_args
    assert call.args[0] == "https://servidor-poligpt/v1/chat/completions"
    assert call.kwargs["verify"] is False
    assert call.kwargs["timeout"] == 60


def test_poligpt_llm_adapter_rejects_unexpected_payload():
    fake_response = Mock()
    fake_response.json.return_value = {"unexpected": "payload"}

    adapter = PoliGPTLLMAdapter(
        base_url="https://api.poligpt.upv.es/v1",
        api_key="clave-de-prueba",
        model="modelo",
    )

    with patch(
        "agente_rag.adapters.llm.poligpt_llm.requests.post",
        return_value=fake_response,
    ):
        with pytest.raises(RuntimeError, match="Respuesta inesperada"):
            adapter.generate("Prompt")