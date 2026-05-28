"""Tests del composition root de la arquitectura hexagonal.

Comprueban que el proveedor LLM se selecciona mediante configuración sin
modificar el dominio ni realizar llamadas reales a servicios externos.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agente_rag.adapters.llm.ollama_llm import OllamaLLMAdapter
from agente_rag.adapters.llm.poligpt_llm import PoliGPTLLMAdapter
from agente_rag.composition import build_llm_adapter


def _settings(**overrides):
    values = {
        "llm_provider": "ollama",
        "ollama_url": "http://localhost:11434/api",
        "llm_model": "qwen2.5:3b",
        "verify_ssl": True,
        "poligpt_base_url": "https://api.poligpt.upv.es/v1",
        "poligpt_api_key": None,
        "poligpt_model": "poligpt",
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