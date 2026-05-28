"""Montaje de dependencias del agente RAG de DNI.

Este módulo es el composition root de la arquitectura hexagonal: selecciona
los adapters concretos mediante configuración y construye el servicio de
dominio utilizado por la interfaz de entrada ``consultar.py``.
"""

from __future__ import annotations

from agente_rag.adapters.llm.ollama_llm import OllamaLLMAdapter
from agente_rag.adapters.llm.poligpt_llm import PoliGPTLLMAdapter
from agente_rag.adapters.retriever.hybrid_retriever import HybridRetrieverAdapter
from agente_rag.config import SETTINGS
from agente_rag.domain.chatbot_service import ChatbotService
from agente_rag.domain.ports import LLMPort


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


def build_chatbot_service() -> ChatbotService:
    """Construye el agente con los adapters seleccionados."""
    return ChatbotService(
        llm=build_llm_adapter(),
        retriever=HybridRetrieverAdapter(),
    )