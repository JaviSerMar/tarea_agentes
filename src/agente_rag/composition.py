"""Montaje de dependencias del agente RAG de DNI.

Este módulo es el composition root de la arquitectura hexagonal: selecciona
los adapters concretos y construye el servicio de dominio que usa la interfaz
de entrada ``consultar.py``.
"""

from __future__ import annotations

from agente_rag.adapters.llm.ollama_llm import OllamaLLMAdapter
from agente_rag.adapters.retriever.hybrid_retriever import HybridRetrieverAdapter
from agente_rag.config import SETTINGS
from agente_rag.domain.chatbot_service import ChatbotService


def build_chatbot_service() -> ChatbotService:
    """Construye el agente usando Ollama local y retrieval híbrido."""
    llm = OllamaLLMAdapter(
        base_url=SETTINGS.ollama_url,
        model=SETTINGS.llm_model,
        verify_ssl=SETTINGS.verify_ssl,
    )
    retriever = HybridRetrieverAdapter()

    return ChatbotService(llm=llm, retriever=retriever)