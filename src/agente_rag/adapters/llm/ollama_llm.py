"""Adapter de modelo de lenguaje para Ollama local."""

from __future__ import annotations

import time

import requests

from agente_rag.domain.entities import GenerationMetrics
from agente_rag.domain.ports import GeneratedText


class OllamaLLMAdapter:
    """Implementación de LLMPort mediante la API local de Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        verify_ssl: bool = True,
        timeout: int = 180,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._verify_ssl = verify_ssl
        self._timeout = timeout

    def generate(self, prompt: str, *, temperature: float = 0.2) -> GeneratedText:
        """Genera una respuesta y captura las métricas nativas de Ollama."""
        started_at = time.perf_counter()

        response = requests.post(
            f"{self._base_url}/generate",
            json={
                "model": self._model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature},
            },
            verify=self._verify_ssl,
            timeout=self._timeout,
        )
        response.raise_for_status()

        elapsed_seconds = time.perf_counter() - started_at
        payload = response.json()

        output_tokens = int(payload.get("eval_count", 0))
        eval_duration_ns = int(payload.get("eval_duration", 0))
        tokens_per_sec = (
            output_tokens / (eval_duration_ns / 1_000_000_000)
            if eval_duration_ns > 0
            else 0.0
        )

        return GeneratedText(
            text=payload["response"],
            metrics=GenerationMetrics(
                prompt_tokens=int(payload.get("prompt_eval_count", 0)),
                output_tokens=output_tokens,
                tokens_per_sec=round(tokens_per_sec, 2),
                latency_s=round(elapsed_seconds, 2),
                model=self._model,
            ),
        )