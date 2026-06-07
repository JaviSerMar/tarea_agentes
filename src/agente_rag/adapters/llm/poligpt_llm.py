"""Adapter de modelo de lenguaje para la API PoliGPT de la UPV."""

from __future__ import annotations

import time

import requests

from agente_rag.domain.entities import GenerationMetrics
from agente_rag.domain.ports import GeneratedText


class PoliGPTLLMAdapter:
    """Implementación de LLMPort mediante la API compatible con OpenAI de PoliGPT."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        verify_ssl: bool = True,
        timeout: int = 180,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._verify_ssl = verify_ssl
        self._timeout = timeout

    def generate(self, prompt: str, *, temperature: float = 0.2) -> GeneratedText:
        """Genera una respuesta mediante PoliGPT y recoge sus métricas."""
        started_at = time.perf_counter()

        response = requests.post(
            f"{self._base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self._model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": temperature,
            },
            verify=self._verify_ssl,
            timeout=self._timeout,
        )
        response.raise_for_status()

        elapsed_seconds = time.perf_counter() - started_at
        payload = response.json()

        try:
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(
                f"Respuesta inesperada del adapter PoliGPT: {payload}"
            ) from error

        usage = payload.get("usage", {})
        prompt_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        total_tokens = prompt_tokens + output_tokens
        tokens_per_sec = (
            total_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0
        )

        return GeneratedText(
            text=text,
            metrics=GenerationMetrics(
                prompt_tokens=prompt_tokens,
                output_tokens=output_tokens,
                tokens_per_sec=round(tokens_per_sec, 2),
                latency_s=round(elapsed_seconds, 2),
                model=self._model,
            ),
        )