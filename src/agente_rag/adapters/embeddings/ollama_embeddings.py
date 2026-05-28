"""Adapter de embeddings para Ollama local."""

from __future__ import annotations

import requests


class OllamaEmbeddingsAdapter:
    """Implementación de EmbedderPort mediante la API local de Ollama."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        verify_ssl: bool = True,
        timeout: int = 60,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._verify_ssl = verify_ssl
        self._timeout = timeout

    def embed(self, text: str) -> list[float]:
        """Transforma un texto en un vector numérico."""
        response = requests.post(
            f"{self._base_url}/embeddings",
            json={
                "model": self._model,
                "prompt": text,
            },
            verify=self._verify_ssl,
            timeout=self._timeout,
        )
        response.raise_for_status()

        payload = response.json()

        if "embedding" not in payload:
            raise RuntimeError(
                f"Respuesta inesperada del adapter de embeddings: {payload}"
            )

        return payload["embedding"]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Vectoriza varios textos preservando su orden."""
        return [self.embed(text) for text in texts]