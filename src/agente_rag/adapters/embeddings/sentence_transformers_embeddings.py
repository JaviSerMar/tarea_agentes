"""Adapter de embeddings local basado en Sentence Transformers.

La dependencia se carga de forma diferida para que el proyecto pueda importar
la arquitectura aunque este adapter alternativo no esté seleccionado.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class SentenceTransformersEmbeddingsAdapter:
    """Implementación de EmbedderPort mediante Sentence Transformers."""

    def __init__(
        self,
        *,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self._model_name = model_name
        self._model_factory = model_factory or self._default_model_factory
        self._model: Any | None = None

    @staticmethod
    def _default_model_factory(model_name: str) -> Any:
        """Carga la dependencia únicamente cuando se usa este adapter."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise RuntimeError(
                "Para usar SentenceTransformersEmbeddingsAdapter instala "
                "sentence-transformers."
            ) from error

        return SentenceTransformer(model_name)

    def _get_model(self) -> Any:
        """Crea el modelo una sola vez y lo reutiliza."""
        if self._model is None:
            self._model = self._model_factory(self._model_name)

        return self._model

    def embed(self, text: str) -> list[float]:
        """Transforma un texto en un embedding numérico."""
        vector = self._get_model().encode(text)

        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        return [float(value) for value in vector]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Vectoriza varios textos preservando su orden."""
        vectors = self._get_model().encode(texts)

        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()

        return [[float(value) for value in vector] for vector in vectors]