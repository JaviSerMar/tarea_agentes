"""Tests del adapter alternativo de embeddings Sentence Transformers.

No cargan modelos reales ni requieren red: inyectan un modelo falso para
comprobar la conversión de vectores y la carga diferida del adapter.
"""

from unittest.mock import Mock

from agente_rag.adapters.embeddings.sentence_transformers_embeddings import (
    SentenceTransformersEmbeddingsAdapter,
)


class FakeVector:
    """Vector de prueba que simula un array con método tolist."""

    def __init__(self, values):
        self._values = values

    def tolist(self):
        return self._values


def test_sentence_transformers_adapter_embeds_one_text():
    fake_model = Mock()
    fake_model.encode.return_value = FakeVector([0.1, 0.2, 0.3])
    factory = Mock(return_value=fake_model)

    adapter = SentenceTransformersEmbeddingsAdapter(
        model_name="modelo-prueba",
        model_factory=factory,
    )

    result = adapter.embed("¿Qué es DNI?")

    assert result == [0.1, 0.2, 0.3]
    factory.assert_called_once_with("modelo-prueba")
    fake_model.encode.assert_called_once_with("¿Qué es DNI?")


def test_sentence_transformers_adapter_embeds_many_texts():
    fake_model = Mock()
    fake_model.encode.return_value = FakeVector([[0.1, 0.2], [0.3, 0.4]])
    factory = Mock(return_value=fake_model)

    adapter = SentenceTransformersEmbeddingsAdapter(
        model_name="modelo-prueba",
        model_factory=factory,
    )

    result = adapter.embed_many(["uno", "dos"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]
    factory.assert_called_once_with("modelo-prueba")
    fake_model.encode.assert_called_once_with(["uno", "dos"])


def test_sentence_transformers_adapter_loads_model_only_once():
    fake_model = Mock()
    fake_model.encode.side_effect = [[0.1], [0.2]]
    factory = Mock(return_value=fake_model)

    adapter = SentenceTransformersEmbeddingsAdapter(
        model_name="modelo-prueba",
        model_factory=factory,
    )

    adapter.embed("uno")
    adapter.embed("dos")

    factory.assert_called_once_with("modelo-prueba")