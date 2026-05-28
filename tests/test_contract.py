"""Tests del contrato y de las salvaguardas del agente DNI.

No llamamos a Ollama: sustituimos ``retrieve`` y ``generate`` por datos
controlados para verificar el formato de salida y la gestión de contradicciones.
"""

from __future__ import annotations

from unittest.mock import patch

from agente_rag.generator import Generation
from agente_rag.retriever import RetrievedChunk

CONTRACT_KEYS = {"respuesta", "fuentes", "chunks", "metricas", "trazas"}


def _fake_retrieved() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            source="08_preguntas_basicas.txt",
            text=(
                "Q: ¿Qué es DNI?\n"
                "A: DNI (Damos Nuestra Ilusión) es una asociación de jóvenes "
                "voluntarios en Valencia."
            ),
            score=1.0,
            chunk_id="08_preguntas_basicas.txt__chunk_0000",
        )
    ]


def _fake_generation(
    text: str = "DNI es una asociación de jóvenes voluntarios en Valencia.",
) -> Generation:
    return Generation(
        text=text,
        prompt_tokens=420,
        output_tokens=37,
        tokens_per_sec=42.1,
        latency_s=1.8,
        model="qwen2.5:3b",
    )


def test_consultar_signature_and_keys():
    import consultar

    with patch(
        "agente_rag.pipeline.retrieve",
        return_value=_fake_retrieved(),
    ), patch(
        "agente_rag.pipeline.generate",
        return_value=_fake_generation(),
    ):
        out = consultar.consultar("¿Qué es DNI?")

    assert set(out.keys()) >= CONTRACT_KEYS, (
        f"faltan claves: {CONTRACT_KEYS - set(out.keys())}"
    )
    assert isinstance(out["respuesta"], str) and out["respuesta"]
    assert isinstance(out["fuentes"], list)
    assert all(isinstance(source, str) for source in out["fuentes"])
    assert isinstance(out["chunks"], list)
    assert isinstance(out["metricas"], dict)


def test_consultar_accepts_conversation_id():
    import consultar

    with patch(
        "agente_rag.pipeline.retrieve",
        return_value=_fake_retrieved(),
    ), patch(
        "agente_rag.pipeline.generate",
        return_value=_fake_generation(),
    ):
        out = consultar.consultar("¿Qué es DNI?", conversation_id="conv-42")

    assert out["conversation_id"] == "conv-42"


def test_fuentes_are_unique_and_preserve_order():
    import consultar

    chunks = [
        RetrievedChunk(
            source="08_preguntas_basicas.txt",
            text="Información general sobre DNI.",
            score=0.9,
            chunk_id="a",
        ),
        RetrievedChunk(
            source="04_filosofia_dni.txt",
            text="Filosofía de DNI.",
            score=0.8,
            chunk_id="b",
        ),
        RetrievedChunk(
            source="08_preguntas_basicas.txt",
            text="Más información general sobre DNI.",
            score=0.7,
            chunk_id="c",
        ),
    ]

    with patch(
        "agente_rag.pipeline.retrieve",
        return_value=chunks,
    ), patch(
        "agente_rag.pipeline.generate",
        return_value=_fake_generation(),
    ):
        out = consultar.consultar("¿Cuál es la filosofía de DNI?")

    assert out["fuentes"] == [
        "08_preguntas_basicas.txt",
        "04_filosofia_dni.txt",
    ]


def test_metricas_have_banda7_fields():
    import consultar

    with patch(
        "agente_rag.pipeline.retrieve",
        return_value=_fake_retrieved(),
    ), patch(
        "agente_rag.pipeline.generate",
        return_value=_fake_generation(),
    ):
        out = consultar.consultar("¿Qué es DNI?")

    metricas = out["metricas"]

    for key in ("prompt_tokens", "output_tokens", "tokens_per_sec", "latencia_s"):
        assert key in metricas, f"falta métrica {key!r}"

    assert metricas["output_tokens"] == 37
    assert metricas["tokens_per_sec"] == 42.1
    assert metricas["modelo"] == "qwen2.5:3b"


def test_multiple_exact_answers_show_all_versions():
    import consultar

    contradictory_chunks = [
        RetrievedChunk(
            source="01_faq_dni.txt",
            text=(
                "Q: ¿A qué hora son los desayunos solidarios?\n"
                "A: Los desayunos son a las 8 de la mañana."
            ),
            score=1.0,
            chunk_id="faq_horario",
        ),
        RetrievedChunk(
            source="11_horarios_ubicaciones.txt",
            text=(
                "Q: ¿A qué hora son los desayunos solidarios?\n"
                "A: Los desayunos suelen realizarse entre las 9:00 y las 12:00h."
            ),
            score=1.0,
            chunk_id="horarios_horario",
        ),
    ]

    with patch(
        "agente_rag.pipeline.retrieve",
        return_value=contradictory_chunks,
    ), patch(
        "agente_rag.pipeline.generate",
        return_value=_fake_generation("Los desayunos son a las 8 de la mañana."),
    ):
        out = consultar.consultar("¿A qué hora son los desayunos solidarios?")

    assert "varias versiones" in out["respuesta"]
    assert "8 de la mañana" in out["respuesta"]
    assert "9:00" in out["respuesta"]
    assert "01_faq_dni.txt" in out["respuesta"]
    assert "11_horarios_ubicaciones.txt" in out["respuesta"]
    assert out["fuentes"] == [
        "01_faq_dni.txt",
        "11_horarios_ubicaciones.txt",
    ]
