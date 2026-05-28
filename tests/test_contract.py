"""Tests del contrato de entrada conectado a la arquitectura hexagonal.

No llaman a Ollama ni a ChromaDB: se sustituye el servicio construido por
``consultar.py`` por un servicio falso que devuelve respuestas controladas.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

from agente_rag.domain.entities import Answer, Chunk, GenerationMetrics, Question

CONTRACT_KEYS = {"respuesta", "fuentes", "chunks", "metricas", "trazas"}


def _fake_metrics() -> GenerationMetrics:
    return GenerationMetrics(
        prompt_tokens=420,
        output_tokens=37,
        tokens_per_sec=42.1,
        latency_s=1.8,
        model="qwen2.5:3b",
    )


def _dni_chunk() -> Chunk:
    return Chunk(
        source="08_preguntas_basicas.txt",
        text=(
            "Q: ¿Qué es DNI?\n"
            "A: DNI (Damos Nuestra Ilusión) es una asociación de jóvenes "
            "voluntarios en Valencia."
        ),
        score=1.0,
        chunk_id="08_preguntas_basicas.txt__chunk_0000",
    )


def _fake_answer(
    *,
    text: str = "DNI es una asociación de jóvenes voluntarios en Valencia.",
    sources: list[str] | None = None,
    chunks: list[Chunk] | None = None,
    conversation_id: str | None = None,
) -> Answer:
    selected_chunks = chunks if chunks is not None else [_dni_chunk()]
    selected_sources = (
        sources if sources is not None else ["08_preguntas_basicas.txt"]
    )

    return Answer(
        text=text,
        sources=selected_sources,
        chunks=selected_chunks,
        metrics=_fake_metrics(),
        traces=None,
        conversation_id=conversation_id,
    )


def _fake_service(answer: Answer) -> Mock:
    service = Mock()
    service.answer.return_value = answer
    return service


def test_consultar_signature_and_keys():
    import consultar

    service = _fake_service(_fake_answer())

    with patch("consultar.build_chatbot_service", return_value=service):
        output = consultar.consultar("¿Qué es DNI?")

    assert set(output.keys()) >= CONTRACT_KEYS
    assert isinstance(output["respuesta"], str) and output["respuesta"]
    assert isinstance(output["fuentes"], list)
    assert all(isinstance(source, str) for source in output["fuentes"])
    assert isinstance(output["chunks"], list)
    assert isinstance(output["metricas"], dict)

    submitted_question = service.answer.call_args.args[0]
    assert isinstance(submitted_question, Question)
    assert submitted_question.text == "¿Qué es DNI?"


def test_consultar_accepts_conversation_id():
    import consultar

    service = _fake_service(_fake_answer(conversation_id="conv-42"))

    with patch("consultar.build_chatbot_service", return_value=service):
        output = consultar.consultar("¿Qué es DNI?", conversation_id="conv-42")

    submitted_question = service.answer.call_args.args[0]

    assert submitted_question.conversation_id == "conv-42"
    assert output["conversation_id"] == "conv-42"


def test_consultar_serializes_sources_preserving_domain_order():
    import consultar

    chunks = [
        Chunk(
            source="08_preguntas_basicas.txt",
            text="Información general sobre DNI.",
            score=0.9,
            chunk_id="a",
        ),
        Chunk(
            source="04_filosofia_dni.txt",
            text="Filosofía de DNI.",
            score=0.8,
            chunk_id="b",
        ),
    ]

    service = _fake_service(
        _fake_answer(
            sources=["08_preguntas_basicas.txt", "04_filosofia_dni.txt"],
            chunks=chunks,
        )
    )

    with patch("consultar.build_chatbot_service", return_value=service):
        output = consultar.consultar("¿Cuál es la filosofía de DNI?")

    assert output["fuentes"] == [
        "08_preguntas_basicas.txt",
        "04_filosofia_dni.txt",
    ]
    assert output["chunks"][0]["source"] == "08_preguntas_basicas.txt"
    assert output["chunks"][1]["source"] == "04_filosofia_dni.txt"


def test_metricas_have_banda7_fields():
    import consultar

    service = _fake_service(_fake_answer())

    with patch("consultar.build_chatbot_service", return_value=service):
        output = consultar.consultar("¿Qué es DNI?")

    metricas = output["metricas"]

    for key in ("prompt_tokens", "output_tokens", "tokens_per_sec", "latencia_s"):
        assert key in metricas, f"falta métrica {key!r}"

    assert metricas["output_tokens"] == 37
    assert metricas["tokens_per_sec"] == 42.1
    assert metricas["modelo"] == "qwen2.5:3b"


def test_consultar_serializes_validated_contradictory_answer():
    import consultar

    chunks = [
        Chunk(
            source="01_faq_dni.txt",
            text=(
                "Q: ¿A qué hora son los desayunos solidarios?\n"
                "A: Los desayunos son a las 8 de la mañana."
            ),
            score=1.0,
            chunk_id="faq_horario",
        ),
        Chunk(
            source="11_horarios_ubicaciones.txt",
            text=(
                "Q: ¿A qué hora son los desayunos solidarios?\n"
                "A: Los desayunos suelen realizarse entre las 9:00 y las 12:00h."
            ),
            score=1.0,
            chunk_id="horarios_horario",
        ),
    ]

    validated_text = (
        "Las fuentes contienen varias versiones para esta pregunta:\n"
        "- Según 01_faq_dni.txt: Los desayunos son a las 8 de la mañana.\n"
        "- Según 11_horarios_ubicaciones.txt: Los desayunos suelen realizarse "
        "entre las 9:00 y las 12:00h."
    )

    service = _fake_service(
        _fake_answer(
            text=validated_text,
            sources=["01_faq_dni.txt", "11_horarios_ubicaciones.txt"],
            chunks=chunks,
        )
    )

    with patch("consultar.build_chatbot_service", return_value=service):
        output = consultar.consultar("¿A qué hora son los desayunos solidarios?")

    assert "varias versiones" in output["respuesta"]
    assert "8 de la mañana" in output["respuesta"]
    assert "9:00" in output["respuesta"]
    assert output["fuentes"] == [
        "01_faq_dni.txt",
        "11_horarios_ubicaciones.txt",
    ]