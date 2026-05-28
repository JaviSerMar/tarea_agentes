"""Tests del chunker sobre el corpus oficial DNI. No requieren Ollama ni red."""

from pathlib import Path

import pytest

from agente_rag.chunker import Chunk, load_corpus, split_documents

CORPUS = Path(__file__).resolve().parents[1] / "corpus"

EXPECTED_DNI_FILES = {
    "01_faq_dni.txt",
    "02_presentacion_desayunos.txt",
    "03_charlas_abuelitos.txt",
    "04_filosofia_dni.txt",
    "05_resis_actividades.txt",
    "06_coles_refuerzo.txt",
    "07_desayunos_logistica.txt",
    "08_preguntas_basicas.txt",
    "09_como_participar.txt",
    "10_proyectos.txt",
    "11_horarios_ubicaciones.txt",
    "12_contacto_redes.txt",
    "13_mensajes_genericos.txt",
    "14_impacto_social.txt",
    "15_desayunos_100_preguntas.txt",
    "16_resis_49_preguntas.txt",
}


def test_load_corpus_returns_official_dni_docs():
    docs = load_corpus(CORPUS)

    assert len(docs) == 16
    assert all("name" in document and "text" in document for document in docs)
    assert {document["name"] for document in docs} == EXPECTED_DNI_FILES


def test_load_corpus_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "no-existe")


def test_split_documents_preserves_source():
    docs = load_corpus(CORPUS)
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)

    assert len(chunks) > 20, "se esperan múltiples chunks con el corpus DNI"
    assert all(isinstance(chunk, Chunk) for chunk in chunks)
    assert all(chunk.source in EXPECTED_DNI_FILES for chunk in chunks)
    assert all(chunk.text and isinstance(chunk.text, str) for chunk in chunks)
    assert all(
        chunk.id.endswith(f"chunk_{chunk.chunk_index:04d}") for chunk in chunks
    )


def test_split_documents_preserves_qa_pairs():
    docs = load_corpus(CORPUS)
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)

    what_is_dni = [
        chunk
        for chunk in chunks
        if "¿Qué es DNI?" in chunk.text
    ]

    assert len(what_is_dni) == 1
    assert "DNI (Damos Nuestra Ilusión) es una asociación" in what_is_dni[0].text
    assert what_is_dni[0].source == "08_preguntas_basicas.txt"


def test_narrative_chunks_keep_size_bounded():
    docs = load_corpus(CORPUS)
    chunks = split_documents(docs, chunk_size=500, chunk_overlap=100)

    narrative_chunks = [
        chunk for chunk in chunks if not chunk.text.lstrip().startswith("Q:")
    ]
    too_large = [chunk for chunk in narrative_chunks if len(chunk.text) > 600]

    assert not too_large, f"chunks narrativos demasiado grandes: {len(too_large)}"