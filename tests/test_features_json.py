"""Tests del features.json — el corrector aborta si no es válido."""

import json
from pathlib import Path

FEATURES = Path(__file__).resolve().parents[1] / "features.json"


def test_features_json_exists_and_is_object():
    assert FEATURES.exists(), "features.json es obligatorio (enunciado §9.1)"
    data = json.loads(FEATURES.read_text(encoding="utf-8"))
    assert isinstance(data, dict), "features.json debe ser un objeto JSON"


def test_features_json_declares_consistent_interface():
    data = json.loads(FEATURES.read_text(encoding="utf-8"))
    assert data["interfaz"] in {"opcion_a", "opcion_b"}

    if data["interfaz"] == "opcion_a":
        assert (FEATURES.parent / data["modulo"]).exists(), (
            "interfaz=opcion_a pero el módulo declarado no existe"
        )
        assert data["endpoint_http"] is None
    else:
        assert isinstance(data["endpoint_http"], str) and data["endpoint_http"], (
            "interfaz=opcion_b requiere endpoint_http no vacío"
        )


def test_features_json_band_keys():
    data = json.loads(FEATURES.read_text(encoding="utf-8"))
    bandas = data["bandas_declaradas"]
    expected = {
        "banda5_pipeline_no_inventa",
        "banda6_cita_archivo",
        "banda7_benchmark_4_modelos",
        "banda8_ragas_metricas_propias",
        "banda10_hexagonal",
    }
    assert set(bandas.keys()) == expected
    assert all(isinstance(v, bool) for v in bandas.values())
