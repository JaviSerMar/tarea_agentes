"""Ejecuta el benchmark del agente RAG DNI con un modelo seleccionado.

Ejemplos de uso:
    python scripts/run_eval.py --provider ollama --model qwen2.5:3b --label local_qwen
    python scripts/run_eval.py --provider ollama --model llama3.2:3b --label local_llama

Las variables de proveedor y modelo se fijan antes de importar el agente para
que la arquitectura hexagonal seleccione el adapter correcto por configuración.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

QUESTIONS_FILE = REPO_ROOT / "benchmark" / "preguntas.json"
RUNS_DIR = REPO_ROOT / "benchmark" / "runs"
REJECTION_PHRASE = "No tengo esa información en mis fuentes"


def parse_args() -> argparse.Namespace:
    """Lee la configuración del modelo que se va a evaluar."""
    parser = argparse.ArgumentParser(description="Benchmark del agente RAG DNI")
    parser.add_argument(
        "--provider",
        choices=["ollama", "poligpt"],
        required=True,
        help="Proveedor del modelo generador.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Nombre del modelo que se evaluará.",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Etiqueta corta usada en el nombre del fichero de resultados.",
    )
    return parser.parse_args()


def configure_model(provider: str, model: str) -> None:
    """Configura el proveedor antes de importar el composition root."""
    os.environ["LLM_PROVIDER"] = provider

    if provider == "ollama":
        os.environ["LLM_MODEL"] = model
    else:
        os.environ["POLIGPT_MODEL"] = model


def source_recall(expected_sources: list[str], returned_sources: list[str]) -> float | None:
    """Calcula qué proporción de fuentes esperadas ha sido recuperada."""
    if not expected_sources:
        return None

    matched_sources = set(expected_sources) & set(returned_sources)
    return round(len(matched_sources) / len(set(expected_sources)), 4)


def rejection_is_correct(question: dict, answer_text: str) -> bool | None:
    """Comprueba el rechazo en preguntas declaradas fuera de ámbito."""
    if question.get("categoria") != "fuera_de_ambito":
        return None

    return REJECTION_PHRASE in answer_text


def main() -> int:
    """Ejecuta todas las preguntas y guarda resultados y resumen."""
    args = parse_args()
    configure_model(args.provider, args.model)

    from consultar import consultar
    from agente_rag.config import SETTINGS

    if not QUESTIONS_FILE.exists():
        print(f"No existe {QUESTIONS_FILE}", file=sys.stderr)
        return 1

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8-sig"))
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    run_started_at = time.time()

    print(f"[benchmark] preguntas     = {len(questions)}")
    print(f"[benchmark] provider      = {args.provider}")
    print(f"[benchmark] model         = {args.model}")
    print(f"[benchmark] vector_store  = {SETTINGS.vector_store_provider}")
    print(f"[benchmark] embedder      = {SETTINGS.embedder_provider}")

    for index, question in enumerate(questions, start=1):
        started_at = time.time()

        try:
            output = consultar(question["pregunta"])
            error = None
        except Exception as exception:
            output = None
            error = str(exception)

        elapsed_seconds = round(time.time() - started_at, 2)

        if output is not None:
            returned_sources = output.get("fuentes", [])
            answer_text = output.get("respuesta", "")
            recall = source_recall(
                question.get("fuentes_esperadas", []),
                returned_sources,
            )
            rejection_ok = rejection_is_correct(question, answer_text)
        else:
            recall = None
            rejection_ok = None

        results.append(
            {
                "id": question["id"],
                "categoria": question["categoria"],
                "pregunta": question["pregunta"],
                "respuesta_referencia": question["respuesta_referencia"],
                "fuentes_esperadas": question["fuentes_esperadas"],
                "source_recall": recall,
                "rejection_ok": rejection_ok,
                "elapsed_s": elapsed_seconds,
                "salida": output,
                "error": error,
            }
        )

        status = "OK" if error is None else "ERROR"
        print(
            f"  [{index}/{len(questions)}] {status} "
            f"({elapsed_seconds:.2f}s) {question['id']} - "
            f"{question['pregunta'][:55]}"
        )

    successful_results = [
        result for result in results if result["salida"] is not None
    ]
    latencies = [
        result["salida"]["metricas"]["latencia_s"]
        for result in successful_results
        if result["salida"].get("metricas")
    ]
    speeds = [
        result["salida"]["metricas"]["tokens_per_sec"]
        for result in successful_results
        if result["salida"].get("metricas")
    ]
    recalls = [
        result["source_recall"]
        for result in results
        if result["source_recall"] is not None
    ]
    rejection_results = [
        result["rejection_ok"]
        for result in results
        if result["rejection_ok"] is not None
    ]

    summary = {
        "provider": args.provider,
        "model": args.model,
        "label": args.label,
        "vector_store_provider": SETTINGS.vector_store_provider,
        "embedder_provider": SETTINGS.embedder_provider,
        "questions_total": len(questions),
        "questions_successful": len(successful_results),
        "avg_latency_s": round(mean(latencies), 4) if latencies else None,
        "avg_tokens_per_sec": round(mean(speeds), 4) if speeds else None,
        "mean_source_recall": round(mean(recalls), 4) if recalls else None,
        "out_of_scope_accuracy": (
            round(sum(rejection_results) / len(rejection_results), 4)
            if rejection_results
            else None
        ),
        "total_elapsed_s": round(time.time() - run_started_at, 2),
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = RUNS_DIR / f"run_{args.label}_{timestamp}.json"
    payload = {
        "summary": summary,
        "results": results,
    }
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("[benchmark] resumen:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[benchmark] resultados -> {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())