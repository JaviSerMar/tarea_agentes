"""Consolida las cuatro ejecuciones del benchmark DNI en artefactos entregables."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

RUNS_DIR = Path("benchmark/runs")
OUTPUT_JSON = Path("benchmark/benchmark.json")
OUTPUT_MD = Path("benchmark/benchmark.md")

EXPECTED_RUNS = [
    (
        "local_qwen",
        "Ollama local",
        "qwen2.5:3b",
        "run_local_qwen_20260529_000447.json",
    ),
    (
        "local_llama",
        "Ollama local",
        "llama3.2:3b",
        "run_local_llama_20260529_000701.json",
    ),
    (
        "poligpt_gemma27b",
        "PoliGPT",
        "gemma3:27b",
        "run_poligpt_gemma27b_20260529_175035.json",
    ),
    (
        "poligpt_llama70b",
        "PoliGPT",
        "llama3.3:70b",
        "run_poligpt_llama70b_20260529_175755.json",
    ),
]

SUBJECTIVE_REVIEW = {
    "local_qwen": {
        "fallos": {},
        "comentario": "Responde correctamente las 12 preguntas y gestiona la contradicción horaria.",
    },
    "local_llama": {
        "fallos": {
            "q07": (
                "Detecta una supuesta contradicción de ubicación y termina rechazando "
                "la respuesta, aunque dispone de la ubicación esperada."
            ),
            "q08": (
                "Presenta como contradicción la documentación recuperada y no ofrece "
                "la respuesta definitiva esperada."
            ),
        },
        "comentario": (
            "Obtiene buenas métricas automáticas, pero sobredetecta contradicciones "
            "en dos respuestas."
        ),
    },
    "poligpt_gemma27b": {
        "fallos": {},
        "comentario": (
            "Responde correctamente las 12 preguntas y mantiene la respuesta principal "
            "cuando aparece información alternativa."
        ),
    },
    "poligpt_llama70b": {
        "fallos": {},
        "comentario": (
            "Responde correctamente las 12 preguntas, aunque con mayor tiempo de ejecución "
            "que gemma3:27b."
        ),
    },
}


def load_reviewed_run(filename: str) -> dict[str, Any]:
    path = RUNS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No se ha encontrado la ejecución revisada: {filename}"
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    payload["_source_file"] = path.name
    return payload


def enrich_summary(
    run: dict[str, Any], display_provider: str, expected_model: str
) -> dict[str, Any]:
    summary = dict(run["summary"])
    results = run["results"]

    elapsed_values = [
        result["elapsed_s"] for result in results if result.get("error") is None
    ]

    return {
        "provider": display_provider,
        "model": expected_model,
        "label": summary["label"],
        "source_file": run["_source_file"],
        "vector_store_provider": summary["vector_store_provider"],
        "embedder_provider": summary.get("embedder_provider", "ollama"),
        "questions_total": summary["questions_total"],
        "questions_successful": summary["questions_successful"],
        "avg_llm_latency_s": summary["avg_latency_s"],
        "avg_end_to_end_s": round(mean(elapsed_values), 4),
        "avg_tokens_per_sec": summary["avg_tokens_per_sec"],
        "mean_source_recall": summary["mean_source_recall"],
        "out_of_scope_accuracy": summary["out_of_scope_accuracy"],
        "total_elapsed_s": summary["total_elapsed_s"],
    }


def build_payload() -> dict[str, Any]:
    consolidated_runs = []

    for label, provider, model, filename in EXPECTED_RUNS:
        run = load_reviewed_run(filename)
        summary = enrich_summary(run, provider, model)

        review = SUBJECTIVE_REVIEW[label]
        reviewed_results = []

        for result in run["results"]:
            observation = review["fallos"].get(result["id"], "")
            subjective_quality = "FALLO" if observation else "ACIERTO"

            reviewed_results.append(
                {
                    "id": result["id"],
                    "categoria": result["categoria"],
                    "pregunta": result["pregunta"],
                    "respuesta_referencia": result["respuesta_referencia"],
                    "respuesta_modelo": result["salida"]["respuesta"]
                    if result.get("salida")
                    else None,
                    "fuentes_esperadas": result["fuentes_esperadas"],
                    "fuentes_recuperadas": result["salida"]["fuentes"]
                    if result.get("salida")
                    else [],
                    "source_recall": result["source_recall"],
                    "rejection_ok": result["rejection_ok"],
                    "elapsed_s": result["elapsed_s"],
                    "metricas": result["salida"]["metricas"]
                    if result.get("salida")
                    else None,
                    "calidad_subjetiva": subjective_quality,
                    "observacion": observation,
                }
            )

        summary["subjective_hits"] = sum(
            result["calidad_subjetiva"] == "ACIERTO"
            for result in reviewed_results
        )
        summary["subjective_fails"] = sum(
            result["calidad_subjetiva"] == "FALLO"
            for result in reviewed_results
        )
        summary["subjective_accuracy"] = round(
            summary["subjective_hits"] / summary["questions_total"], 4
        )
        summary["subjective_comment"] = review["comentario"]

        consolidated_runs.append(
            {
                "summary": summary,
                "results": reviewed_results,
            }
        )

    return {
        "benchmark": "Agente RAG DNI - comparación de cuatro modelos",
        "pipeline_constante": {
            "vector_store": "faiss",
            "embedder": "ollama / nomic-embed-text",
            "retrieval": "híbrido semántico + BM25",
            "preguntas": 12,
            "nota": (
                "Solo cambia el LLM generativo; se conserva el mismo corpus, "
                "chunking, retrieval, vector store y embeddings."
            ),
        },
        "metricas": {
            "avg_llm_latency_s": (
                "Latencia media registrada por el adapter del modelo generativo."
            ),
            "avg_end_to_end_s": (
                "Tiempo medio completo medido por el script por pregunta, "
                "incluyendo retrieval, composición y generación."
            ),
            "avg_tokens_per_sec": "Velocidad media de generación registrada por el adapter.",
            "mean_source_recall": (
                "Proporción media de fuentes esperadas recuperadas por el agente."
            ),
            "out_of_scope_accuracy": (
                "Acierto en preguntas fuera de ámbito: debe rechazar sin inventar."
            ),
        },
        "runs": consolidated_runs,
    }


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Benchmark DNI — comparación de cuatro modelos",
        "",
        "## Condiciones de comparación",
        "",
        "Se han evaluado cuatro modelos sobre el mismo conjunto fijo de 12 preguntas "
        "del corpus DNI. Durante las cuatro ejecuciones se mantuvieron constantes "
        "el corpus, el chunking, el retrieval híbrido, los embeddings de Ollama y "
        "el vector store FAISS; únicamente se cambió el LLM generativo.",
        "",
        "## Resumen de resultados",
        "",
        "| Proveedor | Modelo | Ejecución | Calidad subjetiva | Latencia LLM media (s) | Tiempo end-to-end medio (s) | Tokens/s | Source recall | Fuera de ámbito |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for run in payload["runs"]:
        summary = run["summary"]
        lines.append(
            "| {provider} | `{model}` | {ok}/{total} | {hits}/{total} | "
            "{llm:.4f} | {e2e:.4f} | {speed:.4f} | {recall:.2f} | {oos:.2f} |".format(
                provider=summary["provider"],
                model=summary["model"],
                ok=summary["questions_successful"],
                total=summary["questions_total"],
                hits=summary["subjective_hits"],
                llm=summary["avg_llm_latency_s"],
                e2e=summary["avg_end_to_end_s"],
                speed=summary["avg_tokens_per_sec"],
                recall=summary["mean_source_recall"],
                oos=summary["out_of_scope_accuracy"],
            )
        )

    lines.extend(
        [
            "",
            "## Interpretación de resultados",
            "",
            "Los cuatro modelos completaron las 12 consultas y obtuvieron el mismo "
            "recall medio de fuentes (0.95) y el mismo acierto en preguntas fuera "
            "de ámbito (1.00). Esto indica que el retrieval y la salvaguarda "
            "anti-alucinación se comportaron de forma estable durante la comparación.",
            "",
            "Sin embargo, la revisión manual sí muestra diferencias de calidad. "
            "`qwen2.5:3b`, `gemma3:27b` y `llama3.3:70b` respondieron correctamente "
            "las 12 preguntas. En cambio, `llama3.2:3b` falló en `q07` y `q08` "
            "porque interpretó como contradicciones casos en los que podía ofrecer "
            "una respuesta válida y fundamentada.",
            "",
            "Entre los modelos con 12 aciertos, `gemma3:27b` mediante PoliGPT obtuvo "
            "la menor latencia media del LLM y la mayor velocidad media de generación. "
            "Por ello, con los datos actuales, es el modelo con mejor equilibrio entre "
            "calidad observada y rendimiento. `qwen2.5:3b` constituye una alternativa "
            "local sólida, ya que también logró 12 aciertos sin depender de la VPN ni "
            "de un servicio remoto.",
            "",
            "## Incidencias cualitativas detectadas",
            "",
            "| Modelo | Pregunta | Incidencia |",
            "|---|---|---|",
            "| `llama3.2:3b` | q07 | Detecta una supuesta contradicción de ubicación y termina rechazando la respuesta, aunque dispone del CEIP Antonio Ferrandis. |",
            "| `llama3.2:3b` | q08 | Presenta como contradicción la documentación necesaria y no ofrece la respuesta definitiva esperada. |",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    payload = build_payload()

    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    OUTPUT_MD.write_text(build_markdown(payload), encoding="utf-8")

    print(f"Generado: {OUTPUT_JSON}")
    print(f"Generado: {OUTPUT_MD}")
    print("Revisión subjetiva incorporada en benchmark.json y benchmark.md")


if __name__ == "__main__":
    main()