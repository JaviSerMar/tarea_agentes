"""Evalúa las respuestas consolidadas del benchmark DNI mediante RAGAs.

El script utiliza PoliGPT como modelo juez y como proveedor de embeddings
únicamente para la evaluación. El pipeline original del agente no se modifica.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

INPUT_FILE = Path("benchmark/benchmark.json")
DEFAULT_OUTPUT_FILE = Path("evaluacion/ragas_results.json")

JUDGE_MODEL = "gemma3:27b"
EMBEDDING_MODEL = "poligpt-embed-bge-m3"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta las cuatro métricas RAGAs sobre el benchmark DNI."
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Evalúa solo una ejecución concreta, por ejemplo poligpt_gemma27b.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limita el número de preguntas por modelo para pruebas rápidas.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="Ruta del JSON de resultados.",
    )
    return parser.parse_args()


def safe_value(value: Any) -> float | None:
    """Convierte el valor de una métrica a float JSON válido."""
    if value is None:
        return None

    numeric = float(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None

    return round(numeric, 6)


def average(values: list[float | None]) -> float | None:
    valid_values = [value for value in values if value is not None]
    if not valid_values:
        return None
    return round(mean(valid_values), 6)


def build_scorers() -> dict[str, Any]:
    load_dotenv()

    base_url = os.environ["POLIGPT_BASE_URL"]
    api_key = os.environ["POLIGPT_API_KEY"]

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=180.0,
        max_retries=1,
    )

    llm = llm_factory(JUDGE_MODEL, client=client)
    embeddings = embedding_factory(
        "openai",
        model=EMBEDDING_MODEL,
        client=client,
    )

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


def score_question(
    question: dict[str, Any],
    scorers: dict[str, Any],
) -> dict[str, float | None]:
    user_input = question["pregunta"]
    response = question["respuesta_modelo"]
    reference = question["respuesta_referencia"]
    retrieved_contexts = question["retrieved_contexts"]

    faithfulness = scorers["faithfulness"].score(
        user_input=user_input,
        response=response,
        retrieved_contexts=retrieved_contexts,
    )
    answer_relevancy = scorers["answer_relevancy"].score(
        user_input=user_input,
        response=response,
    )
    context_precision = scorers["context_precision"].score(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )
    context_recall = scorers["context_recall"].score(
        user_input=user_input,
        reference=reference,
        retrieved_contexts=retrieved_contexts,
    )

    return {
        "faithfulness": safe_value(faithfulness.value),
        "answer_relevancy": safe_value(answer_relevancy.value),
        "context_precision": safe_value(context_precision.value),
        "context_recall": safe_value(context_recall.value),
    }


def evaluate_run(
    run: dict[str, Any],
    scorers: dict[str, Any],
    limit: int | None,
) -> dict[str, Any]:
    summary = run["summary"]
    questions = run["results"][:limit] if limit else run["results"]

    print(f"\n[ragas] Modelo: {summary['model']} ({summary['label']})")
    evaluated_questions = []

    for position, question in enumerate(questions, start=1):
        print(
            f"  [{position}/{len(questions)}] Evaluando "
            f"{question['id']} - {question['pregunta'][:48]}"
        )

        try:
            ragas_metrics = score_question(question, scorers)
            error = None
        except Exception as exception:
            ragas_metrics = {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_precision": None,
                "context_recall": None,
            }
            error = str(exception)
            print(f"    ERROR: {error}")

        evaluated_questions.append(
            {
                "id": question["id"],
                "pregunta": question["pregunta"],
                "calidad_subjetiva": question["calidad_subjetiva"],
                "ragas": ragas_metrics,
                "metricas_propias": {
                    "expected_source_coverage": question["source_recall"],
                    "out_of_scope_rejection_accuracy": question["rejection_ok"],
                },
                "error": error,
            }
        )

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    ragas_summary = {
        metric_name: average(
            [question["ragas"][metric_name] for question in evaluated_questions]
        )
        for metric_name in metric_names
    }

    own_metrics_summary = {
        "expected_source_coverage": average(
            [
                question["metricas_propias"]["expected_source_coverage"]
                for question in evaluated_questions
            ]
        ),
        "out_of_scope_rejection_accuracy": average(
            [
                question["metricas_propias"]["out_of_scope_rejection_accuracy"]
                for question in evaluated_questions
            ]
        ),
    }

    return {
        "label": summary["label"],
        "provider": summary["provider"],
        "model": summary["model"],
        "questions_evaluated": len(evaluated_questions),
        "subjective_hits": summary["subjective_hits"],
        "subjective_fails": summary["subjective_fails"],
        "ragas_summary": ragas_summary,
        "metricas_propias_summary": own_metrics_summary,
        "results": evaluated_questions,
    }


def main() -> None:
    args = parse_args()

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    runs = benchmark["runs"]
    if args.label:
        runs = [run for run in runs if run["summary"]["label"] == args.label]
        if not runs:
            raise ValueError(f"No existe ningún modelo con label: {args.label}")

    scorers = build_scorers()
    evaluated_runs = [
        evaluate_run(run, scorers, args.limit)
        for run in runs
    ]

    payload = {
        "evaluation": "RAGAs y métricas propias para Agente RAG DNI",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "judge": {
            "llm": JUDGE_MODEL,
            "embeddings": EMBEDDING_MODEL,
            "provider": "PoliGPT",
            "note": (
                "Estos modelos se usan solo como evaluadores; "
                "el pipeline RAG evaluado conserva FAISS y embeddings Ollama."
            ),
        },
        "ragas_metrics": {
            "faithfulness": "Fidelidad de la respuesta al contexto recuperado.",
            "answer_relevancy": "Relevancia de la respuesta respecto a la pregunta.",
            "context_precision": "Utilidad de los chunks recuperados para responder.",
            "context_recall": "Cobertura de la información necesaria en los chunks.",
        },
        "metricas_propias": {
            "expected_source_coverage": (
                "Proporción de archivos fuente esperados que aparecen entre "
                "las fuentes recuperadas por el agente."
            ),
            "out_of_scope_rejection_accuracy": (
                "Proporción de preguntas fuera del ámbito DNI rechazadas "
                "correctamente sin inventar información."
            ),
        },
        "runs": evaluated_runs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[ragas] Resultados guardados en: {args.output}")


if __name__ == "__main__":
    main()