"""Punto de entrada del contrato (Opción A — módulo Python).

El corrector importa esta función con la signatura EXACTA que define el
enunciado §9. No la cambies de sitio ni le añadas argumentos posicionales:
si rompe el contrato, la nota es 0 automáticamente.

Uso manual desde CLI:
    python consultar.py "¿Hay una asignatura sobre videojuegos en GTI?"
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from agente_rag.composition import build_chatbot_service  # noqa: E402
from agente_rag.domain.entities import Question  # noqa: E402

def consultar(pregunta: str, conversation_id: str | None = None) -> dict:
    """Función obligatoria del contrato, conectada al dominio hexagonal."""
    service = build_chatbot_service()
    answer = service.answer(
        Question(text=pregunta, conversation_id=conversation_id)
    )

    metrics = answer.metrics

    return {
        "respuesta": answer.text,
        "fuentes": answer.sources,
        "chunks": [
            {
                "source": chunk.source,
                "text": chunk.text,
                "score": chunk.score,
            }
            for chunk in answer.chunks
        ],
        "metricas": {
            "prompt_tokens": metrics.prompt_tokens,
            "output_tokens": metrics.output_tokens,
            "tokens_per_sec": metrics.tokens_per_sec,
            "latencia_s": metrics.latency_s,
            "modelo": metrics.model,
        }
        if metrics is not None
        else None,
        "trazas": answer.traces,
        "conversation_id": answer.conversation_id,
    }


def _main(argv: list[str]) -> int:
    if len(argv) < 2:
        print('Uso: python consultar.py "<pregunta>"', file=sys.stderr)
        return 2
    pregunta = " ".join(argv[1:])
    result = consultar(pregunta)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
