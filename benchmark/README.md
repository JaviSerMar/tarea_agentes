# Benchmark del Agente RAG DNI

Este directorio contiene la evaluación comparativa del agente RAG sobre el corpus oficial de DNI (Damos Nuestra Ilusión).

## Objetivo

El benchmark compara cuatro modelos generativos manteniendo constante el resto del pipeline:

* Corpus DNI de 16 documentos.
* Chunking adaptado a documentos narrativos y pares `Q:/A:`.
* Retrieval híbrido semántico + BM25.
* Embeddings locales mediante Ollama (`nomic-embed-text`).
* Vector store FAISS.

De esta forma, la comparación mide el efecto del LLM generativo y no cambios en el sistema de recuperación.

## Modelos evaluados

| Proveedor    | Modelo         |
| ------------ | -------------- |
| Ollama local | `qwen2.5:3b`   |
| Ollama local | `llama3.2:3b`  |
| PoliGPT      | `gemma3:27b`   |
| PoliGPT      | `llama3.3:70b` |

Los modelos PoliGPT requieren conexión a la VPN UPV cuando se trabaja fuera del campus. Antes de ejecutar las pruebas se consultó el catálogo disponible de modelos, ya que puede cambiar con el tiempo.

## Preguntas evaluadas

El fichero `preguntas.json` contiene 12 preguntas:

* Preguntas factuales directas sobre DNI y sus proyectos.
* Preguntas logísticas sobre horarios, ubicaciones y documentación.
* Un caso de contradicción real del corpus: el horario de los desayunos solidarios.
* Dos preguntas fuera de ámbito, que deben rechazarse sin inventar información.

## Ejecutar una evaluación de benchmark

Con el entorno virtual activo y la configuración deseada en `.env`, se puede ejecutar un modelo local:

```powershell
python scripts\run_eval.py --provider ollama --model qwen2.5:3b --label local_qwen
```

Para ejecutar un modelo PoliGPT es necesario configurar de forma privada `POLIGPT_BASE_URL` y `POLIGPT_API_KEY` en `.env`, además de mantener activa la VPN UPV cuando se trabaja fuera del campus:

```powershell
python scripts\run_eval.py --provider poligpt --model gemma3:27b --label poligpt_gemma27b
```

Las ejecuciones crudas se guardan localmente en `benchmark/runs/`. Esta carpeta no se versiona, ya que los entregables consolidados se generan mediante:

```powershell
python scripts\build_benchmark_report.py
```

## Ficheros generados

* `benchmark/benchmark.json`: resultados estructurados de los cuatro modelos, respuestas, fuentes, chunks recuperados y valoración manual por pregunta.
* `benchmark/benchmark.md`: tabla legible del benchmark, resultados RAGAs, métricas propias e interpretación final.
* `evaluacion/ragas_results.json`: resultados detallados de las cuatro métricas RAGAs para los cuatro modelos.
* `evaluacion/metricas_propias.md`: definición, justificación e interpretación de las dos métricas propias.

## Métricas del benchmark base

Para cada respuesta se registran:

* Tokens de entrada y salida.
* Latencia del LLM proporcionada por el adapter.
* Tiempo total end-to-end medido por el script.
* Tokens por segundo.
* Fuentes recuperadas y cobertura de fuentes esperadas.
* Acierto de rechazo en preguntas fuera de ámbito.
* Calidad subjetiva manual (`ACIERTO` o `FALLO`).

## Evaluación RAGAs

La evaluación RAGAs se ejecuta mediante:

```powershell
python scripts\run_ragas_eval.py --output evaluacion\ragas_results.json
```

Para esta evaluación se utilizó PoliGPT exclusivamente como juez:

* LLM juez: `gemma3:27b`.
* Embeddings para evaluación de relevancia: `poligpt-embed-bge-m3`.

Estos componentes se usan únicamente para puntuar las respuestas y no modifican el pipeline RAG original del agente, que continúa utilizando FAISS y embeddings Ollama.

Las cuatro métricas RAGAs calculadas son:

* `faithfulness`: fidelidad de la respuesta al contexto recuperado.
* `answer_relevancy`: relevancia de la respuesta respecto a la pregunta.
* `context_precision`: utilidad de los chunks recuperados.
* `context_recall`: cobertura de la información necesaria en los chunks.

## Métricas propias

Se han definido dos métricas propias, pertinentes para el agente DNI:

* `expected_source_coverage`: proporción de archivos fuente esperados que aparecen entre las fuentes recuperadas.
* `out_of_scope_rejection_accuracy`: proporción de preguntas fuera del ámbito DNI que se rechazan correctamente sin inventar información.

## Resultados globales

| Modelo         | Calidad manual | Faithfulness | Answer relevancy | Context precision | Context recall | Expected Source Coverage | Out-of-Scope Rejection Accuracy |
| -------------- | -------------: | -----------: | ---------------: | ----------------: | -------------: | -----------------------: | ------------------------------: |
| `qwen2.5:3b`   |          12/12 |     0.788889 |         0.638091 |          0.669444 |       0.833333 |                     0.95 |                            1.00 |
| `llama3.2:3b`  |          10/12 |     0.716667 |         0.471928 |          0.669444 |       0.833333 |                     0.95 |                            1.00 |
| `gemma3:27b`   |          12/12 |     0.716667 |         0.612921 |          0.669444 |       0.833333 |                     0.95 |                            1.00 |
| `llama3.3:70b` |          12/12 |     0.775463 |         0.582995 |          0.669444 |       0.833333 |                     0.95 |                            1.00 |

## Interpretación de resultados

Las métricas directamente relacionadas con la recuperación son iguales en los cuatro modelos: `context_precision`, `context_recall` y `expected_source_coverage` mantienen los mismos valores. Este resultado es coherente, porque se conservó el mismo sistema de retrieval, los mismos embeddings y el mismo vector store en todas las ejecuciones.

La salvaguarda anti-alucinación también funcionó correctamente: todos los modelos alcanzaron `1.00` en `out_of_scope_rejection_accuracy`.

Las diferencias aparecen en la generación final de la respuesta. `llama3.2:3b` falló manualmente en dos preguntas (`q07` y `q08`) al sobredetectar contradicciones. `qwen2.5:3b`, `gemma3:27b` y `llama3.3:70b` obtuvieron 12 aciertos sobre 12.

Entre los modelos sin fallos manuales, `qwen2.5:3b` obtuvo los mejores valores RAGAs de `faithfulness` y `answer_relevancy`. `gemma3:27b` fue el modelo más rápido, pero sus métricas de calidad generativa quedaron por debajo de las de Qwen.

## Modelo seleccionado

Se selecciona **`qwen2.5:3b` mediante Ollama local** como modelo final recomendado para el agente.

La elección se basa en cuatro razones:

* Logra `12/12` aciertos en la revisión manual.
* Obtiene el mejor valor de `faithfulness`.
* Obtiene el mejor valor de `answer_relevancy`.
* Funciona en local sin depender de VPN ni de disponibilidad de un servicio remoto.

`gemma3:27b` queda como alternativa remota recomendable cuando se prioriza la velocidad de respuesta.
