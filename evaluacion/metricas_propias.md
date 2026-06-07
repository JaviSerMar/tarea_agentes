# Evaluación RAGAs y métricas propias — Agente RAG DNI

## Contexto de la evaluación

Se han evaluado los cuatro modelos empleados en el benchmark del agente DNI sobre el mismo conjunto de 12 preguntas. El pipeline evaluado conserva el mismo corpus, chunking, retrieval híbrido, embeddings Ollama y vector store FAISS; únicamente cambia el LLM generativo.

Para calcular las métricas RAGAs se ha utilizado PoliGPT exclusivamente como evaluador:

- LLM juez: `gemma3:27b`.
- Modelo de embeddings para evaluación de relevancia: `poligpt-embed-bge-m3`.

Estos modelos de evaluación no sustituyen los componentes del agente evaluado.

## Métricas RAGAs

Las cuatro métricas solicitadas son:

- **Faithfulness**: mide hasta qué punto la respuesta se mantiene fiel a los chunks recuperados, penalizando posibles invenciones.
- **Answer relevancy**: mide si la respuesta se ajusta a la pregunta formulada, sin desviarse innecesariamente.
- **Context precision**: mide si los chunks recuperados son útiles para elaborar la respuesta.
- **Context recall**: mide si los chunks recuperados contienen la información necesaria respecto a la respuesta de referencia.

## Métricas propias

### 1. Expected Source Coverage

**Definición:** proporción de archivos fuente esperados que aparecen entre las fuentes recuperadas por el agente.

**Justificación:** en el dominio DNI no basta con redactar una respuesta plausible; la información debe proceder de los documentos correctos. Esta métrica permite detectar si el retrieval encuentra los archivos que realmente contienen la respuesta, especialmente en preguntas sobre horarios, ubicaciones o contradicciones entre fuentes.

**Interpretación:** un valor próximo a `1.00` indica que el agente recupera casi siempre las fuentes esperadas.

### 2. Out-of-Scope Rejection Accuracy

**Definición:** proporción de preguntas fuera del ámbito DNI que el agente rechaza correctamente mediante la respuesta anti-alucinación.

**Justificación:** uno de los requisitos fundamentales de la práctica es que el agente no invente información no presente en el corpus. Esta métrica evalúa específicamente el comportamiento ante preguntas ajenas a DNI, como alquileres o becas universitarias.

**Interpretación:** un valor de `1.00` indica que todas las preguntas fuera de ámbito del benchmark fueron rechazadas correctamente.

## Resultados globales

| Modelo | Faithfulness | Answer relevancy | Context precision | Context recall | Expected Source Coverage | Out-of-Scope Rejection Accuracy | Calidad manual |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qwen2.5:3b` | 0.788889 | 0.638091 | 0.669444 | 0.833333 | 0.95 | 1.00 | 12/12 |
| `llama3.2:3b` | 0.716667 | 0.471928 | 0.669444 | 0.833333 | 0.95 | 1.00 | 10/12 |
| `gemma3:27b` | 0.716667 | 0.612921 | 0.669444 | 0.833333 | 0.95 | 1.00 | 12/12 |
| `llama3.3:70b` | 0.775463 | 0.582995 | 0.669444 | 0.833333 | 0.95 | 1.00 | 12/12 |

## Interpretación

Las métricas relacionadas directamente con la recuperación son iguales en los cuatro modelos: `context_precision` alcanza `0.669444`, `context_recall` alcanza `0.833333` y `expected_source_coverage` alcanza `0.95`. Este resultado es coherente con el diseño experimental, ya que los cuatro modelos utilizaron exactamente el mismo sistema de retrieval y solo se sustituyó el LLM generativo.

La métrica `out_of_scope_rejection_accuracy` alcanza `1.00` en todos los casos. Por tanto, la salvaguarda anti-alucinación funcionó correctamente ante las preguntas fuera de ámbito incluidas en el benchmark.

Las diferencias relevantes aparecen en la calidad de la respuesta generada. `qwen2.5:3b` obtiene los mejores resultados de `faithfulness` (`0.788889`) y `answer_relevancy` (`0.638091`), además de lograr `12/12` aciertos en la revisión manual. `gemma3:27b` también logra `12/12` en la revisión manual y fue el modelo más rápido del benchmark, pero sus valores RAGAs de fidelidad y relevancia son inferiores a los de Qwen. `llama3.2:3b` es el modelo más débil: además de sus valores más bajos de relevancia, falló manualmente en dos preguntas al sobredetectar contradicciones.

## Modelo seleccionado

A la vista del conjunto de evidencias, se selecciona **`qwen2.5:3b`** como modelo recomendado para el agente final. La elección se fundamenta en que:

- obtuvo `12/12` aciertos en la revisión manual;
- consiguió el mejor valor de `faithfulness`;
- consiguió el mejor valor de `answer_relevancy`;
- funciona en local mediante Ollama, sin depender de VPN ni de disponibilidad de un servicio externo.

`gemma3:27b` queda identificado como la mejor alternativa remota cuando se prioriza la velocidad de respuesta, pero no como el modelo final elegido por calidad global.