# Benchmark DNI — comparación de cuatro modelos

## Condiciones de comparación

Se han evaluado cuatro modelos sobre el mismo conjunto fijo de 12 preguntas del corpus DNI. Durante las cuatro ejecuciones se mantuvieron constantes el corpus, el chunking, el retrieval híbrido, los embeddings de Ollama y el vector store FAISS; únicamente se cambió el LLM generativo.

## Resultados del benchmark base

| Proveedor | Modelo | Ejecución | Calidad subjetiva | Latencia LLM media (s) | Tiempo end-to-end medio (s) | Tokens/s | Source recall | Fuera de ámbito |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ollama local | `qwen2.5:3b` | 12/12 | 12/12 | 5.1858 | 7.8483 | 60.4633 | 0.95 | 1.00 |
| Ollama local | `llama3.2:3b` | 12/12 | 10/12 | 7.9933 | 10.3742 | 58.6225 | 0.95 | 1.00 |
| PoliGPT | `gemma3:27b` | 12/12 | 12/12 | 2.2067 | 4.7592 | 282.5058 | 0.95 | 1.00 |
| PoliGPT | `llama3.3:70b` | 12/12 | 12/12 | 6.0333 | 8.5075 | 130.4167 | 0.95 | 1.00 |

La revisión manual detectó que `llama3.2:3b` falló en `q07` y `q08` por interpretar como contradicciones casos en los que podía ofrecer una respuesta válida. Los otros tres modelos respondieron correctamente las 12 preguntas.

## Resultados RAGAs y métricas propias

La evaluación RAGAs se realizó usando `gemma3:27b` de PoliGPT como modelo juez y `poligpt-embed-bge-m3` únicamente como embedding de evaluación. Estos componentes no modifican el pipeline original del agente.

| Modelo | Faithfulness | Answer relevancy | Context precision | Context recall | Expected Source Coverage | Out-of-Scope Rejection Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `qwen2.5:3b` | 0.788889 | 0.638091 | 0.669444 | 0.833333 | 0.95 | 1.00 |
| `llama3.2:3b` | 0.716667 | 0.471928 | 0.669444 | 0.833333 | 0.95 | 1.00 |
| `gemma3:27b` | 0.716667 | 0.612921 | 0.669444 | 0.833333 | 0.95 | 1.00 |
| `llama3.3:70b` | 0.775463 | 0.582995 | 0.669444 | 0.833333 | 0.95 | 1.00 |

Las métricas relacionadas con la recuperación se mantienen constantes entre modelos: todos utilizan el mismo retrieval, los mismos embeddings y el mismo vector store. También todos rechazan correctamente las preguntas fuera de ámbito.

La diferencia aparece en la generación de la respuesta. `qwen2.5:3b` obtiene el mejor resultado de `faithfulness` y de `answer_relevancy`, además de haber conseguido 12/12 aciertos en la revisión manual. `gemma3:27b` también alcanza 12/12 y es el más rápido, pero sus resultados RAGAs de calidad son inferiores a los de Qwen.

## Modelo seleccionado

Se selecciona **`qwen2.5:3b` mediante Ollama local** como modelo final recomendado. La elección se basa en sus 12/12 aciertos manuales, sus mejores valores RAGAs de fidelidad y relevancia, y su funcionamiento local sin depender de VPN ni de disponibilidad de un servicio externo.

`gemma3:27b` queda como alternativa remota especialmente interesante cuando se prioriza la velocidad de respuesta.

## Incidencias cualitativas detectadas

| Modelo | Pregunta | Incidencia |
|---|---|---|
| `llama3.2:3b` | q07 | Detecta una supuesta contradicción de ubicación y termina rechazando la respuesta, aunque dispone del CEIP Antonio Ferrandis. |
| `llama3.2:3b` | q08 | Presenta como contradicción la documentación necesaria y no ofrece la respuesta definitiva esperada. |
