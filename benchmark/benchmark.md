# Benchmark DNI — comparación de cuatro modelos

## Condiciones de comparación

Se han evaluado cuatro modelos sobre el mismo conjunto fijo de 12 preguntas del corpus DNI. Durante las cuatro ejecuciones se mantuvieron constantes el corpus, el chunking, el retrieval híbrido, los embeddings de Ollama y el vector store FAISS; únicamente se cambió el LLM generativo.

## Resumen de resultados

| Proveedor | Modelo | Ejecución | Calidad subjetiva | Latencia LLM media (s) | Tiempo end-to-end medio (s) | Tokens/s | Source recall | Fuera de ámbito |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Ollama local | `qwen2.5:3b` | 12/12 | 12/12 | 5.1858 | 7.8483 | 60.4633 | 0.95 | 1.00 |
| Ollama local | `llama3.2:3b` | 12/12 | 10/12 | 7.9933 | 10.3742 | 58.6225 | 0.95 | 1.00 |
| PoliGPT | `gemma3:27b` | 12/12 | 12/12 | 2.2067 | 4.7592 | 282.5058 | 0.95 | 1.00 |
| PoliGPT | `llama3.3:70b` | 12/12 | 12/12 | 6.0333 | 8.5075 | 130.4167 | 0.95 | 1.00 |

## Interpretación de resultados

Los cuatro modelos completaron las 12 consultas y obtuvieron el mismo recall medio de fuentes (0.95) y el mismo acierto en preguntas fuera de ámbito (1.00). Esto indica que el retrieval y la salvaguarda anti-alucinación se comportaron de forma estable durante la comparación.

Sin embargo, la revisión manual sí muestra diferencias de calidad. `qwen2.5:3b`, `gemma3:27b` y `llama3.3:70b` respondieron correctamente las 12 preguntas. En cambio, `llama3.2:3b` falló en `q07` y `q08` porque interpretó como contradicciones casos en los que podía ofrecer una respuesta válida y fundamentada.

Entre los modelos con 12 aciertos, `gemma3:27b` mediante PoliGPT obtuvo la menor latencia media del LLM y la mayor velocidad media de generación. Por ello, con los datos actuales, es el modelo con mejor equilibrio entre calidad observada y rendimiento. `qwen2.5:3b` constituye una alternativa local sólida, ya que también logró 12 aciertos sin depender de la VPN ni de un servicio remoto.

## Incidencias cualitativas detectadas

| Modelo | Pregunta | Incidencia |
|---|---|---|
| `llama3.2:3b` | q07 | Detecta una supuesta contradicción de ubicación y termina rechazando la respuesta, aunque dispone del CEIP Antonio Ferrandis. |
| `llama3.2:3b` | q08 | Presenta como contradicción la documentación necesaria y no ofrece la respuesta definitiva esperada. |
