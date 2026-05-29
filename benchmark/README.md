# Benchmark del Agente RAG DNI

Este directorio contiene la evaluación comparativa del agente RAG sobre el corpus oficial de DNI (Damos Nuestra Ilusión).

## Objetivo

El benchmark permite comparar cuatro modelos generativos manteniendo constante el resto del pipeline:

- Corpus DNI de 16 documentos.
- Chunking adaptado a documentos narrativos y pares `Q:/A:`.
- Retrieval híbrido semántico + BM25.
- Embeddings locales mediante Ollama (`nomic-embed-text`).
- Vector store FAISS.

De esta forma, la comparación mide el efecto del LLM generativo y no cambios en el sistema de recuperación.

## Modelos evaluados

| Proveedor | Modelo |
|---|---|
| Ollama local | `qwen2.5:3b` |
| Ollama local | `llama3.2:3b` |
| PoliGPT | `gemma3:27b` |
| PoliGPT | `llama3.3:70b` |

Los modelos PoliGPT requieren conexión a la VPN UPV cuando se trabaja fuera del campus. El catálogo disponible se consultó antes de ejecutar las pruebas, ya que puede cambiar con el tiempo.

## Preguntas

El fichero `preguntas.json` contiene 12 preguntas:

- Preguntas factuales directas sobre DNI y sus proyectos.
- Preguntas logísticas sobre horarios, ubicaciones y documentación.
- Un caso de contradicción real del corpus: horario de desayunos solidarios.
- Dos preguntas fuera de ámbito, que deben rechazarse sin inventar información.

## Ejecutar una evaluación

Con el entorno virtual activo y la configuración deseada en `.env`, se puede ejecutar un modelo local:

```powershell
python scripts\run_eval.py --provider ollama --model qwen2.5:3b --label local_qwen