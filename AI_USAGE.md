# AI USAGE.md — Uso de asistentes de IA

## Herramienta utilizada

* **ChatGPT**: utilizado como asistente conversacional durante el desarrollo de la práctica.

No se han utilizado GitHub Copilot, Cursor, Claude, Gemini ni otros asistentes de programación para construir la entrega.

## Tareas en las que se ha utilizado ChatGPT

ChatGPT se ha utilizado como apoyo para comprender la práctica, planificar el desarrollo, proponer fragmentos de código, e interpretar algunos errores. Los estudiantes han trabajado localmente en Visual Studio Code y PowerShell, han ejecutado todas las pruebas y comandos, han comprobado los resultados y han revisado las decisiones incorporadas al repositorio.

### Adaptación inicial al corpus DNI y mejora del RAG

ChatGPT ayudó a orientar la sustitución del ejemplo inicial por el corpus oficial DNI y a plantear mejoras de calidad del agente:

* tratamiento específico de documentos con pares `Q:/A:`;
* retrieval híbrido semántico + BM25;
* prioridad de coincidencias FAQ exactas;
* gestión de contradicciones reales del corpus;
* rechazo de preguntas fuera de ámbito para evitar alucinaciones.

Ficheros afectados durante esta fase:

* `consultar.py`
* módulos iniciales de `src/agente_rag/`
* tests relacionados con recuperación y respuesta del agente.

### Arquitectura hexagonal

ChatGPT ayudó a diseñar y revisar la separación entre dominio, ports, adapters y composition root, con el objetivo de poder intercambiar modelos, embeddings y vector stores sin modificar la lógica principal del agente.

Ficheros desarrollados o modificados con apoyo de ChatGPT:

* `src/agente_rag/domain/entities.py`
* `src/agente_rag/domain/ports.py`
* `src/agente_rag/domain/chatbot_service.py`
* `src/agente_rag/composition.py`
* `src/agente_rag/adapters/llm/ollama_llm.py`
* `src/agente_rag/adapters/llm/poligpt_llm.py`
* `src/agente_rag/adapters/embeddings/ollama_embeddings.py`
* `src/agente_rag/adapters/embeddings/sentence_transformers_embeddings.py`
* `src/agente_rag/adapters/retriever/bm25_retriever.py`
* `src/agente_rag/adapters/retriever/semantic_retriever.py`
* `src/agente_rag/adapters/retriever/hybrid_retriever.py`
* `src/agente_rag/adapters/retriever/chroma_vector_store.py`
* `src/agente_rag/adapters/retriever/faiss_vector_store.py`
* `scripts/build_hexagonal_index.py`
* tests asociados a adapters, dominio e integración.

### Benchmark con cuatro modelos

ChatGPT ayudó a definir un benchmark reproducible sobre el corpus DNI, a ejecutar los cuatro modelos requeridos y a interpretar los resultados.

Ficheros desarrollados o modificados con apoyo de ChatGPT:

* `benchmark/preguntas.json`
* `scripts/run_eval.py`
* `scripts/build_benchmark_report.py`
* `benchmark/README.md`
* `benchmark/benchmark.json`
* `benchmark/benchmark.md`

Modelos comparados:

* Ollama local: `qwen2.5:3b`
* Ollama local: `llama3.2:3b`
* PoliGPT: `gemma3:27b`
* PoliGPT: `llama3.3:70b`

ChatGPT también ayudó a revisar manualmente las respuestas y a identificar que `llama3.2:3b` fallaba en dos preguntas por sobredetectar contradicciones.

### Evaluación RAGAs y métricas propias

ChatGPT ayudó a instalar y compatibilizar RAGAs, configurar PoliGPT como juez de evaluación, definir dos métricas propias y generar la documentación de resultados.

Ficheros desarrollados o modificados con apoyo de ChatGPT:

* `requirements.txt`
* `scripts/run_ragas_eval.py`
* `evaluacion/ragas_results.json`
* `evaluacion/metricas_propias.md`
* `benchmark/benchmark.json`
* `benchmark/benchmark.md`
* `benchmark/README.md`

Métricas RAGAs utilizadas:

* `faithfulness`
* `answer_relevancy`
* `context_precision`
* `context_recall`

Métricas propias definidas:

* `expected_source_coverage`
* `out_of_scope_rejection_accuracy`

### Configuración y documentación final

ChatGPT ayudó a revisar la configuración declarativa y la documentación necesaria para la entrega.

Ficheros modificados o revisados con apoyo de ChatGPT:

* `.env.example`
* `features.json`
* `AI USAGE.md`
* documentación final, README principal y documentación técnica.

## Grado de revisión humana

Todo el código y la documentación propuestos mediante ChatGPT han sido copiados, adaptados, ejecutados y revisados localmente por los estudiantes antes de incorporarlos al repositorio.

Los estudiantes han:

* ejecutado los tests automáticos del proyecto;
* comprobado manualmente respuestas del agente;
* realizado y revisado el benchmark;
* ejecutado la evaluación RAGAs;
* verificado los resultados antes de documentarlos;
* realizado los commits de Git de forma manual.

No se incluye código generado por IA que se entregue sin revisión posterior.

## Uso de modelos dentro del sistema

Además de ChatGPT como ayuda al desarrollo, la práctica utiliza modelos LLM y de embeddings como parte funcional del agente o de su evaluación:

* Ollama local para generación y embeddings del agente.
* PoliGPT para modelos alternativos del benchmark.
* PoliGPT como juez LLM y proveedor de embeddings durante la evaluación RAGAs.

Estos modelos forman parte de la implementación y evaluación del sistema, no son herramientas utilizadas para redactar o programar la entrega.

## Compromiso

Los estudiantes declaran haber leído y comprendido el código y la documentación incluidos en la entrega y estar en condiciones de explicar las decisiones principales del sistema durante la defensa oral.

Firma: ______________________________
Javier Serrano Marco
Javier Camarena Cuartero
Jaime Ferrer Prats