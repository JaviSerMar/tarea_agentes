# Arquitectura del Agente RAG DNI

## 1. Objetivo de la arquitectura

El sistema implementa un agente RAG para responder preguntas sobre la asociación DNI (Damos Nuestra Ilusión) utilizando únicamente el corpus oficial proporcionado.

La solución se ha diseñado con **arquitectura hexagonal** o **ports & adapters**, con el objetivo de separar la lógica del agente de las tecnologías externas utilizadas para generar respuestas, producir embeddings o almacenar vectores.

Esta separación permite:

- cambiar de Ollama a PoliGPT sin modificar el dominio;
- cambiar entre FAISS y ChromaDB mediante configuración;
- intercambiar el proveedor de embeddings;
- probar la lógica principal mediante dobles de prueba sin depender de red ni de modelos reales.

## 2. Flujo general de una consulta

El flujo real de ejecución es:

```text
consultar.py
    ↓
composition.py
    ↓
ChatbotService
    ↓
RetrieverPort + LLMPort
    ↓
Adapters configurados
```

El proceso completo es:

1. El usuario formula una pregunta mediante `consultar.py`.
2. El composition root construye los adapters definidos en la configuración.
3. El retriever recupera chunks relevantes del corpus DNI.
4. El servicio de dominio analiza el contexto recuperado.
5. Si la pregunta está fuera del ámbito del corpus, el sistema devuelve el rechazo anti-alucinación.
6. Si existe una contradicción relevante entre fuentes, se presentan ambas versiones citadas.
7. En los demás casos, se construye un prompt restringido al contexto y se invoca el LLM.
8. Se devuelve la respuesta junto con fuentes, chunks y métricas.

## 3. Capas de la solución

### 3.1. Entrada

El punto de entrada obligatorio para el corrector es:

```text
consultar.py
```

Este fichero expone la función:

```python
def consultar(pregunta: str, conversation_id: str | None = None) -> dict:
    ...
```

`consultar.py` no contiene la lógica principal del agente: actúa como adapter de entrada y delega en la configuración y el servicio de dominio.

### 3.2. Dominio

El dominio contiene la lógica independiente de infraestructura:

```text
src/agente_rag/domain/
├── entities.py
├── ports.py
└── chatbot_service.py
```

Responsabilidades principales:

- representar preguntas, respuestas y chunks;
- definir los contratos que deben cumplir los adapters;
- orquestar retrieval, prompt, respuesta y fuentes;
- mantener la lógica principal desacoplada de Ollama, PoliGPT, FAISS o ChromaDB.

### 3.3. Ports

Los ports definen las interfaces que el dominio necesita:

- `LLMPort`: generación de respuestas mediante un modelo de lenguaje.
- `EmbedderPort`: conversión de texto a embeddings.
- `RetrieverPort`: recuperación de chunks relevantes.
- Port de vector store: almacenamiento y búsqueda vectorial.

El dominio depende de estos contratos, no de implementaciones concretas.

### 3.4. Adapters

Los adapters implementan las tecnologías externas:

```text
src/agente_rag/adapters/
├── llm/
│   ├── ollama_llm.py
│   └── poligpt_llm.py
├── embeddings/
│   ├── ollama_embeddings.py
│   └── sentence_transformers_embeddings.py
└── retriever/
    ├── semantic_retriever.py
    ├── bm25_retriever.py
    ├── hybrid_retriever.py
    ├── chroma_vector_store.py
    └── faiss_vector_store.py
```

Adapters disponibles:

| Tipo | Implementaciones |
|------|------------------|
| LLM | Ollama, PoliGPT |
| Embeddings | Ollama, Sentence Transformers |
| Vector store | FAISS, ChromaDB |
| Retrieval | Semántico, BM25, híbrido |

### 3.5. Composition root y configuración

La selección de adapters se realiza en:

```text
src/agente_rag/composition.py
src/agente_rag/config.py
```

La configuración se controla mediante variables de entorno. La configuración final recomendada es:

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b
EMBEDDER_PROVIDER=ollama
EMBED_MODEL=nomic-embed-text
VECTOR_STORE_PROVIDER=faiss
FAISS_PATH=./data/dni.index
```

Cambiar un adapter no requiere modificar el dominio, sino únicamente seleccionar otra configuración compatible.

## 4. Recuperación de información

### 4.1. Chunking adaptado al corpus DNI

El corpus contiene documentos narrativos y documentos estructurados mediante pares `Q:/A:`. Para evitar pérdida de información, el agente conserva unidos esos pares cuando realiza el chunking.

Esta decisión mejora especialmente preguntas frecuentes como:

- qué es DNI;
- horarios;
- documentación necesaria;
- ubicaciones de actividades.

### 4.2. Retrieval híbrido

El agente combina dos estrategias de recuperación:

- **Búsqueda semántica**, basada en embeddings.
- **BM25**, basada en coincidencia léxica.

El retrieval híbrido mejora la robustez ante preguntas con términos exactos, nombres de ubicaciones o formulaciones similares a las preguntas frecuentes del corpus.

### 4.3. Vector store final

La configuración final utiliza **FAISS** como vector store persistente:

```env
VECTOR_STORE_PROVIDER=faiss
FAISS_PATH=./data/dni.index
```

El índice se genera mediante:

```powershell
python scripts\build_hexagonal_index.py
```

Sobre el corpus oficial DNI se generaron 281 chunks.

ChromaDB permanece implementado como alternativa intercambiable.

## 5. Control de alucinaciones y contradicciones

### 5.1. Preguntas fuera de ámbito

Si el agente no encuentra información suficiente en sus fuentes, devuelve:

```text
No tengo esa información en mis fuentes.
```

Este comportamiento se ha validado mediante preguntas sobre alquileres y becas universitarias, ajenas al corpus DNI.

### 5.2. Contradicciones reales del corpus

El corpus contiene información contradictoria que no debe corregirse artificialmente. Por ejemplo, respecto al horario de desayunos solidarios:

- `01_faq_dni.txt` indica las 8:00.
- `11_horarios_ubicaciones.txt` indica normalmente entre las 9:00 y las 12:00.

El agente muestra ambas versiones citando sus fuentes, evitando inventar una única respuesta definitiva.

## 6. Benchmark y modelo seleccionado

Se evaluaron cuatro modelos manteniendo constante todo el pipeline salvo el LLM generativo:

| Proveedor | Modelo |
|---|---|
| Ollama local | `qwen2.5:3b` |
| Ollama local | `llama3.2:3b` |
| PoliGPT | `gemma3:27b` |
| PoliGPT | `llama3.3:70b` |

Los resultados completos están en:

```text
benchmark/benchmark.json
benchmark/benchmark.md
```

Tras la revisión manual y la evaluación RAGAs, se selecciona:

```text
qwen2.5:3b mediante Ollama local
```

Motivos:

- obtuvo 12/12 aciertos manuales;
- alcanzó el mejor valor de `faithfulness`;
- alcanzó el mejor valor de `answer_relevancy`;
- funciona localmente sin depender de VPN ni de un servicio externo.

## 7. Evaluación RAGAs y métricas propias

La evaluación incluye las cuatro métricas requeridas:

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

Además, se definieron dos métricas propias:

- `expected_source_coverage`
- `out_of_scope_rejection_accuracy`

Los resultados se almacenan en:

```text
evaluacion/ragas_results.json
evaluacion/metricas_propias.md
```

## 8. Tests y verificabilidad

El proyecto dispone de 54 tests correctos:

```powershell
python -m pytest -q
```

Los tests del dominio utilizan adapters simulados o mocks, de modo que validan la lógica principal sin requerir red, VPN ni modelos remotos.

Además, se han realizado validaciones reales con:

- Ollama local;
- PoliGPT mediante VPN UPV;
- FAISS;
- RAGAs como sistema de evaluación.

## 9. Añadir un adapter nuevo

Para incorporar un nuevo proveedor sin modificar el dominio:

1. Crear una implementación compatible con el port correspondiente dentro de `src/agente_rag/adapters/`.
2. Añadir la selección del nuevo adapter en `composition.py`.
3. Añadir sus variables de configuración en `config.py` y `.env.example`.
4. Crear tests unitarios del adapter y comprobar que el dominio sigue funcionando sin cambios.

Por ejemplo, un nuevo vector store podría añadirse implementando el mismo contrato que cumplen FAISS y ChromaDB.

## 10. Limitaciones y mejoras futuras

Limitaciones actuales:

- La evaluación RAGAs depende de PoliGPT y VPN UPV cuando se ejecuta fuera del campus.
- La detección de contradicciones puede seguir mejorándose para distinguir contradicciones reales de información complementaria.
- El agente no mantiene memoria conversacional entre consultas.
- No se ha implementado interfaz gráfica, ya que no forma parte de las bandas declaradas.

Mejoras futuras posibles:

- añadir memoria conversacional controlada;
- mejorar el reranking de chunks;
- incluir una interfaz web;
- ampliar el benchmark con más preguntas ambiguas y multi-documento.