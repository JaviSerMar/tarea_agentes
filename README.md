# Agente RAG DNI — Asistente Inteligente de Conocimiento

Proyecto de la asignatura **Inteligencia Artificial** del Grado en Tecnologías Interactivas de la Universitat Politècnica de València.

El sistema implementa un agente RAG capaz de responder preguntas sobre la asociación **DNI (Damos Nuestra Ilusión)** utilizando únicamente el corpus oficial proporcionado. El agente recupera información relevante, genera una respuesta fundamentada, cita los archivos fuente utilizados y rechaza preguntas cuya respuesta no aparece en sus fuentes.

## Funcionalidades implementadas

La solución implementa las bandas 5, 6, 7, 8 y 10 de la práctica:

- Pipeline RAG completo sobre los 16 documentos oficiales de DNI.
- Rechazo anti-alucinación para preguntas fuera del corpus.
- Cita de archivos fuente en cada respuesta.
- Chunking adaptado a documentos narrativos y pares `Q:/A:`.
- Retrieval híbrido: búsqueda semántica + BM25.
- Gestión explícita de contradicciones reales del corpus.
- Arquitectura hexagonal con dominio, ports y adapters.
- Adapters intercambiables para LLM, embeddings y vector store.
- Benchmark reproducible con cuatro modelos.
- Evaluación RAGAs con cuatro métricas obligatorias.
- Dos métricas propias justificadas para el dominio DNI.
- Tests automatizados del dominio y de los adapters.

## Modelo final seleccionado

El modelo recomendado para la ejecución final del agente es:

```text
qwen2.5:3b mediante Ollama local
```

La elección se fundamenta en los resultados obtenidos:

- `12/12` aciertos en la revisión manual del benchmark.
- Mejor valor de `faithfulness` en RAGAs: `0.788889`.
- Mejor valor de `answer_relevancy` en RAGAs: `0.638091`.
- Funcionamiento completamente local, sin depender de VPN ni de un servicio remoto.

`gemma3:27b` mediante PoliGPT queda como alternativa remota especialmente interesante cuando se prioriza la velocidad de respuesta.

## Arquitectura

La solución utiliza arquitectura hexagonal o **ports & adapters**. La lógica del agente se mantiene separada de las tecnologías externas, de forma que es posible cambiar el modelo generativo, el proveedor de embeddings o el vector store mediante configuración.

```text
                 consultar.py
                      |
                      v
              composition.py
                      |
                      v
              ChatbotService
            dominio puro del RAG
             /        |        \
            v         v         v
         LLMPort  RetrieverPort  Entidades
            |         |
     +------+--+   +--+----------------------+
     |         |   |                         |
 OllamaLLM  PoliGPTLLM   HybridRetriever / Semantic / BM25
                             |
                    +--------+--------+
                    |                 |
                 FAISS             ChromaDB
```

### Capas principales

```text
src/agente_rag/
├── domain/
│   ├── entities.py
│   ├── ports.py
│   └── chatbot_service.py
├── adapters/
│   ├── llm/
│   │   ├── ollama_llm.py
│   │   └── poligpt_llm.py
│   ├── embeddings/
│   │   ├── ollama_embeddings.py
│   │   └── sentence_transformers_embeddings.py
│   └── retriever/
│       ├── semantic_retriever.py
│       ├── bm25_retriever.py
│       ├── hybrid_retriever.py
│       ├── chroma_vector_store.py
│       └── faiss_vector_store.py
├── composition.py
└── config.py
```

El flujo real de una consulta es:

```text
consultar.py → composition.py → ChatbotService → adapters configurados
```

## Pipeline RAG

El agente realiza el siguiente proceso:

1. Carga el corpus oficial DNI.
2. Divide los documentos en chunks, conservando unidos los pares `Q:/A:` cuando procede.
3. Genera embeddings locales mediante `nomic-embed-text`.
4. Recupera información relevante utilizando FAISS y retrieval híbrido.
5. Construye un prompt limitado al contexto recuperado.
6. Genera la respuesta mediante el LLM configurado.
7. Devuelve respuesta, fuentes, chunks y métricas.
8. Si la pregunta está fuera del corpus, devuelve:

```text
No tengo esa información en mis fuentes.
```

Cuando el corpus contiene versiones contradictorias, como el horario de los desayunos solidarios, el agente presenta ambas versiones con sus archivos fuente en lugar de ocultar la contradicción.

## Requisitos

- Windows 10 o sistema compatible.
- Python 3.11.
- Ollama instalado y en ejecución.
- Modelos Ollama necesarios:

```powershell
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Para utilizar PoliGPT fuera del campus se requiere conexión a la VPN de la UPV y una clave privada configurada únicamente en `.env`.

## Instalación en Windows / PowerShell

Desde la raíz del proyecto:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Copia el fichero de configuración de ejemplo:

```powershell
Copy-Item .env.example .env
```

El archivo `.env` es privado y no debe añadirse nunca a Git.

## Configuración final recomendada

La configuración final utilizada para el agente es:

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b
OLLAMA_URL=http://localhost:11434/api

EMBEDDER_PROVIDER=ollama
EMBED_MODEL=nomic-embed-text

VECTOR_STORE_PROVIDER=faiss
FAISS_PATH=./data/dni.index

CORPUS_DIR=./corpus
VERIFY_SSL=true
```

Los adapters permiten cambiar a PoliGPT, Sentence Transformers o ChromaDB mediante variables de entorno, sin modificar el dominio.

## Construcción del índice FAISS

Con el entorno virtual activo y Ollama funcionando:

```powershell
python scripts\build_hexagonal_index.py
```

La indexación real del corpus DNI genera 281 chunks y almacena el índice en:

```text
data/dni.index
```

## Ejecutar una consulta

Ejemplo de pregunta directa:

```powershell
python consultar.py "¿Qué es DNI?"
```

Ejemplo de pregunta fuera de ámbito:

```powershell
python consultar.py "¿Cuánto cuesta alquilar un piso en Valencia?"
```

Ejemplo de contradicción presente en el corpus:

```powershell
python consultar.py "¿A qué hora son los desayunos solidarios?"
```

La función obligatoria expuesta para el corrector es:

```python
def consultar(pregunta: str, conversation_id: str | None = None) -> dict:
    ...
```

La respuesta contiene:

```json
{
  "respuesta": "Texto generado por el agente",
  "fuentes": ["archivo_fuente.txt"],
  "chunks": [],
  "metricas": {},
  "trazas": null
}
```

## Benchmark con cuatro modelos

El benchmark utiliza 12 preguntas del dominio DNI, incluyendo preguntas factuales, logísticas, una contradicción real y dos preguntas fuera de ámbito.

Modelos evaluados:

| Proveedor | Modelo |
|---|---|
| Ollama local | `qwen2.5:3b` |
| Ollama local | `llama3.2:3b` |
| PoliGPT | `gemma3:27b` |
| PoliGPT | `llama3.3:70b` |

Durante la comparación se mantuvieron constantes el corpus, el chunking, el retrieval, los embeddings y el vector store. Solo se cambió el LLM generativo.

Los resultados consolidados están disponibles en:

```text
benchmark/benchmark.json
benchmark/benchmark.md
```

## Evaluación RAGAs y métricas propias

Se han calculado las cuatro métricas RAGAs requeridas:

- `faithfulness`
- `answer_relevancy`
- `context_precision`
- `context_recall`

Además, se han definido dos métricas propias:

- `expected_source_coverage`: cobertura de archivos fuente esperados.
- `out_of_scope_rejection_accuracy`: acierto al rechazar preguntas fuera del corpus.

Resultados resumidos:

| Modelo | Calidad manual | Faithfulness | Answer relevancy | Context precision | Context recall |
|---|---:|---:|---:|---:|---:|
| `qwen2.5:3b` | 12/12 | 0.788889 | 0.638091 | 0.669444 | 0.833333 |
| `llama3.2:3b` | 10/12 | 0.716667 | 0.471928 | 0.669444 | 0.833333 |
| `gemma3:27b` | 12/12 | 0.716667 | 0.612921 | 0.669444 | 0.833333 |
| `llama3.3:70b` | 12/12 | 0.775463 | 0.582995 | 0.669444 | 0.833333 |

Los resultados detallados se encuentran en:

```text
evaluacion/ragas_results.json
evaluacion/metricas_propias.md
```

## Tests

La solución dispone de tests automatizados, incluidos tests del dominio sin dependencia de red y tests de adapters.

Para ejecutarlos:

```powershell
python -m pytest -q
```

Resultado actual:

```text
54 tests correctos
```

## Estructura principal del repositorio

```text
pracAgentes/
├── consultar.py
├── features.json
├── GRUPO.md
├── AI_USAGE.md
├── .env.example
├── corpus/
├── data/
├── benchmark/
│   ├── preguntas.json
│   ├── benchmark.json
│   ├── benchmark.md
│   └── README.md
├── evaluacion/
│   ├── ragas_results.json
│   └── metricas_propias.md
├── docs/
│   ├── ARCHITECTURE.md
│   └── CONTRACT.md
├── scripts/
│   ├── build_hexagonal_index.py
│   ├── build_benchmark_report.py
│   ├── run_eval.py
│   └── run_ragas_eval.py
├── src/agente_rag/
└── tests/
```

## Seguridad y credenciales

- `.env` contiene configuración privada y nunca debe commitearse.
- La clave de PoliGPT solo debe almacenarse en `.env`.
- `.env.example` contiene únicamente valores de ejemplo seguros.
- Los resultados entregados no incluyen credenciales.

## Uso de inteligencia artificial

El uso de ChatGPT durante el desarrollo, depuración y documentación se declara de forma honesta en:

```text
AI_USAGE.md
```

## Integrantes

Los integrantes del grupo y su reparto de trabajo se documentan en:

```text
GRUPO.md
```