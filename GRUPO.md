# Grupo de prácticas — Agente RAG DNI

## Integrantes

|    Apellidos, Nombre      |    DNI    |      Correo UPV      | Grado/Curso |
|---------------------------|-----------|----------------------|-------------|
| Serrano Marco, Javier     | 21698497K | jsermar1@epsg.upv.es |   GTI / 4º  |
| Camarena Cuartero, Javier | 20947810P | jcamcua@upv.edu.es   |   GTI / 4º  |
| Ferrer Prats, Jaime       | 48198409T | jferpra@epsg.upv.es  |   GTI / 4º  |

## Reparto de trabajo

El proyecto se ha desarrollado de forma colaborativa. El reparto concreto de responsabilidades ha sido el siguiente:

- **Javier Serrano Marco**: benchmark de modelos, evaluación RAGAs y análisis de resultados.
- **Javier Camarena Cuartero**: desarrollo del agente RAG, arquitectura hexagonal y adapters.
- **Jaime Ferrer Prats**: tests, documentación final y preparación de la defensa.

## Trabajo realizado en el proyecto

El grupo ha desarrollado un agente RAG sobre el corpus oficial de DNI con las siguientes funcionalidades:

- Pipeline RAG con rechazo de preguntas fuera de ámbito.
- Cita de archivos fuente en las respuestas.
- Retrieval híbrido semántico + BM25 y tratamiento de contradicciones del corpus.
- Arquitectura hexagonal con adapters intercambiables para LLM, embeddings y vector store.
- Benchmark con dos modelos locales y dos modelos PoliGPT.
- Evaluación mediante RAGAs y dos métricas propias.
- Documentación y preparación de la entrega final.

## Uso de inteligencia artificial

El uso de asistentes de IA se detalla de forma específica y honesta en `AI_USAGE.md`.

## Declaración

Los integrantes declaramos haber revisado el código y la documentación entregados, y estar en condiciones de explicar las decisiones principales del sistema durante la defensa oral.

Firmas:

- Javier Serrano Marco
- Javier Camarena Cuartero
- Jaime Ferrer Prats