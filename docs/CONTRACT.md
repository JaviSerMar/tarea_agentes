# Contrato de interfaz — Agente RAG DNI

## Opción utilizada

La entrega utiliza la **opción A: módulo Python**. El fichero `consultar.py`, situado en la raíz del repositorio, expone la función que utilizará el corrector:

```python
def consultar(pregunta: str, conversation_id: str | None = None) -> dict:
    ...
```

Esta elección se declara en `features.json`:

```json
{
  "interfaz": "modulo_python",
  "modulo": "consultar.py",
  "endpoint_http": null,
  "arquitectura": "hexagonal"
}
```

## Formato de salida

La función devuelve un diccionario con la siguiente estructura:

```json
{
  "respuesta": "Texto generado a partir de las fuentes recuperadas.",
  "fuentes": [
    "08_preguntas_basicas.txt"
  ],
  "chunks": [
    {
      "source": "08_preguntas_basicas.txt",
      "text": "Q: ¿Qué es DNI? A: DNI (Damos Nuestra Ilusión)...",
      "score": 1.0
    }
  ],
  "metricas": {
    "prompt_tokens": 359,
    "output_tokens": 98,
    "tokens_per_sec": 142.84,
    "latencia_s": 3.2,
    "modelo": "qwen2.5:3b"
  },
  "trazas": null
}
```

## Campos devueltos

|    Clave    |         Tipo         |                                     Uso                                     |
|-------------|----------------------|-----------------------------------------------------------------------------|
| `respuesta` |         `str`        | Respuesta final del agente.                                                 |
| `fuentes`   |      `list[str]`     | Archivos del corpus recuperados y utilizados para fundamentar la respuesta. |
| `chunks`    |     `list[dict]`     | Fragmentos recuperados, con fuente, texto y score.                          |
| `metricas`  |        `dict`        | Tokens, velocidad de generación, latencia y modelo empleado.                |
| `trazas`    | `list[dict] \| None` | Campo opcional reservado para trazas internas.                              |

## Comportamiento ante preguntas fuera de ámbito

Cuando la información solicitada no está en el corpus DNI, el agente devuelve la frase anti-alucinación:

```text
No tengo esa información en mis fuentes.
```

Ejemplo:

```python
consultar("¿Cuánto cuesta alquilar un piso en Valencia?")
```

Resultado esperado en `respuesta`:

```text
No tengo esa información en mis fuentes.
```

## Comportamiento ante contradicciones del corpus

Si los documentos recuperados contienen versiones distintas sobre un mismo dato, el agente no inventa una única respuesta. En su lugar, presenta las versiones disponibles y cita sus archivos fuente.

Ejemplo:

```python
consultar("¿A qué hora son los desayunos solidarios?")
```

La respuesta debe reflejar que:

- `01_faq_dni.txt` indica las 8:00.
- `11_horarios_ubicaciones.txt` indica normalmente entre las 9:00 y las 12:00.

## Relación con la arquitectura hexagonal

Aunque el corrector entra por `consultar.py`, la lógica interna no está implementada como un script monolítico. El flujo es:

```text
consultar.py → composition.py → ChatbotService → ports → adapters
```

De esta forma:

- `consultar.py` actúa como adapter de entrada;
- `ChatbotService` contiene la lógica principal del dominio;
- los ports definen contratos abstractos;
- los adapters concretos permiten elegir Ollama o PoliGPT, FAISS o ChromaDB y distintos proveedores de embeddings.

## Ejecución manual

Con la configuración final local:

```powershell
python consultar.py "¿Qué es DNI?"
python consultar.py "¿A qué hora son los desayunos solidarios?"
python consultar.py "¿Cuánto cuesta alquilar un piso en Valencia?"
```

## Verificación mediante tests

El contrato se comprueba mediante la batería de tests:

```powershell
python -m pytest -q
```

Resultado validado durante el desarrollo:

```text
54 tests correctos
```