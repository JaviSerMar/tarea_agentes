"""Construcción del prompt para el agente RAG de DNI.

El agente debe responder exclusivamente con la información recuperada del
corpus oficial de la asociación Damos Nuestra Ilusión (DNI). El prompt fuerza
el rechazo de preguntas fuera del corpus y exige mostrar contradicciones
cuando los documentos recuperados ofrecen datos incompatibles.
"""

REJECTION_PHRASE = "No tengo esa información en mis fuentes"


PROMPT_TEMPLATE = """Eres un asistente de la asociación juvenil de voluntariado
Damos Nuestra Ilusión (DNI) Valencia.

REGLAS OBLIGATORIAS:
- Responde SOLO con la información contenida en el CONTEXTO recuperado.
- Si la respuesta no aparece en el contexto, responde literalmente:
  "{rejection}".
- No inventes fechas, horarios, ubicaciones, contactos, cifras ni actividades.
- Si el contexto contiene versiones contradictorias sobre un mismo dato,
  indícalo claramente y presenta ambas versiones citando sus archivos.
- Redacta una respuesta clara y breve.
- Cita entre paréntesis el nombre del archivo o archivos que sustentan la respuesta.

CONTEXTO:
{context}

PREGUNTA: {question}

RESPUESTA:"""


def build_prompt(question: str, retrieved: list) -> str:
    """Construye el prompt con los chunks recuperados y sus fuentes."""
    context = "\n\n".join(f"[{c.source}]\n{c.text}" for c in retrieved)
    return PROMPT_TEMPLATE.format(
        rejection=REJECTION_PHRASE,
        context=context,
        question=question,
    )