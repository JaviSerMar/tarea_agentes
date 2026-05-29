"""Interfaz Streamlit para demostrar el asistente RAG de DNI."""

from __future__ import annotations

import streamlit as st

from consultar import consultar


st.set_page_config(
    page_title="Asistente DNI",
    page_icon="🤝",
    layout="wide",
)

st.title("🤝 Asistente Inteligente de DNI")
st.caption(
    "Consulta información de Damos Nuestra Ilusión a partir del corpus oficial."
)

with st.sidebar:
    st.header("Consultas para la demo")
    st.markdown(
        """
- ¿Qué es DNI?
- ¿A qué hora son los desayunos solidarios?
- ¿Cuánto cuesta alquilar un piso en Valencia?
        """
    )
    st.info(
        "El asistente responde únicamente con información recuperada "
        "de sus fuentes documentales."
    )

pregunta = st.text_input(
    "Escribe tu pregunta",
    placeholder="Ejemplo: ¿Qué es DNI?",
)

consultar_pulsado = st.button(
    "Consultar",
    type="primary",
    use_container_width=True,
)

if consultar_pulsado:
    if not pregunta.strip():
        st.warning("Escribe una pregunta antes de consultar.")
    else:
        with st.spinner("Consultando la base de conocimiento..."):
            try:
                resultado = consultar(pregunta.strip())
            except Exception as exc:
                st.error(
                    "No se ha podido obtener una respuesta. "
                    "Comprueba que Ollama esté iniciado y vuelve a intentarlo."
                )
                with st.expander("Detalle técnico"):
                    st.code(str(exc))
            else:
                respuesta = resultado.get("respuesta", "")
                fuentes = resultado.get("fuentes") or []
                chunks = resultado.get("chunks") or []
                metricas = resultado.get("metricas") or {}

                st.subheader("Respuesta")
                if respuesta.strip() == "No tengo esa información en mis fuentes.":
                    st.warning(respuesta)
                else:
                    st.success(respuesta)

                st.subheader("Fuentes")
                if fuentes:
                    for fuente in fuentes:
                        st.markdown(f"- `{fuente}`")
                else:
                    st.caption("No se han citado fuentes.")

                st.subheader("Métricas")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Modelo", metricas.get("modelo", "—"))
                col2.metric("Latencia", f"{metricas.get('latencia_s', '—')} s")
                col3.metric("Tokens entrada", metricas.get("prompt_tokens", "—"))
                col4.metric("Tokens salida", metricas.get("output_tokens", "—"))

                if metricas.get("tokens_per_sec") is not None:
                    st.caption(
                        f"Velocidad de generación: "
                        f"{metricas['tokens_per_sec']} tokens/s"
                    )

                st.subheader("Chunks recuperados")
                if chunks:
                    for indice, chunk in enumerate(chunks, start=1):
                        source = chunk.get("source", "fuente desconocida")
                        score = chunk.get("score")
                        score_texto = (
                            f" · score: {score:.4f}"
                            if isinstance(score, (int, float))
                            else ""
                        )
                        with st.expander(
                            f"Chunk {indice} · {source}{score_texto}"
                        ):
                            st.text(chunk.get("text", ""))
                else:
                    st.caption("No hay chunks disponibles para mostrar.")