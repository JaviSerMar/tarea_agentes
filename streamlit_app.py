"""Interfaz Streamlit para demostrar el asistente RAG de DNI."""

from __future__ import annotations

from html import escape

import streamlit as st

from consultar import consultar


st.set_page_config(
    page_title="Asistente DNI",
    page_icon="🤝",
    layout="wide",
    initial_sidebar_state="expanded",
)

EJEMPLOS = [
    "¿Qué es DNI?",
    "¿A qué hora son los desayunos solidarios?",
    "¿Cuánto cuesta alquilar un piso en Valencia?",
]

if "pregunta" not in st.session_state:
    st.session_state.pregunta = ""

if "resultado" not in st.session_state:
    st.session_state.resultado = None

if "ultima_pregunta" not in st.session_state:
    st.session_state.ultima_pregunta = ""


st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(255, 186, 73, 0.10), transparent 30%),
                radial-gradient(circle at bottom left, rgba(34, 197, 94, 0.08), transparent 28%),
                #0d1117;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        .hero {
            padding: 2.2rem 2.4rem;
            margin-bottom: 1.8rem;
            border-radius: 22px;
            border: 1px solid rgba(255, 193, 71, 0.24);
            background: linear-gradient(
                135deg,
                rgba(255, 193, 71, 0.14),
                rgba(18, 23, 32, 0.92) 48%,
                rgba(34, 197, 94, 0.08)
            );
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
        }

        .hero-badge {
            display: inline-block;
            padding: 0.30rem 0.72rem;
            margin-bottom: 1rem;
            border-radius: 999px;
            background: rgba(255, 193, 71, 0.13);
            border: 1px solid rgba(255, 193, 71, 0.30);
            color: #ffd166;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.03em;
        }

        .hero h1 {
            margin: 0 0 0.55rem 0;
            font-size: 2.45rem;
            line-height: 1.1;
            color: #f8fafc;
        }

        .hero p {
            margin: 0;
            max-width: 720px;
            color: #b4bdc9;
            font-size: 1.02rem;
            line-height: 1.65;
        }

        .section-title {
            margin-top: 1.6rem;
            margin-bottom: 0.75rem;
            color: #f8fafc;
            font-size: 1.28rem;
            font-weight: 700;
        }

        .query-label {
            color: #8d99a8;
            font-size: 0.92rem;
            margin-bottom: 0.9rem;
        }

        .source-chip {
            display: inline-block;
            margin-right: 0.55rem;
            margin-bottom: 0.45rem;
            padding: 0.38rem 0.72rem;
            border-radius: 999px;
            color: #48e38a;
            background: rgba(34, 197, 94, 0.10);
            border: 1px solid rgba(34, 197, 94, 0.22);
            font-family: monospace;
            font-size: 0.88rem;
        }

        div[data-testid="stMetric"] {
            padding: 1rem 1.05rem;
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.16);
            background: rgba(22, 27, 36, 0.88);
        }

        div[data-testid="stMetricLabel"] {
            color: #98a4b3;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc;
        }

        div[data-testid="stExpander"] {
            border-radius: 14px;
            border-color: rgba(148, 163, 184, 0.18);
            background: rgba(17, 22, 30, 0.62);
        }

        div.stButton > button[kind="primary"] {
            min-height: 3rem;
            border-radius: 12px;
            border: none;
            font-weight: 700;
            background: linear-gradient(90deg, #f59e0b, #f97316);
            color: #111827;
        }

        div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(90deg, #fbbf24, #fb923c);
            color: #111827;
        }

        section[data-testid="stSidebar"] {
            background: #11161f;
            border-right: 1px solid rgba(148, 163, 184, 0.12);
        }

        section[data-testid="stSidebar"] div.stButton > button {
            text-align: left;
            border-radius: 10px;
        }

        .footer-note {
            margin-top: 2.5rem;
            padding-top: 1rem;
            border-top: 1px solid rgba(148, 163, 184, 0.12);
            color: #778392;
            font-size: 0.84rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## 🤝 DNI")
    st.caption("Damos Nuestra Ilusión")
    st.divider()

    st.markdown("### Preguntas de demostración")
    st.caption("Selecciona una consulta y pulsa **Consultar**.")

    for ejemplo in EJEMPLOS:
        if st.button(ejemplo, use_container_width=True):
            st.session_state.pregunta = ejemplo

    st.divider()
    st.markdown("### Capacidades")
    st.markdown(
        """
        - Respuestas fundamentadas
        - Fuentes documentales
        - Detección de contradicciones
        - Rechazo fuera de ámbito
        - Métricas de ejecución
        """
    )

    st.info(
        "El asistente solo responde con información recuperada "
        "del corpus oficial de DNI."
    )


st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AGENTE RAG · DEMOSTRACIÓN INTERACTIVA</div>
        <h1>Asistente Inteligente de DNI</h1>
        <p>
            Consulta información sobre Damos Nuestra Ilusión mediante un agente
            que recupera evidencias del corpus oficial, cita sus fuentes y evita
            responder cuando no dispone de información suficiente.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

pregunta = st.text_input(
    "Formula una pregunta sobre DNI",
    key="pregunta",
    placeholder="Ejemplo: ¿Qué es DNI?",
)

consultar_pulsado = st.button(
    "Consultar al agente",
    type="primary",
    use_container_width=True,
)

if consultar_pulsado:
    if not pregunta.strip():
        st.warning("Escribe una pregunta antes de consultar.")
    else:
        with st.spinner("Recuperando información y generando respuesta..."):
            try:
                st.session_state.resultado = consultar(pregunta.strip())
                st.session_state.ultima_pregunta = pregunta.strip()
            except Exception as exc:
                st.session_state.resultado = None
                st.error(
                    "No se ha podido obtener una respuesta. "
                    "Comprueba que Ollama esté iniciado y vuelve a intentarlo."
                )
                with st.expander("Detalle técnico"):
                    st.code(str(exc))

resultado = st.session_state.resultado

if resultado:
    respuesta = resultado.get("respuesta", "")
    fuentes = resultado.get("fuentes") or []
    chunks = resultado.get("chunks") or []
    metricas = resultado.get("metricas") or {}

    st.markdown('<div class="section-title">Respuesta</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="query-label">Consulta: {escape(st.session_state.ultima_pregunta)}</div>',
        unsafe_allow_html=True,
    )

    if respuesta.strip() == "No tengo esa información en mis fuentes.":
        st.warning(respuesta)
    else:
        st.success(respuesta)

    st.markdown('<div class="section-title">Fuentes utilizadas</div>', unsafe_allow_html=True)
    if fuentes:
        chips = "".join(
            f'<span class="source-chip">{escape(fuente)}</span>'
            for fuente in fuentes
        )
        st.markdown(chips, unsafe_allow_html=True)
    else:
        st.caption("No se han citado fuentes.")

    st.markdown('<div class="section-title">Métricas de ejecución</div>', unsafe_allow_html=True)
    st.caption(f"Modelo utilizado: `{metricas.get('modelo', '—')}`")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latencia", f"{metricas.get('latencia_s', '—')} s")
    col2.metric("Tokens entrada", metricas.get("prompt_tokens", "—"))
    col3.metric("Tokens salida", metricas.get("output_tokens", "—"))
    col4.metric("Tokens/s", metricas.get("tokens_per_sec", "—"))

    st.markdown('<div class="section-title">Evidencias recuperadas</div>', unsafe_allow_html=True)
    if chunks:
        for indice, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "fuente desconocida")
            score = chunk.get("score")
            score_texto = (
                f" · relevancia: {score:.4f}"
                if isinstance(score, (int, float))
                else ""
            )
            with st.expander(f"Evidencia {indice} · {source}{score_texto}"):
                st.text(chunk.get("text", ""))
    else:
        st.caption("No hay chunks disponibles para mostrar.")

st.markdown(
    """
    <div class="footer-note">
        Asistente DNI · Interfaz demostrativa construida sobre el pipeline RAG del proyecto.
    </div>
    """,
    unsafe_allow_html=True,
)
