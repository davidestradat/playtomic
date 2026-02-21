"""
Playtomic Club Manager — Panel de Analítica con IA

Aplicación Streamlit que permite a los administradores de clubes de pádel
hacer preguntas en lenguaje natural sobre ocupación, reservas, ingresos y operaciones.
"""

import os

import streamlit as st
from dotenv import load_dotenv

from playtomic_api import PlaytomicAPI
from llm_agent import PlaytomicAgent
from charts import build_charts

load_dotenv()


def get_secret(key: str, default: str = "") -> str:
    """Get config from Streamlit Cloud secrets first, then .env fallback."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)


# ── Configuración ───────────────────────────────────────────────────────

CLIENT_ID = get_secret("PLAYTOMIC_CLIENT_ID")
CLIENT_SECRET = get_secret("PLAYTOMIC_CLIENT_SECRET")
TENANT_ID = get_secret("PLAYTOMIC_TENANT_ID")
OPENAI_KEY = get_secret("OPENAI_API_KEY")
OPENAI_MODEL = get_secret("OPENAI_MODEL", "gpt-4o")
CLUB_TIMEZONE = get_secret("CLUB_TIMEZONE", "America/Cancun")

# ── Página ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Playtomic Club Manager",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    div[data-testid="stChatMessage"] { padding: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Barra lateral ──────────────────────────────────────────────────────

with st.sidebar:
    st.title("Playtomic Club Manager")
    st.caption("Analítica de club con IA")
    st.divider()

    if st.button("Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        if "agent" in st.session_state:
            st.session_state.agent.reset_conversation()
        st.rerun()

    st.divider()
    st.caption("Hecho con Streamlit + OpenAI + Playtomic API")

# ── Validar configuración ──────────────────────────────────────────────

missing = []
if not CLIENT_ID or not CLIENT_SECRET:
    missing.append("PLAYTOMIC_CLIENT_ID / PLAYTOMIC_CLIENT_SECRET")
if not TENANT_ID:
    missing.append("PLAYTOMIC_TENANT_ID")
if not OPENAI_KEY:
    missing.append("OPENAI_API_KEY")

st.header("Asistente del Club")

if missing:
    st.error("Falta configuración en el archivo `.env`:")
    for item in missing:
        st.markdown(f"- `{item}`")
    st.stop()

# ── Inicializar agente ─────────────────────────────────────────────────

if "agent" not in st.session_state:
    api = PlaytomicAPI(CLIENT_ID, CLIENT_SECRET, tz_name=CLUB_TIMEZONE)
    st.session_state.agent = PlaytomicAgent(
        api=api,
        tenant_id=TENANT_ID,
        openai_api_key=OPENAI_KEY,
        model=OPENAI_MODEL,
        timezone_name=CLUB_TIMEZONE,
    )

agent = st.session_state.agent

# ── Interfaz de chat ───────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mensaje de bienvenida
if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🎾"):
        st.markdown(
            "¡Hola! Soy tu **Asistente de Club Playtomic**. "
            "Te puedo ayudar con:\n\n"
            '- **Ocupación de canchas** — *"¿Qué tan lleno está el club mañana?"*\n'
            '- **Detalle de reservas** — *"¿Quién jugó en Hirostar ayer?"*\n'
            '- **Ingresos** — *"¿Cuánto facturamos esta semana?"*\n'
            '- **Jugadores** — *"¿Quiénes son los que más reservan este mes?"*\n'
            '- **Alertas operativas** — *"¿Cuál es nuestra tasa de cancelación?"*\n\n'
            "¡Pregúntame lo que necesites sobre tu club de pádel!"
        )

# Historial del chat
for msg in st.session_state.messages:
    avatar = "🎾" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        for fig in msg.get("charts", []):
            st.plotly_chart(fig, use_container_width=True)

# Entrada de chat
if prompt := st.chat_input("Pregunta sobre tu club... (ej: '¿Quién jugó en Hirostar ayer?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🎾"):
        with st.spinner("Analizando los datos del club..."):
            try:
                response, chart_data = agent.chat(prompt)
                st.markdown(response)

                all_figures = []
                for tool_name, tool_result in chart_data:
                    all_figures.extend(build_charts(tool_name, tool_result))

                for fig in all_figures:
                    st.plotly_chart(fig, use_container_width=True)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "charts": all_figures,
                })
            except Exception as e:
                error_msg = f"Lo siento, ocurrió un error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
