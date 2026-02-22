"""
UtopIA — Asistente Inteligente de Utopia Padel Cancún

Aplicación Streamlit que permite a los administradores del club
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
    page_title="UtopIA — Utopia Padel Cancún",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    div[data-testid="stChatMessage"] { padding: 1rem; }
    #MainMenu { visibility: hidden; }
    header[data-testid="stHeader"] .stActionButton { display: none; }
    .stDeployButton { display: none; }
    [data-testid="manage-app-button"] { display: none; }
    .viewerBadge_container__r5tak { display: none; }
    ._profileContainer_gzau3_53 { display: none; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Barra lateral ──────────────────────────────────────────────────────

QUICK_PROMPTS = {
    "📊 Resumen de hoy": "¿Cómo estuvo el club hoy? Dame un resumen completo de reservas, ocupación e ingresos.",
    "📅 Resumen de la semana": "¿Cómo va la semana? Muéstrame ocupación e ingresos de esta semana.",
    "💰 Ingresos del mes": "¿Cuánto hemos facturado este mes? Desglose por cancha y día.",
    "👥 Top jugadores": "¿Quiénes son los jugadores que más reservan este mes?",
    "⚠️ Alertas operativas": "¿Hay algo que deba atender? Cancelaciones, impagos, horarios muertos.",
    "🕐 Disponibilidad mañana": "¿Qué disponibilidad hay mañana? ¿Qué canchas están libres?",
}

with st.sidebar:
    st.title("UtopIA")
    st.caption("Asistente Inteligente — Utopia Padel Cancún")
    st.divider()

    st.subheader("Consultas rápidas")
    for label, prompt_text in QUICK_PROMPTS.items():
        if st.button(label, use_container_width=True):
            st.session_state.pending_prompt = prompt_text
            st.rerun()

    st.divider()

    if st.button("🗑️ Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        if "agent" in st.session_state:
            st.session_state.agent.reset_conversation()
        st.rerun()

    st.divider()
    st.caption("Hecho con UtopIA + Playtomic API")

# ── Validar configuración ──────────────────────────────────────────────

missing = []
if not CLIENT_ID or not CLIENT_SECRET:
    missing.append("PLAYTOMIC_CLIENT_ID / PLAYTOMIC_CLIENT_SECRET")
if not TENANT_ID:
    missing.append("PLAYTOMIC_TENANT_ID")
if not OPENAI_KEY:
    missing.append("OPENAI_API_KEY")

st.header("UtopIA")

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
            "¡Hola! Soy **UtopIA**, tu asistente inteligente de **Utopia Padel Cancún**. "
            "Te puedo ayudar con:\n\n"
            '- **Ocupación de canchas** — *"¿Qué tan lleno está el club mañana?"*\n'
            '- **Detalle de reservas** — *"¿Quién jugó en Hirostar ayer?"*\n'
            '- **Ingresos** — *"¿Cuánto facturamos esta semana?"*\n'
            '- **Jugadores** — *"¿Quiénes son los que más reservan este mes?"*\n'
            '- **Alertas operativas** — *"¿Cuál es nuestra tasa de cancelación?"*\n\n'
            "¡Pregúntame lo que necesites!"
        )

# Historial del chat
for msg in st.session_state.messages:
    avatar = "🎾" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        for fig in msg.get("charts", []):
            st.plotly_chart(fig, use_container_width=True)

# Procesar consulta rápida del sidebar
if "pending_prompt" in st.session_state:
    prompt = st.session_state.pop("pending_prompt")
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
