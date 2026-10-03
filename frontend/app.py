"""Frontend: un único chat con el sistema agéntico.

    uv run streamlit run frontend/app.py

El chat habla con `backend.chat.AsistenteChat`, un agente que decide en cada pregunta qué
agentes del sistema activar (Predictor, Analista, Recomendador).
"""
import sys
import threading
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import streamlit as st  # noqa: E402
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx  # noqa: E402

from config import settings  # noqa: E402
from backend.pipeline import consumo  # noqa: E402
from backend.chat.asistente import AsistenteChat  # noqa: E402
from backend.chat.herramientas import HerramientasChat  # noqa: E402
from frontend.components.trazas import mostrar_llamadas, mostrar_recomendaciones  # noqa: E402

st.set_page_config(page_title="Asistente de retención", page_icon="🏦", layout="centered")

# Un ejemplo por herramienta del Asistente, en el mismo orden: buscar_cliente,
# consultar_predictor, consultar_analista, recomendar, resumen_cartera.
EJEMPLOS = [
    "Busca a los clientes que se apellidan Sun y dime sus datos básicos",
    "¿Qué riesgo de abandono tiene Gray, el cliente alemán de 69 años?",
    "¿Por qué tiene ese riesgo el cliente del perfil balance_cero?",
    "Dame una recomendación de retención para el perfil perfil_limpio",
    "¿Qué clientes de Alemania están en revisión y cuáles tienen más riesgo?",
]


@st.cache_resource(show_spinner="Cargando modelo y base histórica...")
def herramientas_compartidas() -> HerramientasChat:
    """Modelo, df_train y test set se cargan una vez por proceso (solo lectura)."""
    return HerramientasChat()


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


# ---------------------------------------------------------------------------
# Estado de la sesión
# ---------------------------------------------------------------------------
# Si el código cambió con la app abierta, la sesión puede guardar un Asistente de una versión
# anterior de la clase (p. ej. sin `tokens_agentes`): en ese caso se crea uno nuevo.
_previo = st.session_state.get("asistente")
if not isinstance(_previo, AsistenteChat) or not hasattr(_previo, "tokens_agentes"):
    st.session_state.asistente = AsistenteChat(herramientas=herramientas_compartidas())
    st.session_state.historial = []   # [{'rol', 'texto', 'llamadas', 'tokens'}]
asistente: AsistenteChat = st.session_state.asistente
historial: list = st.session_state.historial
hay_clave = settings.hay_api_key()


def _escritor(estado):
    """estado.write utilizable desde los hilos de CrewAI.

    Los agentes corren en un hilo aparte (asyncio.to_thread) y los avisos de espera por el
    límite de Groq se emiten desde ahí. Streamlit solo deja escribir desde un hilo con el
    contexto de la sesión, así que se le asocia el contexto del script antes de escribir.
    """
    ctx = get_script_run_ctx()

    def escribir(mensaje: str) -> None:
        if ctx is not None and get_script_run_ctx(suppress_warning=True) is None:
            add_script_run_ctx(threading.current_thread(), ctx)
        estado.write(mensaje)
    return escribir


def _procesar(accion, etiqueta: str):
    """Ejecuta una acción del asistente mostrando el progreso y guarda el turno."""
    with st.chat_message("assistant"):
        with st.status(etiqueta, expanded=False) as estado:
            try:
                turno = accion(_escritor(estado))
            except Exception as e:  # noqa: BLE001 — se muestra al usuario, la sesión sigue viva
                estado.update(label="Error", state="error")
                historial.append({'rol': 'assistant', 'llamadas': [], 'tokens': 0,
                                  'texto': f"⚠️ No he podido completar la respuesta: `{e}`"})
                return
            estado.update(label="Listo", state="complete")
    historial.append({'rol': 'assistant', 'texto': turno.texto, 'llamadas': turno.llamadas,
                      'tokens': turno.tokens})


# ---------------------------------------------------------------------------
# Barra lateral: estado y consumo
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Estado")
    if hay_clave:
        st.success("Groq configurado")
    else:
        st.error("Falta `GROQ_API_KEY` en `.env`")
    gasto = consumo.resumen()
    st.metric("Tokens gastados hoy", _miles(gasto['hoy']))
    st.progress(min(gasto['hoy'] / settings.LIMITE_TOKENS_DIA, 1.0),
                text=f"frente al límite diario ({_miles(settings.LIMITE_TOKENS_DIA)})")
    st.caption(f"Día UTC {gasto['dia_utc']} · {gasto['llamadas_hoy']} llamadas a Groq · "
               f"últimas 24 h: {_miles(gasto['ultimas_24h'])}.")
    st.caption(f"Esta conversación: {_miles(asistente.tokens_totales)} "
               f"(Asistente {_miles(asistente.tokens_llm)} · Agentes {_miles(asistente.tokens_agentes)}).")
    st.caption("Cuenta todas las llamadas a Groq hechas con este repositorio en este ordenador "
               "(chat, agentes y scripts por consola), incluidos los intentos que luego se "
               "repitieron. No incluye el notebook de Colab ni otros programas con la misma clave: "
               "el total exacto de la cuenta está en la consola de Groq. "
               f"Límite por minuto: {_miles(settings.LIMITE_TOKENS_MINUTO)}.")
    if st.button("Nueva conversación", width="stretch"):
        asistente.reiniciar()
        historial.clear()
        st.rerun()

# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
st.title("Asistente de retención de clientes")
st.caption("Sistema multiagente de predicción y retención de churn · CrewAI + Groq + LightGBM")

if not hay_clave:
    st.info("Para conversar hace falta una clave de Groq: copia `.env.example` como `.env`, pon tu "
            "`GROQ_API_KEY` y reinicia la aplicación.")

prompt_ejemplo = None
if not historial:
    with st.chat_message("assistant"):
        st.markdown(
            "Hola. Pregúntame por cualquier cliente del conjunto de test y activaré los agentes del "
            "sistema que hagan falta: el **Predictor** para el riesgo, el **Analista** para el porqué "
            "y el **Recomendador** (tras el router) para la acción de retención. Antes de activar el "
            "Analista o el Recomendador te diré cuántos tokens cuesta y te pediré confirmación. "
            "También puedes preguntarme por la cartera completa (sin coste de agentes).\n\n"
            "Cada cifra sale de un agente o herramienta; puedes verlo en *Herramientas usadas*. "
            "Prueba, por ejemplo:")
        for i, ejemplo in enumerate(EJEMPLOS):
            if st.button(ejemplo, key=f"ejemplo_{i}", disabled=not hay_clave, width="stretch"):
                prompt_ejemplo = ejemplo

for entrada in historial:
    if not entrada['texto'] and not entrada.get('llamadas'):
        continue
    with st.chat_message("user" if entrada['rol'] == 'user' else "assistant"):
        if entrada['texto']:
            st.markdown(entrada['texto'])
        mostrar_recomendaciones(entrada.get('llamadas', []))
        mostrar_llamadas(entrada.get('llamadas', []))
        if entrada.get('tokens'):
            st.caption(f"{_miles(entrada['tokens'])} tokens")

# Acción pendiente de confirmación (Recomendador)
if asistente.pendiente:
    previa = asistente.pendiente['previa']
    with st.chat_message("assistant"):
        agentes = ", ".join(previa['agentes'])
        st.warning(
            f"Para **{previa['cliente']}** voy a activar: **{agentes}**. Consume unos "
            f"**{_miles(previa['tokens_estimados'])} tokens** de Groq y puede tardar varios minutos si "
            f"se alcanza el límite por minuto. {previa.get('nota', '')} ¿Lo ejecuto?")
        c1, c2 = st.columns(2)
        if c1.button("Ejecutar", type="primary", key="confirmar_si", width="stretch"):
            _procesar(lambda ev: asistente.confirmar(True, on_evento=ev),
                      "Los agentes están trabajando (puede tardar unos minutos)...")
            st.rerun()
        if c2.button("Cancelar", key="confirmar_no", width="stretch"):
            _procesar(lambda ev: asistente.confirmar(False, on_evento=ev), "Cancelando...")
            st.rerun()

prompt = st.chat_input("Pregunta por un cliente: su riesgo, el porqué o qué hacer…", disabled=not hay_clave)
prompt = prompt or prompt_ejemplo
if prompt:
    historial.append({'rol': 'user', 'texto': prompt, 'llamadas': [], 'tokens': 0})
    with st.chat_message("user"):
        st.markdown(prompt)
    _procesar(lambda ev: asistente.responder(prompt, on_evento=ev), "Consultando al sistema...")
    st.rerun()
