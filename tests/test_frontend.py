"""Frontend de chat (AppTest de Streamlit, sin Groq: el LLM y los agentes se sustituyen por falsos)."""
import pytest

pytest.importorskip("streamlit")
pytest.importorskip("crewai")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import RAIZ  # noqa: E402
from tests.llm_falso import LLMFalso, llamada, respuesta  # noqa: E402
from tests.test_chat import GRAY_ID, AgentesFalsos  # noqa: E402

pytestmark = [pytest.mark.datos, pytest.mark.crewai]

APP = str(RAIZ / "frontend" / "app.py")


@pytest.fixture
def entorno(monkeypatch, tmp_path):
    """Clave simulada, agentes falsos y resultados en una carpeta temporal."""
    from config import settings
    import backend.chat.herramientas as herr
    monkeypatch.setattr(settings, "hay_api_key", lambda: True)
    monkeypatch.setattr(herr, "ruta_ejecuciones", lambda nombre: tmp_path / nombre)
    falsos = AgentesFalsos()
    monkeypatch.setattr(herr, "correr_predictor", falsos.predictor)
    monkeypatch.setattr(herr, "correr_analista", falsos.analista)
    monkeypatch.setattr(herr, "correr_recomendador", falsos.recomendador)
    st.cache_resource.clear()
    yield falsos
    st.cache_resource.clear()


def _app(monkeypatch, guion=None):
    import backend.chat.asistente as asistente_mod
    if guion is not None:
        monkeypatch.setattr(asistente_mod, "completar_litellm", LLMFalso(guion))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_sin_clave_muestra_aviso_y_desactiva_el_chat(entorno, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "hay_api_key", lambda: False)
    at = _app(monkeypatch)
    assert any("GROQ_API_KEY" in i.value for i in at.info)
    assert at.chat_input[0].disabled


def test_bienvenida_con_ejemplos(entorno, monkeypatch):
    at = _app(monkeypatch)
    assert len([b for b in at.button if b.key and b.key.startswith("ejemplo_")]) == 4
    assert not at.chat_input[0].disabled


def test_pregunta_de_riesgo_activa_el_predictor(entorno, monkeypatch):
    at = _app(monkeypatch, [
        respuesta(llamadas=[llamada("consultar_predictor", {"customer_id": GRAY_ID})], tokens=120),
        respuesta("**Gray** tiene un 17,37 % de probabilidad de churn.", tokens=80),
    ])
    at.chat_input[0].set_value("¿Qué riesgo tiene Gray?").run()
    assert not at.exception, at.exception
    assert any("17,37" in m.value for m in at.markdown)
    assert any("Herramientas usadas (1)" in e.label for e in at.expander)
    assert entorno.llamadas == [('Predictor', 'Gray')]
    assert at.session_state.asistente.tokens_llm == 200
    assert at.session_state.asistente.tokens_agentes == 2000


def test_boton_de_ejemplo(entorno, monkeypatch):
    at = _app(monkeypatch, [respuesta("Respuesta al ejemplo.")])
    at.button(key="ejemplo_3").click().run()
    assert not at.exception, at.exception
    assert any("Respuesta al ejemplo." in m.value for m in at.markdown)


def test_confirmacion_y_tarjeta_de_recomendacion(entorno, monkeypatch):
    at = _app(monkeypatch, [
        respuesta(llamadas=[llamada("recomendar", {"customer_id": GRAY_ID})]),
        respuesta("Propongo una campaña de tarjeta por contact center."),
    ])
    at.chat_input[0].set_value("¿Qué hacemos con Gray?").run()
    assert any("Predictor, Analista, Recomendador" in w.value for w in at.warning)
    assert entorno.llamadas == []
    at.button(key="confirmar_si").click().run()
    assert not at.exception, at.exception
    assert [a for a, _ in entorno.llamadas] == ['Predictor', 'Analista', 'Recomendador']
    assert any("Recomendación del Agente Recomendador" in m.value for m in at.markdown)
    assert at.session_state.asistente.tokens_agentes == 39000
    assert not at.warning


def test_cancelar(entorno, monkeypatch):
    at = _app(monkeypatch, [
        respuesta(llamadas=[llamada("recomendar", {"customer_id": GRAY_ID})]),
        respuesta("Vale, no lo ejecuto."),
    ])
    at.chat_input[0].set_value("¿Qué hacemos con Gray?").run()
    at.button(key="confirmar_no").click().run()
    assert entorno.llamadas == [] and any("no lo ejecuto" in m.value for m in at.markdown)


def test_nueva_conversacion(entorno, monkeypatch):
    at = _app(monkeypatch, [respuesta("hola")])
    at.chat_input[0].set_value("hola").run()
    at.sidebar.button[0].click().run()
    assert at.session_state.historial == []
