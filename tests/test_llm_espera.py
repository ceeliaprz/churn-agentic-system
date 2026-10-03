"""LLMConEspera: ante un rate limit por minuto se repite solo la llamada al LLM, no la tarea.
Sin Groq: se sustituye LLM.call de CrewAI por una función con guion."""
import pytest

pytestmark = pytest.mark.crewai


class RateLimitError(Exception):
    """Mismo nombre de tipo que la excepción de LiteLLM."""


TPM = RateLimitError('GroqException - {"error":{"message":"Rate limit reached for model '
                     '`openai/gpt-oss-20b` ... on tokens per minute (TPM): Limit 8000, Used 6772, '
                     'Requested 2405. Please try again in 8.8275s."}}')


def test_espera_sugerida():
    from backend.agents.llm import espera_para_reintentar_llamada as espera
    assert espera(TPM) == pytest.approx(10.8275)
    assert espera(RateLimitError("Please try again in 1m2.5s")) == pytest.approx(64.5)
    assert espera(RateLimitError("rate limit")) == 65.0               # sin sugerencia: el máximo
    assert espera(RateLimitError("Request too large ... reduce your message size")) is None
    assert espera(RateLimitError("Limit 200000 on tokens per day (TPD) ... try again in 9m3s")) is None
    assert espera(ValueError("otro error")) is None


@pytest.fixture
def llm_con_guion(monkeypatch):
    from crewai import LLM
    import backend.agents.llm as mod
    esperas = []
    monkeypatch.setattr(mod.time, "sleep", esperas.append)

    def crear(guion):
        llamadas = []

        def call_falso(self, messages, *a, **kw):
            llamadas.append(messages)
            siguiente = guion.pop(0)
            if isinstance(siguiente, Exception):
                raise siguiente
            return siguiente
        monkeypatch.setattr(LLM, "call", call_falso)
        return mod.crear_llm_analista(), llamadas
    return crear, esperas


def test_repite_solo_la_llamada(llm_con_guion):
    from backend.pipeline.retries import aviso_actual
    crear, esperas = llm_con_guion
    llm, llamadas = crear([TPM, TPM, "respuesta final"])
    avisos = []
    aviso_actual.set(avisos.append)
    mensajes = [{"role": "user", "content": "resultado de las 4 tools"}]
    assert llm.call(mensajes) == "respuesta final"
    assert llamadas == [mensajes] * 3                   # la misma llamada, con lo ya hecho
    assert esperas == [pytest.approx(10.8275)] * 2
    assert len(avisos) == 2 and "repito solo la última llamada" in avisos[0]


def test_no_repite_lo_que_esperar_no_arregla(llm_con_guion):
    crear, esperas = llm_con_guion
    llm, llamadas = crear([RateLimitError("Request too large for model")])
    with pytest.raises(RateLimitError):
        llm.call("hola")
    assert len(llamadas) == 1 and esperas == []


def test_se_rinde_tras_el_maximo(llm_con_guion):
    from config import settings
    crear, esperas = llm_con_guion
    llm, llamadas = crear([TPM] * settings.MAX_ESPERAS_POR_LLAMADA)
    with pytest.raises(RateLimitError):
        llm.call("hola")
    assert len(llamadas) == settings.MAX_ESPERAS_POR_LLAMADA
    assert len(esperas) == settings.MAX_ESPERAS_POR_LLAMADA - 1


def test_configuracion_y_copia():
    import copy
    from backend.agents.llm import LLMConEspera, crear_llm_analista, crear_llm_predictor
    from config import settings
    llm = crear_llm_analista()
    assert isinstance(llm, LLMConEspera)
    assert llm.model == settings.LLM_ANALISTA['model'] and llm.temperature == 0.0
    assert isinstance(copy.copy(llm), LLMConEspera)
    assert isinstance(crear_llm_predictor(), LLMConEspera)


def test_un_aviso_que_falla_no_tumba_la_llamada(llm_con_guion):
    """Caso real: Streamlit lanza NoSessionContext (sin mensaje) si se escribe desde el hilo de
    CrewAI. El aviso es informativo y la llamada debe seguir."""
    from backend.pipeline.retries import aviso_actual
    crear, esperas = llm_con_guion
    llm, _ = crear([TPM, "ok"])

    class NoSessionContext(Exception):
        pass

    def aviso_roto(_mensaje):
        raise NoSessionContext()
    aviso_actual.set(aviso_roto)
    assert llm.call("hola") == "ok" and len(esperas) == 1
