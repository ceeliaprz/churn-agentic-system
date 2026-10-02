"""Flujo del pipeline completo sin llamar a Groq.

Se sustituye `ejecutar_con_reintentos` por un falso LLM que invoca las tools reales de cada
agente (como haría el modelo) y devuelve una salida pydantic plausible. Comprueba el
encadenado Predictor → Analista → router → Recomendador, el guardado por etapa y la
reanudación (solo se reintenta el Recomendador).
"""
from types import SimpleNamespace

import pytest

pytest.importorskip("crewai")
pytestmark = [pytest.mark.datos, pytest.mark.crewai]


class FalsoLLM:
    def __init__(self, fallar_recomendador=False):
        self.llamadas = []
        self.fallar_recomendador = fallar_recomendador

    async def __call__(self, crew, **_kw):
        from backend.agents.schemas import AnalisisCliente, PrediccionChurn, RecomendacionChurn
        tarea = crew.tasks[0]
        herramientas = {t.name: t for t in crew.agents[0].tools}
        clase = tarea.output_pydantic
        self.llamadas.append(clase.__name__)
        uso = SimpleNamespace(total_tokens=100)

        if clase is PrediccionChurn:
            datos = eval(tarea.description.split("Datos del cliente: ")[1].split("\n")[0])
            texto = herramientas["Predecir Churn de Cliente"].run(**datos)
            pct = float(texto.split("Probabilidad de churn: ")[1].rstrip("%"))
            salida = PrediccionChurn(cliente="X", churn_prediction=int(pct >= 50),
                                     churn_probability_pct=pct)
        elif clase is AnalisisCliente:
            senales = [herramientas["Productos"].run(num_productos=1),
                       herramientas["Actividad"].run(is_active_member=1, has_crcard=0)]
            salida = AnalisisCliente(cliente="X", senales=senales,
                                     orientacion="riesgo_por_encima_de_la_media", conclusion="c")
        else:
            if self.fallar_recomendador:
                return None
            salida = RecomendacionChurn(
                cliente="X", ruta="revision", variante_agente="cartera_masiva",
                resumen_cliente="r", resumen_riesgo="r", accion_principal="a",
                palanca_objetivo="tarjeta", canal="contact center", justificacion="j",
                limitaciones="l")
        return SimpleNamespace(pydantic=salida, token_usage=uso)


def test_pipeline_completo_y_reanudacion(monkeypatch, tmp_path, casos):
    import backend.pipeline.retries as retries
    from backend.pipeline.orchestrator import ejecutar_pipeline_cliente
    from backend.pipeline.retries import ejecutar_sync
    from backend.pipeline.storage import AlmacenResultados

    caso = casos['desacuerdo_franco']  # Gray: Ruta 3
    almacen = AlmacenResultados(tmp_path / "res.json")

    # 1ª ejecución: el Recomendador "agota los reintentos"
    falso = FalsoLLM(fallar_recomendador=True)
    monkeypatch.setattr(retries, "ejecutar_con_reintentos", falso)
    reg = ejecutar_sync(ejecutar_pipeline_cliente(
        "gray", caso['cliente'], caso['datos'], almacen, on_evento=None))
    assert falso.llamadas == ["PrediccionChurn", "AnalisisCliente", "RecomendacionChurn"]
    assert reg['recomendacion'] is None
    assert reg['ruta_info']['ruta'] == 3
    assert reg['prediccion']['churn_probability_pct'] == pytest.approx(17.37, abs=0.01)
    assert reg['tools_llamadas'] == ["Riesgo por número de productos",
                                     "Score de actividad y abandono por segmento"]
    assert AlmacenResultados(tmp_path / "res.json").get("gray") is not None

    # 2ª ejecución con forzar=False: solo se repite el Recomendador
    falso2 = FalsoLLM()
    monkeypatch.setattr(retries, "ejecutar_con_reintentos", falso2)
    reg2 = ejecutar_sync(ejecutar_pipeline_cliente(
        "gray", caso['cliente'], caso['datos'], AlmacenResultados(tmp_path / "res.json"),
        forzar=False, on_evento=None))
    assert falso2.llamadas == ["RecomendacionChurn"]
    assert reg2['recomendacion']['canal'] == "contact center"

    # 3ª ejecución: ya completo, no se llama a nada
    falso3 = FalsoLLM()
    monkeypatch.setattr(retries, "ejecutar_con_reintentos", falso3)
    ejecutar_sync(ejecutar_pipeline_cliente(
        "gray", caso['cliente'], caso['datos'], AlmacenResultados(tmp_path / "res.json"),
        forzar=False, on_evento=None))
    assert falso3.llamadas == []


def test_modo_determinista(casos):
    from backend.pipeline.orchestrator import ejecutar_caso_determinista
    caso = casos['balance_alto']
    r = ejecutar_caso_determinista(caso['cliente'], caso['datos'])
    assert r['ruta_info']['variante'] == 'alto_valor'
    assert r['senales'] == caso['senales']
    assert r['simulacion']['resultados'][0][0] == 'Llevar a 2 producto(s)'
    assert r['recomendacion_tradicional']['canal'] == 'gestor personal'


def test_reintento_no_duplica_senales_del_analista(monkeypatch, casos):
    """Si el Analista choca con el rate limit a mitad de tarea y se repite, balance_de_senales
    solo debe ver las tools del intento que termina (antes se acumulaban las de ambos)."""
    import backend.pipeline.retries as retries
    from backend.agents.schemas import AnalisisCliente
    from backend.pipeline.orchestrator import ejecutar_analista

    async def sin_espera(_s):
        return None
    monkeypatch.setattr(retries.asyncio, "sleep", sin_espera)
    intentos = []

    async def kickoff_falso(self, *a, **k):
        herramientas = {t.name: t for t in self.agents[0].tools}
        herramientas["Productos"].run(num_productos=1)                 # x1.37
        herramientas["Actividad"].run(is_active_member=1, has_crcard=0)  # x0.84 → producto 1.15
        intentos.append(1)
        if len(intentos) == 1:
            raise Exception("rate_limit_exceeded: Please try again in 1s")
        return SimpleNamespace(pydantic=AnalisisCliente(
            cliente="Gray", senales=["s"], orientacion="riesgo_en_linea_con_la_media", conclusion="c"),
            token_usage=SimpleNamespace(total_tokens=10))

    from crewai import Crew
    monkeypatch.setattr(Crew, "kickoff_async", kickoff_falso)
    caso = casos['desacuerdo_franco']
    r = retries.ejecutar_sync(ejecutar_analista(caso['cliente'], caso['datos'], on_evento=None))
    assert len(intentos) == 2
    assert len(r['senales_tools']) == 2                        # no 4
    assert r['orientacion_calculada'] == "riesgo_en_linea_con_la_media"   # 1.15; duplicado sería 1.32 → "encima"
