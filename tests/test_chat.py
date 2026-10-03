"""Asistente conversacional: herramientas (agentes del notebook) y bucle de function calling.

Sin Groq: los tres agentes CrewAI se sustituyen por falsos que devuelven lo mismo que
devolverían (formato de orchestrator.ejecutar_predictor / ejecutar_analista /
ejecutar_recomendador), y el LLM del Asistente por un LLM con guion.
"""
import json

import pytest

from tests.conftest import leer_json_notebook
from tests.llm_falso import LLMFalso, llamada, respuesta

pytestmark = pytest.mark.datos

GRAY_ID = 15656865        # perfil desacuerdo_franco (X_test_raw.loc[1198]) → Ruta 3
HSIUNG_ID = 15749540      # perfil balance_cero → Ruta 1


class AgentesFalsos:
    """Devuelven los datos reales de casos_recomendador.json como si fueran los agentes."""

    def __init__(self):
        self.casos = {c['cliente']: c for c in leer_json_notebook("casos_recomendador.json").values()}
        self.llamadas = []
        self.motivos = []

    def predictor(self, nombre, datos, ctx, on_evento):
        self.llamadas.append(('Predictor', nombre))
        return {'prediccion': {'cliente': nombre, **self.casos[nombre]['prediccion']},
                'tokens_predictor': 2000}

    def analista(self, nombre, datos, ctx, on_evento, motivo=None):
        self.llamadas.append(('Analista', nombre))
        self.motivos.append(motivo)
        caso = self.casos[nombre]
        return {'senales': caso['senales'], 'conclusion_analista': 'conclusión',
                'orientacion_llm': 'riesgo_por_encima_de_la_media',
                'orientacion_calculada': caso['orientacion_calculada'], 'tokens_analista': 12000,
                'tools_llamadas': ['Riesgo por número de productos'], 'senales_tools': caso['senales']}

    def recomendador(self, nombre, datos, prediccion, senales, ruta_info, ctx, conclusion, on_evento):
        self.llamadas.append(('Recomendador', nombre))
        assert conclusion == 'conclusión'
        return {'recomendacion': {'accion_principal': 'Campaña de tarjeta', 'canal': 'contact center',
                                  'variante_agente': ruta_info['variante']},
                'tokens_recomendador': 25000}


@pytest.fixture
def agentes(monkeypatch):
    import backend.chat.herramientas as mod
    falsos = AgentesFalsos()
    monkeypatch.setattr(mod, "correr_predictor", falsos.predictor)
    monkeypatch.setattr(mod, "correr_analista", falsos.analista)
    monkeypatch.setattr(mod, "correr_recomendador", falsos.recomendador)
    return falsos


@pytest.fixture
def herramientas(tmp_path, x_test):
    from backend.chat.herramientas import HerramientasChat
    from backend.pipeline.storage import AlmacenResultados
    return HerramientasChat(x_test=x_test, almacen=AlmacenResultados(tmp_path / "pipeline.json"))


# --- Herramientas ----------------------------------------------------------------------
def test_esquemas():
    from backend.chat.herramientas import ESQUEMAS, HerramientasChat
    nombres = [e['function']['name'] for e in ESQUEMAS]
    assert nombres == ["buscar_cliente", "consultar_predictor", "consultar_analista", "recomendar",
                       "resumen_cartera"]
    for e in ESQUEMAS:
        assert callable(getattr(HerramientasChat, e['function']['name']))
        for p in e['function']['parameters']['properties'].values():
            assert p['type'] in ('integer', 'string', 'boolean'), "Groq solo acepta escalares"


def test_buscar_cliente(herramientas):
    r = json.loads(herramientas.buscar_cliente("Gray"))
    assert r['total_coincidencias'] == 2
    assert {c['perfil_congelado'] for c in r['clientes']} == {None, 'desacuerdo_franco'}
    assert json.loads(herramientas.buscar_cliente("desacuerdo_franco"))['clientes'][0]['customer_id'] == GRAY_ID
    assert json.loads(herramientas.buscar_cliente(str(HSIUNG_ID)))['clientes'][0]['apellido'] == "Hsiung"


def test_predictor_se_ejecuta_una_vez(herramientas, agentes):
    r = json.loads(herramientas.consultar_predictor(GRAY_ID))
    assert r['churn_probability_pct'] == 17.37 and r['tokens_agentes'] == 2000
    assert r['router'] is None                     # sin Analista todavía no hay ruta
    again = json.loads(herramientas.consultar_predictor(GRAY_ID))
    assert again['resultado_guardado'] and again['tokens_agentes'] == 0
    assert agentes.llamadas == [('Predictor', 'Gray')]


def test_analista_y_router(herramientas, agentes):
    herramientas.consultar_predictor(GRAY_ID)
    r = json.loads(herramientas.consultar_analista(GRAY_ID))
    assert r['orientacion_calculada'] == 'riesgo_por_encima_de_la_media'
    assert r['router']['ruta'] == 3 and r['router']['variante'] == 'cartera_masiva'
    json.loads(herramientas.consultar_analista(GRAY_ID, repetir=True))
    assert [a for a, _ in agentes.llamadas] == ['Predictor', 'Analista', 'Analista']


def test_reanalisis_con_motivo(herramientas, agentes):
    """El motivo llega al Analista solo al repetir, y se conserva el análisis anterior."""
    primero = json.loads(herramientas.consultar_analista(GRAY_ID))
    assert primero['analisis_anterior'] is None and primero['motivo_reanalisis'] is None
    motivo = "No consultó el contexto demográfico y el cliente es alemán de 69 años."
    segundo = json.loads(herramientas.consultar_analista(GRAY_ID, repetir=True, motivo=motivo))
    assert agentes.motivos == [None, motivo]
    assert segundo['motivo_reanalisis'] == motivo
    assert segundo['analisis_anterior']['herramientas_invocadas'] == ['Riesgo por número de productos']
    # Un motivo sin repetir=true también cuenta como reanálisis (y pide confirmación)
    assert herramientas.coste_estimado('consultar_analista', {'customer_id': GRAY_ID, 'motivo': motivo})
    json.loads(herramientas.consultar_analista(GRAY_ID, motivo="Otra vez"))
    assert agentes.motivos[-1] == "Otra vez"
    # Leer lo guardado no ejecuta al Analista ni cambia el motivo
    guardado = json.loads(herramientas.consultar_analista(GRAY_ID))
    assert guardado['resultado_guardado'] and len(agentes.motivos) == 3


def test_tarea_del_analista_con_motivo():
    from backend.agents.analyst import descripcion_tarea_analisis, texto_reanalisis
    datos = {'Age': 69, 'Geography': 'Germany'}
    base = descripcion_tarea_analisis("Gray", datos)
    assert "segundo análisis" not in base
    assert descripcion_tarea_analisis("Gray", datos, "   ") == base
    con = descripcion_tarea_analisis("Gray", datos, "Falta el contexto demográfico.")
    assert con == base + texto_reanalisis("Falta el contexto demográfico.")
    assert "Falta el contexto demográfico." in con


def test_resumen_cartera(herramientas, agentes):
    """Vista de cartera sin agentes: mismas cifras que la cobertura de la memoria."""
    r = json.loads(herramientas.resumen_cartera())
    assert r['total_coincidencias'] == 2000
    assert r['reparto_rutas'] == {'Ruta 1': 972, 'Ruta 2': 569, 'Ruta 3': 459}
    assert agentes.llamadas == []
    probs = [c['prob_pct'] for c in r['clientes']]
    assert len(probs) == 10 and probs == sorted(probs, reverse=True)
    assert r['clientes'][0]['apellido'] == 'Chiu'
    assert {c['origen'] for c in r['clientes']} == {'cálculo 4 herramientas'}
    assert r['clientes'][0]['orientacion'] in ('por encima', 'en línea', 'por debajo')

    de = json.loads(herramientas.resumen_cartera(pais="Alemania", ruta=3, limite=3))
    assert de['total_coincidencias'] == 141 and len(de['clientes']) == 3
    assert de['clientes'][0]['apellido'] == 'Houghton'
    # Con filtro de país y ruta no se repiten esos campos en cada cliente (ahorra tokens)
    assert all('pais' not in c and 'ruta' not in c for c in de['clientes'])
    assert de['reparto_rutas'] == {'Ruta 3': 141}
    from config import settings
    assert len(json.loads(herramientas.resumen_cartera(limite=500))['clientes']) == settings.CHAT_CARTERA_LIMITE_MAX
    assert 'error' in json.loads(herramientas.resumen_cartera(ruta=7))

    # Variante: misma regla que el router (percentil de saldo >= 75)
    riesgo_alto = json.loads(herramientas.resumen_cartera(ruta=2, limite=50))['clientes']
    assert all(c['variante'] in ('alto_valor', 'cartera_masiva') for c in riesgo_alto)
    assert json.loads(herramientas.resumen_cartera(ruta=1, limite=1))['clientes'][0]['variante'] is None


def test_resumen_cartera_es_compacto(herramientas):
    """La lista máxima cabe en el presupuesto del historial."""
    from config import settings
    texto = herramientas.resumen_cartera(limite=500)
    assert len(texto) // settings.CHAT_CARACTERES_POR_TOKEN < settings.CHAT_PRESUPUESTO_TOKENS_HISTORIAL


def test_resumen_cartera_usa_el_analisis_real(herramientas, agentes, monkeypatch):
    """Si el Agente Analista ya analizó a un cliente, la vista usa ese análisis y no el cálculo
    con las 4 herramientas."""
    base = json.loads(herramientas.resumen_cartera(pais="Germany", ruta=3, limite=50))
    assert base['clientes_con_analisis_real'] == 0

    # Analista real que solo consulta 2 herramientas y da "en línea con la media" (como en 8.1)
    def analista_parcial(nombre, datos, ctx, on_evento, motivo=None):
        return {'senales': ['productos', 'actividad'], 'conclusion_analista': 'c',
                'orientacion_llm': 'riesgo_por_encima_de_la_media',
                'orientacion_calculada': 'riesgo_en_linea_con_la_media', 'tokens_analista': 9000,
                'tools_llamadas': ['Riesgo por número de productos',
                                   'Score de actividad y abandono por segmento'],
                'senales_tools': []}
    import backend.chat.herramientas as mod
    monkeypatch.setattr(mod, "correr_analista", analista_parcial)
    herramientas.consultar_predictor(GRAY_ID)
    herramientas.consultar_analista(GRAY_ID)

    r = json.loads(herramientas.resumen_cartera(pais="Germany", ruta=3, limite=50))
    assert r['clientes_con_analisis_real'] == 1
    from backend.decision.cartera import cargar_cobertura, construir_vista_cartera
    v = construir_vista_cartera(cargar_cobertura(), herramientas.x_test, herramientas.ctx.base.df_train,
                                herramientas.almacen.datos)
    fila = v[v['customer_id'] == GRAY_ID].iloc[0]
    assert fila['orientacion_calculada'] == 'riesgo_en_linea_con_la_media'
    assert fila['origen'] == "análisis del Agente Analista (2 de 4 herramientas)"
    assert fila['ruta'] == 3      # la ruta no cambia: probabilidad < 30 % y Analista no "por debajo"
    # Gray (17 %) queda lejos del top 50 de Alemania-Ruta 3: se amplía el límite para verlo
    from config import settings
    monkeypatch.setattr(settings, "CHAT_CARTERA_LIMITE_MAX", 200)
    todos = json.loads(herramientas.resumen_cartera(pais="Germany", ruta=3, limite=200))['clientes']
    gray = next(c for c in todos if c['customer_id'] == GRAY_ID)
    assert gray['origen'] == 'Agente Analista (2/4)' and gray['orientacion'] == 'en línea'


def test_resumen_cartera_analista_sin_predictor(herramientas, agentes, monkeypatch):
    """Caso Houghton: el chat ejecutó solo al Analista (sin Predictor, así que el registro no
    tiene ruta_info). La vista debe usar igualmente su análisis, con la ruta calculada con la
    probabilidad del modelo. Aquí con Gray (Alemania, Ruta 3), que tiene agente falso."""
    from config import settings
    herramientas.consultar_analista(GRAY_ID)
    assert 'ruta_info' not in herramientas.almacen.get(f"customer_{GRAY_ID}")
    monkeypatch.setattr(settings, "CHAT_CARTERA_LIMITE_MAX", 200)
    r = json.loads(herramientas.resumen_cartera(pais="Germany", ruta=3, limite=200))
    assert r['clientes_con_analisis_real'] == 1 and r['total_coincidencias'] == 141
    gray = next(c for c in r['clientes'] if c['customer_id'] == GRAY_ID)
    assert gray['origen'] == 'Agente Analista (1/4)' and gray['orientacion'] == 'por encima'


def test_recomendar_completa_la_cadena(herramientas, agentes):
    r = json.loads(herramientas.recomendar(GRAY_ID))
    assert r['etapas_ejecutadas'] == ['Predictor', 'Analista', 'Recomendador']
    assert r['router']['ruta'] == 3 and r['recomendacion']['canal'] == 'contact center'
    assert r['tokens_agentes'] == 39000
    # Guardado con el formato del pipeline (8.1), reutilizable por scripts/run_pipeline.py
    reg = herramientas.almacen.get(f"customer_{GRAY_ID}")
    assert set(reg) >= {'cliente', 'datos', 'prediccion', 'senales', 'conclusion_analista',
                        'orientacion_llm', 'orientacion_calculada', 'ruta_info', 'recomendacion'}
    otra = json.loads(herramientas.recomendar(GRAY_ID))
    assert otra['resultado_guardado'] and otra['tokens_agentes'] == 0 and len(agentes.llamadas) == 3


def test_recomendar_ruta_1_no_activa_recomendador(herramientas, agentes):
    r = json.loads(herramientas.recomendar(HSIUNG_ID))
    assert r['router']['ruta'] == 1 and r['recomendacion'] is None
    assert [a for a, _ in agentes.llamadas] == ['Predictor', 'Analista']


def test_coste_estimado(herramientas, agentes):
    assert herramientas.coste_estimado('consultar_predictor', {'customer_id': GRAY_ID}) is None
    c = herramientas.coste_estimado('recomendar', {'customer_id': GRAY_ID})
    assert c['agentes'] == ['Predictor', 'Analista', 'Recomendador'] and c['tokens_estimados'] == 32500
    herramientas.consultar_predictor(GRAY_ID)
    assert herramientas.coste_estimado('recomendar', {'customer_id': GRAY_ID})['agentes'] == ['Analista', 'Recomendador']
    herramientas.consultar_analista(GRAY_ID)
    assert herramientas.coste_estimado('consultar_analista', {'customer_id': GRAY_ID}) is None
    assert herramientas.coste_estimado('consultar_analista', {'customer_id': GRAY_ID, 'repetir': True})
    herramientas.recomendar(HSIUNG_ID)
    assert herramientas.coste_estimado('recomendar', {'customer_id': HSIUNG_ID}) is None   # Ruta 1


def test_prompt_prohibe_recomendar_sin_recomendador():
    from backend.chat.asistente import prompt_sistema
    assert "Nunca propongas, sugieras ni anticipes acciones de retención" in prompt_sistema()
    assert "comparación breve (1–2 frases) con analisis_anterior" in prompt_sistema()
    assert "Llama a recomendar solo si el usuario pide qué hacer" in prompt_sistema()


def test_errores(herramientas, agentes, monkeypatch):
    import backend.chat.herramientas as mod
    assert 'error' in json.loads(herramientas.consultar_predictor(1))
    assert 'error' in json.loads(herramientas.ejecutar("no_existe", {}))
    assert 'error' in json.loads(herramientas.ejecutar("consultar_predictor", {"id": 3}))
    monkeypatch.setattr(mod, "correr_predictor", lambda *a: None)       # agotó reintentos
    assert 'error' in json.loads(herramientas.consultar_predictor(GRAY_ID))

    class SinMensaje(Exception):
        pass

    def falla(*a, **kw):
        raise SinMensaje()
    monkeypatch.setattr(mod, "correr_analista", falla)
    # Una excepción sin texto no debe dar un error vacío: se indica al menos su tipo
    assert 'SinMensaje' in json.loads(herramientas.ejecutar("consultar_analista", {"customer_id": GRAY_ID}))['error']


def test_pipeline_cli_reanuda_tras_registro_parcial_del_chat(herramientas, agentes, monkeypatch):
    """Si el chat solo ejecutó el Predictor, run_pipeline (forzar=False) no debe romperse."""
    import backend.pipeline.retries as retries
    from backend.pipeline.orchestrator import ejecutar_pipeline_cliente
    herramientas.consultar_predictor(GRAY_ID)
    llamado = []

    async def falso(crew, **kw):
        llamado.append(crew.tasks[0].output_pydantic.__name__)
        return None
    monkeypatch.setattr(retries, "ejecutar_con_reintentos", falso)
    reg = herramientas.almacen.get(f"customer_{GRAY_ID}")
    retries.ejecutar_sync(ejecutar_pipeline_cliente(f"customer_{GRAY_ID}", reg['cliente'], reg['datos'],
                                                    herramientas.almacen, forzar=False, on_evento=None))
    assert llamado == ['PrediccionChurn']        # se trató como no empezado, sin KeyError


# --- Bucle del Asistente ------------------------------------------------------------------
@pytest.fixture
def asistente_con(herramientas, agentes):
    from backend.chat.asistente import AsistenteChat

    def crear(guion):
        llm = LLMFalso(guion)
        return AsistenteChat(herramientas=herramientas, completar=llm), llm
    return crear


def test_respuesta_directa(asistente_con):
    a, llm = asistente_con([respuesta("Hola, ¿en qué te ayudo?", tokens=50)])
    t = a.responder("hola")
    assert t.texto == "Hola, ¿en qué te ayudo?" and t.llamadas == [] and t.tokens == 50
    enviado = llm.recibido[0]
    assert enviado['messages'][0]['role'] == 'system'
    assert enviado['tool_choice'] == 'auto' and enviado['model'] == "groq/openai/gpt-oss-20b"


def test_riesgo_activa_solo_el_predictor(asistente_con, agentes):
    a, llm = asistente_con([
        respuesta(llamadas=[llamada("buscar_cliente", {"consulta": "desacuerdo_franco"})]),
        respuesta(llamadas=[llamada("consultar_predictor", {"customer_id": GRAY_ID})]),
        respuesta("Gray tiene un 17,37 % de probabilidad de churn."),
    ])
    t = a.responder("¿Qué riesgo tiene Gray?")
    assert [ll['nombre'] for ll in t.llamadas] == ["buscar_cliente", "consultar_predictor"]
    assert agentes.llamadas == [('Predictor', 'Gray')]
    assert a.tokens_agentes == 2000 and t.pendiente is None
    ultimo = llm.recibido[-1]['messages']
    assert [m['role'] for m in ultimo] == ['system', 'user', 'assistant', 'tool', 'assistant', 'tool']
    assert ultimo[3]['tool_call_id'] == ultimo[2]['tool_calls'][0]['id']


def test_recomendar_pide_confirmacion(asistente_con, agentes):
    a, llm = asistente_con([
        respuesta(llamadas=[llamada("recomendar", {"customer_id": GRAY_ID})]),
        respuesta("Propongo una campaña de tarjeta por contact center."),
    ])
    t = a.responder("¿Qué hacemos con Gray?")
    assert t.pendiente['agentes'] == ['Predictor', 'Analista', 'Recomendador']
    assert agentes.llamadas == [] and len(llm.recibido) == 1          # nada gastado todavía
    t2 = a.confirmar(True)
    assert [x for x, _ in agentes.llamadas] == ['Predictor', 'Analista', 'Recomendador']
    assert t2.texto.startswith("Propongo") and a.pendiente is None
    assert a.tokens_agentes == 39000


def test_cancelar(asistente_con, agentes):
    a, llm = asistente_con([
        respuesta(llamadas=[llamada("consultar_analista", {"customer_id": GRAY_ID}),
                            llamada("recomendar", {"customer_id": GRAY_ID}, "c2")]),
        respuesta("De acuerdo, no lo ejecuto."),
    ])
    assert a.responder("analiza y recomienda para Gray").pendiente
    t = a.confirmar(False)
    assert t.texto == "De acuerdo, no lo ejecuto." and agentes.llamadas == []
    tools = [m for m in llm.recibido[-1]['messages'] if m['role'] == 'tool']
    assert len(tools) == 2 and all(json.loads(m['content'])['cancelado'] for m in tools)


def test_nuevo_mensaje_cancela_pendiente(asistente_con):
    a, llm = asistente_con([
        respuesta(llamadas=[llamada("recomendar", {"customer_id": GRAY_ID})]),
        respuesta("Vale, hablemos de otra cosa."),
    ])
    a.responder("recomienda para Gray")
    t = a.responder("mejor dime otra cosa")
    assert a.pendiente is None and t.texto.startswith("Vale")
    assert [m['role'] for m in llm.recibido[-1]['messages']] == ['system', 'user', 'assistant', 'tool', 'user']


def test_ruta_1_no_pide_confirmacion_si_ya_hay_analisis(asistente_con, herramientas, agentes):
    herramientas.consultar_predictor(HSIUNG_ID)
    herramientas.consultar_analista(HSIUNG_ID)
    a, _ = asistente_con([respuesta(llamadas=[llamada("recomendar", {"customer_id": HSIUNG_ID})]),
                          respuesta("El router cierra el caso.")])
    t = a.responder("recomienda para Hsiung")
    assert t.pendiente is None and json.loads(t.llamadas[0]['resultado'])['router']['ruta'] == 1


def test_argumentos_json_invalidos(asistente_con):
    a, _ = asistente_con([respuesta(llamadas=[llamada("consultar_predictor", "{no es json")]),
                          respuesta("Perdona, repito.")])
    assert 'error' in json.loads(a.responder("riesgo").llamadas[0]['resultado'])


def test_reintento_por_rate_limit(asistente_con, monkeypatch):
    import backend.chat.asistente as mod
    esperas = []
    monkeypatch.setattr(mod.time, "sleep", esperas.append)
    a, _ = asistente_con([Exception("rate_limit_exceeded: Please try again in 7.5s"), respuesta("Ya está.")])
    eventos = []
    assert a.responder("hola", on_evento=eventos.append).texto == "Ya está."
    assert esperas == [60.0] and "Límite de Groq" in eventos[0]


def test_error_no_recuperable_retira_el_mensaje(asistente_con):
    a, _ = asistente_con([ValueError("clave inválida")])
    with pytest.raises(ValueError):
        a.responder("hola")
    assert a.mensajes == []


def test_limite_de_pasos(asistente_con, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "CHAT_MAX_PASOS", 2)
    a, _ = asistente_con([respuesta(llamadas=[llamada("buscar_cliente", {"consulta": "Sun"}, f"c{i}")])
                          for i in range(2)])
    assert "límite de pasos" in a.responder("bucle").texto


def test_recorte_de_historial_empieza_en_usuario(asistente_con, monkeypatch):
    from config import settings
    monkeypatch.setattr(settings, "CHAT_MAX_MENSAJES", 4)
    guion = []
    for i in range(3):
        guion += [respuesta(llamadas=[llamada("buscar_cliente", {"consulta": "Sun"}, f"c{i}")]),
                  respuesta(f"r{i}")]
    a, llm = asistente_con(guion)
    for i in range(3):
        a.responder(f"pregunta {i}")
    for enviado in llm.recibido:
        mensajes = enviado['messages']
        assert mensajes[1]['role'] == 'user'
        ids = {tc['id'] for m in mensajes if m['role'] == 'assistant' for tc in m.get('tool_calls') or []}
        assert all(m['tool_call_id'] in ids for m in mensajes if m['role'] == 'tool')


# --- Recorte del historial (límite de 8.000 tokens por minuto de Groq) ---------------------
DEMASIADO_GRANDE = Exception("RateLimitError: Request too large for model `openai/gpt-oss-20b` "
                             "... TPM: Limit 8000, Requested 8298, please reduce your message size")


def _historial_largo(a, turnos=4, tamano=6000):
    """Simula turnos anteriores con resultados de herramientas grandes (listas de cartera)."""
    for i in range(turnos):
        a.mensajes += [
            {"role": "user", "content": f"pregunta {i}"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"c{i}", "type": "function",
                 "function": {"name": "resumen_cartera", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": f"c{i}", "content": "x" * tamano},
            {"role": "assistant", "content": f"respuesta {i}"},
        ]


def test_historial_respeta_el_presupuesto(asistente_con):
    from backend.chat.asistente import RESULTADO_RECORTADO
    from config import settings
    a, _ = asistente_con([])
    _historial_largo(a)
    enviados = a._mensajes_para_llm()[1:]
    assert a._tokens_estimados(enviados) <= settings.CHAT_PRESUPUESTO_TOKENS_HISTORIAL
    assert enviados[0]['role'] == 'user'                      # no rompe pares tool_call/tool
    tools = [m for m in enviados if m['role'] == 'tool']
    # Los resultados antiguos se recortan; el del penúltimo turno se conserva entero
    assert tools[-1]['content'] == "x" * 6000
    assert all(m['content'] == RESULTADO_RECORTADO for m in tools[:-2])
    assert a.mensajes[2]['content'] == "x" * 6000              # el historial real no se toca


def test_historial_corto_no_se_recorta(asistente_con):
    a, _ = asistente_con([])
    _historial_largo(a, turnos=2, tamano=500)
    assert a._mensajes_para_llm()[1:] == a.mensajes


def test_peticion_demasiado_grande_reintenta_recortando(asistente_con):
    from backend.chat.asistente import RESULTADO_RECORTADO
    a, llm = asistente_con([DEMASIADO_GRANDE, respuesta("Hecho.")])
    _historial_largo(a, turnos=2, tamano=3000)
    avisos = []
    t = a.responder("analiza al primero de la lista", on_evento=avisos.append)
    assert t.texto == "Hecho." and len(llm.recibido) == 2
    assert any("demasiado grande" in m for m in avisos)
    # Reintento inmediato y agresivo: solo se conservan los resultados del último turno
    tools = [m for m in llm.recibido[1]['messages'] if m['role'] == 'tool']
    assert tools and all(m['content'] == RESULTADO_RECORTADO for m in tools)


def test_peticion_demasiado_grande_dos_veces_da_error_claro(asistente_con, monkeypatch):
    import backend.chat.asistente as mod
    monkeypatch.setattr(mod.time, "sleep", lambda s: pytest.fail("no debe esperar"))
    a, llm = asistente_con([DEMASIADO_GRANDE, DEMASIADO_GRANDE])
    with pytest.raises(RuntimeError, match="Nueva conversación"):
        a.responder("hola")
    assert len(llm.recibido) == 2 and a.mensajes == []
