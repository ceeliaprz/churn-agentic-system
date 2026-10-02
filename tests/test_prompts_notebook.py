"""Prompts, backstories, docstrings y esquemas idénticos al notebook.

Ejecuta las celdas originales con CrewAI sustituido por stubs y compara con el repo.
Requiere crewai instalado (los módulos backend.agents lo importan), pero NO llama a Groq.
"""
import inspect

import pytest

from backend.analytics import segments
from tests.conftest import celda_notebook

pytest.importorskip("crewai")
pytestmark = [pytest.mark.datos, pytest.mark.crewai]


def _limpio(doc: str) -> str:
    return inspect.cleandoc(doc or "")


def test_docstring_simulacion(ns_stub, base):
    from backend.agents import tools as tools_repo
    from backend.ml.predictor import predict_churn
    ns = {**ns_stub, "predict_churn": predict_churn, "_prod_estable": base.prod_estable}
    exec(celda_notebook('@tool("Simulación reanalisis")'), ns)
    assert _limpio(ns["simular_reanalisis_tool"].__doc__) == _limpio(tools_repo.DOC_SIMULACION)



def test_tool_predictor_docstring_y_texto(ns_stub, modelo, casos):
    from backend.agents import tools as tools_repo
    from backend.ml.predictor import predict_churn
    ns = {**ns_stub, "predict_churn": predict_churn}
    exec(celda_notebook('@tool("Predecir Churn de Cliente")'), ns)
    assert _limpio(ns["predecir_churn_tool"].__doc__) == _limpio(tools_repo.DOC_PREDICTOR)
    for caso in casos.values():
        assert (ns["predecir_churn_tool"](**caso['datos'])
                == tools_repo.texto_prediccion(modelo.predict_churn(caso['datos'])))


def test_agentes_y_tarea_predictor(ns_stub, casos):
    from backend.agents import predictor as pred_repo
    ns = {**ns_stub, "predecir_churn_tool": "TOOL"}
    exec(celda_notebook("agente_predictor = Agent("), ns)
    exec(celda_notebook("def crear_tarea_prediccion"), ns)
    ag = ns["agente_predictor"]
    assert (ag["role"], ag["goal"], ag["backstory"]) == (pred_repo.ROLE, pred_repo.GOAL, pred_repo.BACKSTORY)
    for caso in casos.values():
        t = ns["crear_tarea_prediccion"](caso['cliente'], caso['datos'])
        assert t["description"] == pred_repo.descripcion_tarea_prediccion(caso['cliente'], caso['datos'])
        assert t["expected_output"] == pred_repo.EXPECTED_OUTPUT_PREDICCION
    from backend.agents.schemas import PrediccionChurn
    assert ns["PrediccionChurn"].model_json_schema() == PrediccionChurn.model_json_schema()


def test_agente_y_tarea_analista(ns_stub, base, casos):
    from backend.agents import analyst as an_repo
    ns = {**ns_stub, "referencias_texto": base.referencias_texto,
          "calcular_percentil_balance": 1, "consultar_riesgo_productos": 2,
          "consultar_actividad_vinculacion": 3, "consultar_contexto_demografico": 4}
    exec(celda_notebook("agente_analista = Agent("), ns)
    exec(celda_notebook("def analizar_cliente"), ns)
    ag = ns["agente_analista"]
    assert ag["role"] == an_repo.ROLE and ag["goal"] == an_repo.GOAL
    assert ag["backstory"] == an_repo.backstory_analista(base.referencias_texto)
    assert ag["max_iter"] == 8
    for caso in casos.values():
        t = ns["analizar_cliente"](caso['cliente'], caso['datos'])
        assert t["description"] == an_repo.descripcion_tarea_analisis(caso['cliente'], caso['datos'])
        assert t["expected_output"] == an_repo.EXPECTED_OUTPUT_ANALISIS
    from backend.agents.schemas import AnalisisCliente
    assert ns["AnalisisCliente"].model_json_schema() == AnalisisCliente.model_json_schema()


def test_agentes_y_tarea_recomendador(ns_stub, base, casos):
    from backend.agents import recommender as rec_repo
    from backend.decision.router import enrutar_caso
    ns = {**ns_stub, "simular_reanalisis_tool": "TOOL"}
    exec(celda_notebook("from typing import List, Literal"), ns)   # imports de Literal/Field
    exec(celda_notebook("_BACKSTORY_COMUN = ("), ns)
    exec(celda_notebook("def crear_tarea_recomendacion"), ns)
    for variante in ("alto_valor", "cartera_masiva"):
        ag = ns["crear_agente_recomendador"](variante)
        assert ag["backstory"] == f"{rec_repo.enfoque_variante(variante)}\n\n{rec_repo._BACKSTORY_COMUN}"
        assert ag["goal"] == rec_repo.GOAL
        assert ag["role"] == f'Agente Recomendador de Retención — {variante}'

    # Todas las combinaciones de ruta/variante, con y sin conclusión del Analista
    rutas = [
        {'ruta': 2, 'variante': 'alto_valor', 'motivo': 'm1'},
        {'ruta': 2, 'variante': 'cartera_masiva', 'motivo': 'm2'},
        {'ruta': 3, 'variante': 'cartera_masiva', 'motivo': 'm3'},
        {'ruta': 3, 'variante': 'alto_valor', 'motivo': 'm4'},
    ]
    for caso in casos.values():
        for ruta_info in rutas + [enrutar_caso(caso, base.df_train)]:
            if ruta_info['ruta'] == 1:
                continue
            for conclusion in (None, "Conclusión de prueba."):
                args = (caso['cliente'], caso['datos'], caso['prediccion'], caso['senales'], ruta_info)
                t = ns["crear_tarea_recomendacion"](*args, agente="AG", conclusion_analista=conclusion)
                assert t["description"] == rec_repo.descripcion_tarea_recomendacion(
                    *args, conclusion_analista=conclusion)
                assert t["expected_output"] == rec_repo.EXPECTED_OUTPUT_RECOMENDACION
    from backend.agents.schemas import RecomendacionChurn
    assert ns["RecomendacionChurn"].model_json_schema() == RecomendacionChurn.model_json_schema()


def test_tools_crewai_se_construyen(modelo, base):
    """Las factorías crean tools de CrewAI con los nombres del notebook y registran llamadas."""
    from backend.agents.tool_registry import RegistroLlamadas
    from backend.agents.tools import crear_tool_predictor, crear_tool_simulacion, crear_tools_analista
    registro = RegistroLlamadas()
    tools = crear_tools_analista(base, registro)
    assert [t.name for t in tools] == ["Balance", "Productos", "Actividad", "Contexto"]
    tools[1].run(num_productos=2)
    assert registro.tools_llamadas() == [segments.NOMBRE_REGISTRO_PRODUCTOS]
    assert crear_tool_predictor(modelo).name == "Predecir Churn de Cliente"
    assert crear_tool_simulacion(modelo, base.prod_estable).name == "Simulación reanalisis"
