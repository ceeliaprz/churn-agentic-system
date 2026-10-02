"""Equivalencia directa con el código del notebook.

Ejecuta (exec) las celdas originales de notebooks/notebook_pipeline_agentico.ipynb con CrewAI
sustituido por stubs y compara su salida con la del repo: textos de las tools, docstrings,
prompts de las tareas, backstories de los agentes y esquemas pydantic.
"""
import inspect

import pytest

from backend.analytics import segments
from tests.conftest import celda_notebook

pytestmark = pytest.mark.datos


def _limpio(doc: str) -> str:
    return inspect.cleandoc(doc or "")


@pytest.fixture
def ns_analista(ns_stub, base):
    """Ejecuta las celdas 27, 29 y 30 (referencias, registro y tools del Analista)."""
    import numpy as np
    import pandas as pd
    ns = {**ns_stub, "pd": pd, "np": np, "df_train": base.df_train, "print": lambda *a, **k: None}
    exec(celda_notebook("def calcular_referencias"), ns)
    exec(celda_notebook("registro_llamadas = []"), ns)
    exec(celda_notebook("def _formato_segmento"), ns)
    return ns


def test_referencias_y_docstrings(ns_analista, base):
    assert ns_analista["BASE_CHURN"] == base.base_churn
    assert list(ns_analista["Q_BALANCE_POS"]) == list(base.q_balance_pos)
    assert ns_analista["referencias_texto"] == base.referencias_texto
    assert ns_analista["_prod_estable"] == base.prod_estable
    assert ns_analista["DOC_CONTEXTO"] == base.doc_contexto
    assert ns_analista["DOC_PRODUCTOS"] == base.doc_productos
    assert _limpio(ns_analista["consultar_actividad_vinculacion"].__doc__) == _limpio(segments.DOC_ACTIVIDAD)
    assert _limpio(ns_analista["calcular_percentil_balance"].__doc__) == _limpio(segments.DOC_BALANCE)


def test_salidas_de_las_tools_300_clientes(ns_analista, base, x_test):
    ns = ns_analista
    for _, fila in x_test.head(300).iterrows():
        assert (ns["_consultar_riesgo_productos"](int(fila['NumOfProducts']))
                == segments.consultar_riesgo_productos(base, int(fila['NumOfProducts'])))
        assert (ns["consultar_actividad_vinculacion"](int(fila['IsActiveMember']), int(fila['HasCrCard']))
                == segments.consultar_actividad_vinculacion(base, int(fila['IsActiveMember']),
                                                            int(fila['HasCrCard'])))
        assert (ns["calcular_percentil_balance"](float(fila['Balance']))
                == segments.calcular_percentil_balance(base, float(fila['Balance'])))
        assert (ns["_consultar_contexto_demografico"](fila['Geography'], int(fila['Age']))
                == segments.consultar_contexto_demografico(base, fila['Geography'], int(fila['Age'])))


def test_router_notebook_igual(ns_stub, base, casos):
    ns = {**ns_stub, "df_train": base.df_train}
    exec(celda_notebook("def enrutar_cliente"), ns)
    from backend.decision.router import enrutar_caso
    for caso in casos.values():
        assert ns["enrutar_cliente"](caso['prediccion']['churn_probability_pct'],
                                     caso['orientacion_calculada'],
                                     caso['datos']['Balance']) == enrutar_caso(caso, base.df_train)


def test_simulacion_notebook_igual(ns_stub, base, modelo, casos):
    from backend.analytics.counterfactual import simular_palancas, texto_simulacion
    from backend.ml.predictor import predict_churn
    ns = {**ns_stub, "predict_churn": predict_churn, "_prod_estable": base.prod_estable,
          "print": lambda *a, **k: None}
    exec(celda_notebook('@tool("Simulación reanalisis")'), ns)
    for caso in casos.values():
        assert (ns["simular_reanalisis_tool"](**caso['datos'])
                == texto_simulacion(simular_palancas(modelo, base.prod_estable, caso['datos'])))
