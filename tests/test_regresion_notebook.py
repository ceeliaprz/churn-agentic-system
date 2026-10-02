"""Tests de regresión: el repo reproduce las cifras del notebook (Fases 2, 3, 4 y 7 del plan)."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from config import settings
from tests.conftest import leer_json_notebook

pytestmark = pytest.mark.datos


# --- Fase 2: capa ML --------------------------------------------------------
def test_predict_churn_30_de_30(modelo, x_test):
    """Celda 15 del notebook: 30/30 predicciones idénticas a las del entrenamiento."""
    originales = pd.read_csv(settings.RUTA_PREDICCIONES_ORIGINALES)
    from backend.pipeline.cases import datos_cliente
    for i in range(30):
        r = modelo.predict_churn(datos_cliente(x_test.iloc[i]))
        assert r['churn_prediction'] == originales.iloc[i]['pred_clase']
        assert abs(r['churn_probability'] - originales.iloc[i]['pred_proba']) < 1e-6


def test_predicciones_2000_de_2000(modelo, x_test):
    originales = pd.read_csv(settings.RUTA_PREDICCIONES_ORIGINALES)
    lote = modelo.predict_lote(x_test)
    assert (lote['churn_prediction'].values == originales['pred_clase'].values).all()
    assert np.abs(lote['churn_probability'].values - originales['pred_proba'].values).max() < 1e-6


def test_feature_order_oficial():
    from backend.ml.features import FEATURE_ORDER_ESPERADO
    assert list(joblib.load(settings.RUTA_FEATURE_ORDER)) == FEATURE_ORDER_ESPERADO


@pytest.mark.lento
def test_reentrenar_reproduce_artefactos():
    """D1: train.py regenera exactamente el split, el scaler y las predicciones oficiales."""
    from backend.ml.train import entrenar, verificar_contra_oficiales
    v = verificar_contra_oficiales(entrenar())
    assert v['mismo_split'] and v['mismo_feature_order'] and v['mismo_scaler']
    assert v['misma_clase'] and v['misma_proba']
    assert v['roc_auc'] == pytest.approx(0.8654771875110858, abs=1e-9)
    assert v['pr_auc'] == pytest.approx(0.7163136941506724, abs=1e-9)


# --- Fase 3: analítica ------------------------------------------------------
def test_base_churn(base):
    assert base.base_churn == pytest.approx(20.375, abs=0.01)
    assert base.prod_estable == 2


def test_casos_congelados_identicos(casos):
    """construir_caso reproduce casos_recomendador.json (señales, lifts, orientación)."""
    assert casos == leer_json_notebook("casos_recomendador.json")


def test_clientes_de_los_perfiles(casos):
    from backend.pipeline.cases import CLIENTE_ESPERADO
    assert {p: c['cliente'] for p, c in casos.items()} == CLIENTE_ESPERADO


# --- Fase 4: router ---------------------------------------------------------
def test_reparto_6_casos(casos, base):
    from backend.decision.router import construir_reparto
    reparto = construir_reparto(casos, base.df_train).set_index('cliente')
    esperado = {
        'Hsiung': ('cierre_sin_recomendador', None, 0.0),
        'Lo': ('recomendador_riesgo_alto', 'alto_valor', 100.0),
        'Gardiner': ('recomendador_riesgo_alto', 'cartera_masiva', 66.9),
        "O'Donnell": ('cierre_sin_recomendador', None, 78.0),
        'Sun': ('recomendador_revision', 'cartera_masiva', 65.1),
        'Gray': ('recomendador_revision', 'cartera_masiva', None),
    }
    for cliente, (ruta, variante, percentil) in esperado.items():
        assert reparto.loc[cliente, 'ruta'] == ruta
        assert reparto.loc[cliente, 'variante'] == variante
        if percentil is not None:
            assert reparto.loc[cliente, 'percentil_saldo'] == pytest.approx(percentil)


def test_rutas_del_pipeline_real_coinciden(base):
    """Los 6 registros del pipeline real (8.1) tienen la ruta que da el router con sus entradas."""
    from backend.decision.router import enrutar_cliente
    for perfil, reg in leer_json_notebook("resultados_pipeline_completo.json").items():
        info = enrutar_cliente(reg['prediccion']['churn_probability_pct'],
                               reg['orientacion_calculada'], reg['datos']['Balance'],
                               df_train=base.df_train)
        assert info['ruta'] == reg['ruta_info']['ruta'], perfil
        assert info['variante'] == reg['ruta_info']['variante'], perfil


# --- Fase 7: evaluación -----------------------------------------------------
def test_baseline_tradicional_identico(casos, base):
    from backend.evaluation.baseline import generar_baseline
    assert generar_baseline(casos, base.df_train) == leer_json_notebook("baseline_tradicional.json")


def test_baseline_tradicional_no_usa_analista(base):
    """El modelo tradicional decide solo con la probabilidad: por debajo del 30 % cierra
    siempre, diga lo que diga el Analista (caso Gray, 17,37 %)."""
    from backend.evaluation.baseline import enrutar_tradicional
    assert enrutar_tradicional(17.37, 78778.0, base.df_train)['ruta'] == 1
    assert enrutar_tradicional(32.83, 116122.0, base.df_train)['ruta'] == 3
    assert enrutar_tradicional(95.13, 250898.0, base.df_train)['ruta'] == 2
    baseline = leer_json_notebook("baseline_tradicional.json")
    assert baseline['desacuerdo_franco']['ruta'] == 'cierre_sin_accion'


def test_validacion_rescatados_con_y_test():
    """Sección 2.2 de evaluaciones: los 81 rescatados abandonan un 12,3 % frente al 4,3 % de
    los que ambos cierran (Fisher unilateral, p = 0,0045). Usa la cobertura congelada del
    notebook, así que no recorre los 2.000 clientes."""
    from backend.evaluation.coverage import validar_rescatados
    cobertura = pd.read_csv(settings.RESULTS_NOTEBOOK_DIR / "comparacion_cobertura_riesgo.csv")
    y_test = pd.read_csv(settings.RUTA_Y_TEST_RAW)['Exited']
    x_test = pd.read_csv(settings.RUTA_X_TEST_RAW)
    assert (cobertura['cliente'].values == x_test['Surname'].values).all()

    v = validar_rescatados(cobertura, y_test)
    referencia = pd.read_csv(settings.RESULTS_NOTEBOOK_DIR / "tabla_validacion_rescatados.csv")
    pd.testing.assert_frame_equal(v['tabla'], referencia, check_dtype=False)
    assert v['p_valor'] == pytest.approx(0.0045, abs=5e-5)


@pytest.mark.lento
def test_cobertura_riesgo(x_test, modelo, base):
    """Sección 9: 2.000 clientes, 1.053 cerrados por umbral único, 81 rescatados (4,05%)."""
    from backend.evaluation.coverage import comparar_cobertura_riesgo, resumen_cobertura
    resultado = comparar_cobertura_riesgo(x_test, modelo, base)
    r = resumen_cobertura(resultado)
    assert (r['total'], r['cerrados_tradicional'], r['rescatados']) == (2000, 1053, 81)
    assert r['pct_rescatados_total'] == pytest.approx(4.05)
    referencia = pd.read_csv(settings.RESULTS_NOTEBOOK_DIR / "comparacion_cobertura_riesgo.csv")
    pd.testing.assert_frame_equal(resultado, referencia)


# --- knowledge_base ---------------------------------------------------------
def test_reference_stats_actualizado(base):
    """Si existe knowledge_base/reference_stats.json, coincide con df_train
    (regenerar con scripts/build_knowledge_base.py)."""
    if not settings.RUTA_REFERENCE_STATS.exists():
        pytest.skip("reference_stats.json aún no generado")
    with open(settings.RUTA_REFERENCE_STATS, encoding="utf-8") as f:
        guardado = json.load(f)
    assert guardado == json.loads(json.dumps(base.resumen(), ensure_ascii=False))
