"""Cobertura de riesgo: modelo tradicional de umbral único vs. sistema completo
(notebook agéntico, celda 69). 100% determinista y sin coste de Groq.

Resultado de referencia (2026-09-21): 2.000 clientes; el umbral único cerraría sin acción
a 1.053; de ellos el sistema mantiene en revisión a 81 (4,05% del total).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from config.settings import UMBRAL_PROB_BAJA
from backend.analytics.reference import BaseReferencia, get_base_referencia
from backend.decision.router import enrutar_cliente
from backend.ml.predictor import ModeloChurn, get_modelo
from backend.pipeline.cases import construir_caso


def comparar_cobertura_riesgo(X_test_raw: pd.DataFrame, modelo: Optional[ModeloChurn] = None,
                              base: Optional[BaseReferencia] = None) -> pd.DataFrame:
    """Compara, sobre todo el test set, la decisión de un modelo tradicional de umbral único
    (actuar solo si churn_probability_pct >= UMBRAL_PROB_BAJA) frente a la decisión real del
    sistema completo (el router de 3 rutas, que exige acuerdo del Analista para cerrar sin más).
    Usa predict_churn y las 4 tools del Analista en Python directo, igual que construir_caso."""
    modelo = modelo or get_modelo()
    base = base or get_base_referencia()

    filas = []
    for _, fila in X_test_raw.iterrows():
        caso = construir_caso(fila, modelo, base)
        prob = caso['prediccion']['churn_probability_pct']
        orientacion = caso['orientacion_calculada']

        decision_tradicional = 'sin_accion' if prob < UMBRAL_PROB_BAJA else 'accion'

        ruta = enrutar_cliente(
            churn_probability_pct=prob,
            orientacion_calculada=orientacion,
            balance=float(fila['Balance']),
            df_train=base.df_train,
        )
        decision_sistema = 'sin_accion' if ruta['ruta'] == 1 else 'accion'

        filas.append({
            'cliente': caso['cliente'],
            'probabilidad_pct': prob,
            'orientacion_calculada': orientacion,
            'decision_tradicional': decision_tradicional,
            'decision_sistema': decision_sistema,
            'ruta_sistema': ruta['nombre_ruta'],
        })

    return pd.DataFrame(filas)


def validar_rescatados(resultado: pd.DataFrame, y_test: pd.Series) -> dict:
    """Validación con la etiqueta real (notebook agéntico, sección "2.2. Validación de los
    clientes rescatados con la etiqueta real"). Los clientes que el sistema mantiene en
    revisión y el modelo tradicional cierra, ¿abandonan realmente más que los que ambos
    cierran? Devuelve la tabla de tasas por grupo y el p-valor del test exacto de Fisher
    unilateral.

    `resultado` es la salida de comparar_cobertura_riesgo (una fila por cliente, en el orden
    de X_test_raw) e `y_test` la columna Exited de y_test_raw, alineada por posición.

    Resultado de referencia (2026-09-30): rescatados 81 / 10 / 12,3 %; ambos cierran
    972 / 42 / 4,3 %; global 2.000 / 407 / 20,4 %; p = 0,0045.
    """
    from scipy.stats import fisher_exact

    if len(y_test) != len(resultado):
        raise ValueError("y_test y el resultado de cobertura no tienen el mismo número de filas")

    val = resultado.copy()
    val['Exited'] = pd.Series(y_test).astype(int).values

    trad_cierra = val['decision_tradicional'] == 'sin_accion'
    sist_cierra = val['decision_sistema'] == 'sin_accion'
    rescatados = val[trad_cierra & ~sist_cierra]    # el sistema los mantiene en revisión
    ambos_cierran = val[trad_cierra & sist_cierra]  # el sistema también los cierra

    tabla = pd.DataFrame([
        {'grupo': nombre, 'clientes': len(g), 'abandonan': int(g['Exited'].sum()),
         'tasa_abandono_%': round(100 * g['Exited'].sum() / len(g), 1)}
        for nombre, g in [('Rescatados por el sistema', rescatados),
                          ('Ambos cierran (Ruta 1)', ambos_cierran),
                          ('Global test set', val)]
    ])

    _, p_valor = fisher_exact(
        [[int(rescatados['Exited'].sum()), int((rescatados['Exited'] == 0).sum())],
         [int(ambos_cierran['Exited'].sum()), int((ambos_cierran['Exited'] == 0).sum())]],
        alternative='greater')

    return {'tabla': tabla, 'p_valor': float(p_valor)}


def resumen_cobertura(resultado: pd.DataFrame) -> dict:
    total = len(resultado)
    cerrados_trad = int((resultado['decision_tradicional'] == 'sin_accion').sum())
    rescatados = resultado[(resultado['decision_tradicional'] == 'sin_accion')
                           & (resultado['decision_sistema'] == 'accion')]
    return {
        'total': total,
        'cerrados_tradicional': cerrados_trad,
        'pct_cerrados_tradicional': 100 * cerrados_trad / total if total else 0.0,
        'rescatados': int(len(rescatados)),
        'pct_rescatados_total': 100 * len(rescatados) / total if total else 0.0,
        'pct_rescatados_de_cerrados': 100 * len(rescatados) / cerrados_trad if cerrados_trad else 0.0,
        'reparto_rutas': resultado['ruta_sistema'].value_counts().to_dict(),
    }
