"""Simulación contrafactual / reanálisis del Recomendador (notebook agéntico, celda 54).

Re-invoca el modelo cambiando, una a una, las variables sobre las que el banco puede
intervenir (activación, tarjeta, número de productos hacia `prod_estable`) y devuelve solo
el delta en puntos porcentuales. Sensibilidad del modelo, no causalidad.

`simular_palancas` devuelve los datos estructurados (para el pipeline) y
`texto_simulacion` el texto literal que recibía el LLM en el notebook. El `print` del
notebook se sustituye por `logging`.
"""
from __future__ import annotations

import logging

from backend.ml.predictor import ModeloChurn

logger = logging.getLogger(__name__)

TEXTO_SIN_PALANCAS = (
    "Este cliente ya está en la posición más favorable en las tres variables "
    "sobre las que el banco puede intervenir (activo, con tarjeta, en el número "
    "de productos más estable). No hay palanca que simular; la recomendación "
    "tendrá que apoyarse solo en las señales del Analista."
)


def simular_palancas(modelo: ModeloChurn, prod_estable: int, base: dict) -> dict:
    """Devuelve {'prob_base': float|None, 'error': str|None,
                 'resultados': [(nombre_palanca, delta_pp|None), ...] ordenados}."""
    try:
        prob_base = modelo.predict_churn(base)['churn_probability']
    except Exception as e:
        return {'prob_base': None, 'error': str(e), 'resultados': []}

    candidatas = {}
    if base['IsActiveMember'] == 0:
        candidatas['Activar al cliente'] = {**base, 'IsActiveMember': 1}
    if base['HasCrCard'] == 0:
        candidatas['Ofrecer tarjeta de crédito'] = {**base, 'HasCrCard': 1}
    if base['NumOfProducts'] != prod_estable:
        candidatas[f'Llevar a {prod_estable} producto(s)'] = {**base, 'NumOfProducts': prod_estable}

    resultados = []
    for nombre, modificado in candidatas.items():
        try:
            prob_mod = modelo.predict_churn(modificado)['churn_probability']
            resultados.append((nombre, (prob_mod - prob_base) * 100))
        except Exception:
            resultados.append((nombre, None))

    resultados.sort(key=lambda par: (par[1] is None, par[1]))
    return {'prob_base': prob_base, 'error': None, 'resultados': resultados}


def texto_simulacion(simulacion: dict) -> str:
    """Formatea la simulación exactamente como la devolvía `simular_reanalisis_tool`."""
    if simulacion['error'] is not None:
        return f"Error al calcular la situación actual del cliente: {simulacion['error']}"
    if not simulacion['resultados']:
        return TEXTO_SIN_PALANCAS

    lineas = [
        f"- {nombre}: {delta:+.1f} puntos porcentuales" if delta is not None
        else f"- {nombre}: error al simular"
        for nombre, delta in simulacion['resultados']
    ]
    resultado_texto = (
        "Impacto simulado de cada palanca disponible sobre la probabilidad de churn de "
        "este cliente, ordenado de mayor a menor reducción de riesgo (sensibilidad del "
        "modelo, no causalidad):\n" + "\n".join(lineas)
    )
    logger.info("[Simulación y Reanálisis — resultado completo]\n%s", resultado_texto)
    return resultado_texto
