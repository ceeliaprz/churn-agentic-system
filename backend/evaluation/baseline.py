"""Comparador de reglas fijas: el "modelo tradicional"

Una acción genérica por banda de riesgo, sin usar ninguna señal del cliente ni la simulación.
Es la línea base con la que se aplicó la rúbrica: 8/32 frente a 28/32 del pipeline completo.

Corrección 2026-09-30: antes se usaba el router del sistema (`enrutar_caso`), que incluye la
orientación calculada del Analista. El modelo tradicional no tiene Analista, así que decide la
banda solo con la probabilidad del modelo (`enrutar_tradicional`). Solo cambia Gray (17,37 %),
que pasa de revisión a cierre sin acción.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from config.settings import UMBRAL_PROB_ALTA, UMBRAL_PROB_BAJA, UMBRAL_VALOR_ALTO
from backend.decision.router import percentil_balance_numerico


def enrutar_tradicional(churn_probability_pct: float, balance: float,
                        df_train: Optional[pd.DataFrame] = None) -> dict:
    """Modelo tradicional: decide la banda de riesgo usando SOLO la probabilidad del
    modelo, con los mismos umbrales que el router. No hay Analista, así que por debajo
    del 30 % el caso siempre se cierra. El percentil de saldo solo elige el canal."""
    percentil_valor = percentil_balance_numerico(balance, df_train)
    variante = "alto_valor" if percentil_valor >= UMBRAL_VALOR_ALTO else "cartera_masiva"

    if churn_probability_pct < UMBRAL_PROB_BAJA:
        return {'ruta': 1, 'nombre_ruta': 'cierre_sin_accion', 'variante': None}
    if churn_probability_pct >= UMBRAL_PROB_ALTA:
        return {'ruta': 2, 'nombre_ruta': 'riesgo_alto', 'variante': variante}
    return {'ruta': 3, 'nombre_ruta': 'riesgo_medio', 'variante': variante}


def recomendacion_tradicional(nombre_cliente: str, ruta_info: dict) -> dict:
    """Modelo tradicional de reglas fijas: una acción genérica por banda de riesgo, sin usar
    ninguna señal específica del cliente ni la simulación contrafactual. Línea base para medir
    cuánta utilidad añade el Recomendador agéntico."""
    ruta = ruta_info['ruta']

    if ruta == 1:
        accion = "Sin acción. El cliente no supera el umbral de riesgo del modelo."
        canal = None
    elif ruta == 2:
        accion = ("Contacto urgente de retención: ofrecer condiciones preferentes estándar "
                  "(mismo guion para todos los clientes de esta banda de riesgo).")
        canal = "gestor personal" if ruta_info['variante'] == 'alto_valor' else "contact center"
    else:  # ruta 3
        accion = ("Incluir en seguimiento de riesgo medio: contacto de revisión general, "
                  "sin oferta específica.")
        canal = "contact center"

    return {
        'cliente': nombre_cliente,
        'ruta': ruta_info['nombre_ruta'],
        'canal': canal,
        'accion_principal': accion,
        'palanca_objetivo': None,
        'justificacion': (f"Probabilidad de churn del modelo por encima o debajo de los "
                          f"umbrales fijos ({UMBRAL_PROB_BAJA}% / {UMBRAL_PROB_ALTA}%). "
                          f"No se han considerado señales adicionales del cliente."),
    }


def generar_baseline(casos: dict, df_train: Optional[pd.DataFrame] = None) -> dict:
    """Aplica el comparador a todos los casos (formato de construir_caso). Solo usa la
    probabilidad del Predictor y el saldo, nunca la orientación del Analista."""
    baseline = {}
    for perfil, caso in casos.items():
        ruta_info = enrutar_tradicional(caso['prediccion']['churn_probability_pct'],
                                        caso['datos']['Balance'], df_train)
        baseline[perfil] = recomendacion_tradicional(caso['cliente'], ruta_info)
    return baseline
