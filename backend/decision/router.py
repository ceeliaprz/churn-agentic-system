"""Router de tres rutas (notebook agéntico, celdas 50 y 52). Decidido en Python, nunca por un LLM.

Combina la probabilidad del Predictor (riesgo oficial) con la agregación determinista del
Analista (`balance_de_senales`) y el percentil de saldo (variante del Recomendador).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from config.settings import UMBRAL_PROB_ALTA, UMBRAL_PROB_BAJA, UMBRAL_VALOR_ALTO


def _df_train_por_defecto() -> pd.DataFrame:
    from backend.analytics.reference import get_base_referencia
    return get_base_referencia().df_train


def percentil_balance_numerico(balance: float, df_train: Optional[pd.DataFrame] = None) -> float:
    """Mismo cálculo que usa la tool 'Balance' del Analista, pero como número (no como
    texto), para decidir la variante de agente sin tener que parsear el resultado de la
    tool. Un saldo de 0 € se trata como percentil 0: es el extremo bajo de valor, no un
    caso especial de churn (eso ya lo cubre la propia tool del Analista)."""
    if balance <= 0:
        return 0.0
    if df_train is None:
        df_train = _df_train_por_defecto()
    return float((df_train['Balance'] < balance).mean() * 100)


def enrutar_cliente(churn_probability_pct: float, orientacion_calculada: str, balance: float,
                    df_train: Optional[pd.DataFrame] = None) -> dict:
    """Decide la ruta de un cliente. Por diseño, la probabilidad del Predictor manda: si es
    alta, va a Ruta 2 aunque el Analista no esté de acuerdo, porque el Predictor es el
    riesgo oficial y el Analista solo aporta evidencia. Ruta 3 recoge todo lo que no es un cierre
    claro (Ruta 1) ni un riesgo claramente alto (Ruta 2): tanto el desacuerdo franco entre
    fuentes como la zona intermedia del Predictor caen ahí, porque en ambos casos falta
    certeza y conviene forzar el análisis adicional (simulación reanálisis) antes de
    recomendar nada."""
    riesgo_alto = churn_probability_pct >= UMBRAL_PROB_ALTA
    riesgo_bajo = churn_probability_pct < UMBRAL_PROB_BAJA
    analista_bajo = orientacion_calculada == "riesgo_por_debajo_de_la_media"

    percentil_valor = percentil_balance_numerico(balance, df_train)
    variante = "alto_valor" if percentil_valor >= UMBRAL_VALOR_ALTO else "cartera_masiva"

    if riesgo_bajo and analista_bajo:
        return {
            'ruta': 1, 'nombre_ruta': 'cierre_sin_recomendador', 'variante': None,
            'percentil_valor': percentil_valor,
            'motivo': (f"Predictor bajo ({churn_probability_pct:.1f}%) y Analista bajo "
                       f"({orientacion_calculada}): ambas fuentes coinciden en que no hay caso."),
        }

    if riesgo_alto:
        return {
            'ruta': 2, 'nombre_ruta': 'recomendador_riesgo_alto', 'variante': variante,
            'percentil_valor': percentil_valor,
            'motivo': (f"Predictor alto ({churn_probability_pct:.1f}%) → Recomendador, "
                       f"variante '{variante}' (percentil de saldo {percentil_valor:.1f})."),
        }

    # si riesgo bajo y analista alto, o zona intermedia del Predictor
    return {
        'ruta': 3, 'nombre_ruta': 'recomendador_revision', 'variante': variante,
        'percentil_valor': percentil_valor,
        'motivo': (f"Predictor ({churn_probability_pct:.1f}%) no llega a riesgo alto y no hay "
                   f"acuerdo claro de riesgo bajo con el Analista ({orientacion_calculada}): "
                   f"Recomendador en modo revisión, con simulación reanálisis obligatoria."),
    }


def enrutar_caso(caso: dict, df_train: Optional[pd.DataFrame] = None) -> dict:
    """Atajo: aplica el router a un caso con el formato de `construir_caso`."""
    return enrutar_cliente(
        churn_probability_pct=caso['prediccion']['churn_probability_pct'],
        orientacion_calculada=caso['orientacion_calculada'],
        balance=caso['datos']['Balance'],
        df_train=df_train,
    )


def construir_reparto(casos: dict, df_train: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Aplica el router a los casos ya calculados y devuelve la tabla de reparto: qué ruta y
    qué variante le tocó a cada cliente, y por qué."""
    filas = []
    for perfil, caso in casos.items():
        info = enrutar_caso(caso, df_train)
        filas.append({
            'perfil': perfil,
            'cliente': caso['cliente'],
            'probabilidad_pct': caso['prediccion']['churn_probability_pct'],
            'orientacion_analista': caso['orientacion_calculada'],
            'percentil_saldo': round(info['percentil_valor'], 1),
            'ruta': info['nombre_ruta'],
            'variante': info['variante'],
            'motivo': info['motivo'],
        })
    return pd.DataFrame(filas)
