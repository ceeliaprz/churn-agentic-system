"""Agregación determinista de las señales del Analista (notebook agéntico, celda 35).

El router usa SIEMPRE esta orientación calculada, nunca la `orientacion` que redacta el LLM.
"""
from __future__ import annotations

import re
from typing import Iterable, Union

import numpy as np

from config import settings

_PATRON_LIFT = re.compile(r"x(\d+\.\d+)")


def _textos(registro: Iterable[Union[dict, str]]) -> list[str]:
    """Acepta el registro de llamadas del notebook (lista de dicts con 'resultado') o una
    lista de textos de señales."""
    return [llamada['resultado'] if isinstance(llamada, dict) else str(llamada)
            for llamada in registro]


def extraer_lifts(registro: Iterable[Union[dict, str]]) -> list[float]:
    return [float(x) for texto in _textos(registro) for x in _PATRON_LIFT.findall(texto)]


def lift_combinado(registro: Iterable[Union[dict, str]]) -> float | None:
    lifts = extraer_lifts(registro)
    return float(np.prod(lifts)) if lifts else None


def balance_de_senales(registro: Iterable[Union[dict, str]]) -> str:
    """Combina los lifts de todas las señales por producto (equivalente a acumular razones de
    verosimilitud asumiendo independencia). A diferencia de contar señales por dirección, esto
    respeta la magnitud: una señal de x4 no queda neutralizada por varias de x0.8."""
    lifts = extraer_lifts(registro)
    if not lifts:
        return "riesgo_en_linea_con_la_media"
    combinado = float(np.prod(lifts))
    if combinado >= settings.BALANCE_UMBRAL_ENCIMA:
        return "riesgo_por_encima_de_la_media"
    if combinado <= settings.BALANCE_UMBRAL_DEBAJO:
        return "riesgo_por_debajo_de_la_media"
    return "riesgo_en_linea_con_la_media"
