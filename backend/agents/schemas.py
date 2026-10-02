"""Salidas estructuradas de los tres agentes (`output_pydantic`), notebook celdas 21, 34 y 58.

Las descripciones de los campos forman parte del prompt que ve el LLM: se copian literalmente.
"""
from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field


class PrediccionChurn(BaseModel):
    cliente: str
    churn_prediction: int
    churn_probability_pct: float  # 0-100%, tal cual lo devuelve la tool


class AnalisisCliente(BaseModel):
    cliente: str
    senales: List[str] = Field(..., description="Una entrada por herramienta usada, con el nombre "
                                               "de la herramienta y el resultado exacto que devolvió")
    orientacion: Literal[
        "riesgo_por_encima_de_la_media",
        "riesgo_en_linea_con_la_media",
        "riesgo_por_debajo_de_la_media",
    ] = Field(..., description="Hacia dónde inclina el conjunto de señales. Debes elegir una: "
                              "no existe la opción 'mixto'")
    conclusion: str = Field(..., description="1-2 frases justificando la orientación, sin proponer "
                                            "acciones de retención ni mencionar predicciones")


class RecomendacionChurn(BaseModel):
    cliente: str
    ruta: Literal["riesgo_alto", "revision"]
    variante_agente: Literal["alto_valor", "cartera_masiva"]
    resumen_cliente: str = Field(..., description="1-2 frases en lenguaje natural describiendo al cliente a partir "
                                  "de sus datos (edad, antigüedad, saldo, productos...), sin mencionar el riesgo.")
    resumen_riesgo: str = Field(..., description="1-2 frases: probabilidad del Predictor y las señales del Analista "
                                 "cuyo multiplicador (x) sea mayor que 1, que reespaldan el churn. No reevalúa el riesgo, solo lo resume.")
    accion_principal: str = Field(..., description="Acción de retención concreta y viable. Debe actuar sobre la palanca objetivo, "
                                    "pero no debe asumir que una modificación de la variable causa una reducción del churn "
                                    "ni ordenar directamente al cliente modificar sus productos, salvo que exista una "
                                    "justificación explícita independiente de la simulación.")
    palanca_objetivo: str = Field(..., description="Variable sobre la que actúa: productos, actividad, tarjeta u otra")
    canal: str = Field(..., description="Cómo se ejecuta: gestor personal, campaña automatizada, contact center, etc.")
    justificacion: str = Field(..., description="1-3 frases que expliquen por qué se propone la acción, apoyándose en las señales "
                                    "del Analista y, si se utilizó, en la simulación. El resultado de la simulación debe "
                                    "interpretarse como sensibilidad del modelo, no como evidencia de causalidad.")
    limitaciones: str = Field(..., description="Advertencia obligatoria (nunca vacío): sensibilidad no causal si se "
                               "usó la simulación reanálisis, o que la recomendación se basa en tasas históricas "
                               "de segmentos similares (no en una estimación para este cliente) si no se usó.")
