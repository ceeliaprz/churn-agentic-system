"""Agente Predictor (notebook agéntico, celdas 19 y 21).

Única responsabilidad: llamar al modelo de ML a través de su tool y devolver el riesgo de
churn, sin interpretar los datos del cliente. Textos literales del notebook.
"""
from __future__ import annotations

from typing import Optional

from crewai import LLM, Agent, Task

from config import settings
from backend.agents.llm import crear_llm_predictor
from backend.agents.schemas import PrediccionChurn

ROLE = 'Agente Predictor de Churn'
GOAL = ('Calcular el riesgo de abandono de un cliente usando exclusivamente el modelo de ML, '
        'sin interpretar ni analizar sus datos.')
BACKSTORY = (
    'Eres un componente especializado y estrictamente limitado a una única función: '
    'llamar al modelo predictivo de churn para obtener un resultado numérico. '
    'SIEMPRE debes usar la tool disponible para obtener la predicción, nunca debes '
    'estimar o inventar un resultado por tu cuenta. '
    'NUNCA debes explicar por qué un cliente tiene ese riesgo, ni mencionar ninguna '
    'variable o dato del cliente (edad, país, productos, salario, etc.) en tu respuesta. '
    'Ese análisis lo realiza otro agente especializado, no tú. '
    'Tu única salida debe ser: nombre del cliente, predicción de churn y probabilidad exacta.'
)


def crear_agente_predictor(tool_predictor, llm: Optional[LLM] = None) -> Agent:
    return Agent(
        role=ROLE,
        goal=GOAL,
        backstory=BACKSTORY,
        tools=[tool_predictor],
        verbose=settings.VERBOSE_AGENTES,
        llm=llm or crear_llm_predictor(),
    )


def descripcion_tarea_prediccion(nombre_cliente: str, customer_data: dict) -> str:
    return (
        f'Cliente: {nombre_cliente}.\n'
        f'Datos del cliente: {customer_data}\n\n'
        'Llama a la tool de predicción de churn con estos datos.'
        'No añadas ninguna explicación, justificación ni mención a las variables del cliente.'
    )


EXPECTED_OUTPUT_PREDICCION = (
    'Un objeto con: cliente (el nombre), churn_prediction (0 o 1, tal cual lo '
    'devuelve la tool) y churn_probability_pct % (el porcentaje de riesgo de churn, '
    'tal cual lo devuelve la tool, sin recalcularlo ni redondearlo tú).'
)


def crear_tarea_prediccion(nombre_cliente: str, customer_data: dict, agente: Agent) -> Task:
    return Task(
        description=descripcion_tarea_prediccion(nombre_cliente, customer_data),
        expected_output=EXPECTED_OUTPUT_PREDICCION,
        agent=agente,
        output_pydantic=PrediccionChurn,
    )
