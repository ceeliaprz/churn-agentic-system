"""Agente Analista (notebook agéntico, celdas 32 y 34).

Caracteriza al cliente frente al histórico de df_train con 4 tools. Modelo:gpt-oss-20b
"""
from __future__ import annotations

from typing import Optional

from crewai import LLM, Agent, Task

from config import settings
from backend.agents.llm import crear_llm_analista
from backend.agents.schemas import AnalisisCliente

ROLE = 'Analista de Negocio Senior'
GOAL = ('Caracterizar el perfil de un cliente bancario frente al histórico, citando solo la '
        'evidencia que realmente cambia la lectura del caso.')


def backstory_analista(referencias_texto: str) -> str:
    return (
        'Eres un analista de negocio senior especializado en cartera de clientes bancarios.\n\n'
        'No calculas ni recuerdas cifras: cualquier tasa o percentil de tu análisis debe proceder '
        'de una herramienta. Si no la has consultado, no tienes el dato.\n\n'
        'Tu informe se juzga por la calidad de la evidencia que cita, no por su extensión. Antes '
        'de usar una herramienta, valora si su resultado puede añadir algo a tu conclusión sobre '
        'ESTE cliente; si solo va a confirmar algo ya evidente en sus datos, no la uses. Dos '
        'señales bien elegidas valen más que cinco de las que tres son irrelevantes, y dejar una '
        'herramienta sin usar es una decisión correcta.\n\n'
        'Referencias de la distribución de la base, para juzgar si un dato del cliente es atípico '
        f'(describen la base, no el riesgo de abandono):\n{referencias_texto}\n\n'
        'No propones acciones de retención ni predices si el cliente abandonará: solo caracterizas '
        'su perfil frente al histórico.'
    )


def crear_agente_analista(tools_analista: list, referencias_texto: str,
                          llm: Optional[LLM] = None) -> Agent:
    return Agent(
        role=ROLE,
        goal=GOAL,
        backstory=backstory_analista(referencias_texto),
        max_iter=settings.MAX_ITER_ANALISTA,
        tools=tools_analista,
        verbose=settings.VERBOSE_AGENTES,
        llm=llm or crear_llm_analista(),
    )


def texto_reanalisis(motivo: str) -> str:
    """Párrafo que se añade a la tarea cuando el Asistente (capa de orquestación) pide repetir
    el análisis. Es una indicación, no una orden: el Analista sigue decidiendo qué
    herramientas consulta. Solo existe en el repo (el notebook no tiene reanálisis)."""
    return (
        f"\n\nEste es un segundo análisis de este cliente, solicitado por el asistente que "
        f"coordina el sistema. Motivo: {motivo.strip()}\n"
        f"Tenlo en cuenta al decidir qué herramientas consultar. Puedes seguir usando cualquiera "
        f"de ellas, y todas las cifras deben salir de las herramientas."
    )


def descripcion_tarea_analisis(nombre_cliente: str, cliente: dict,
                               motivo_reanalisis: Optional[str] = None) -> str:
    """Sin motivo, el texto es idéntico al del notebook. Con motivo, se le añade
    texto_reanalisis()."""
    datos_formateados = "\n".join(f"- {k}: {v}" for k, v in cliente.items())
    descripcion = (
        f"Analiza el perfil del cliente '{nombre_cliente}' frente al histórico de la base.\n"
        f"Datos del cliente:\n{datos_formateados}\n\n"
        f"Empieza por las variables de comportamiento del cliente con la entidad (productos "
        f"contratados y estado de actividad), que son las que más separan la base histórica. "
        f"Añade después las variables de perfil (saldo y contexto demográfico) solo si su "
        f"resultado puede cambiar la lectura del caso. Responde en español."
        f"Al elegir la orientación, pondera la intensidad de cada señal, no su número: "
        f"una señal muy alejada de la media (por ejemplo x4) pesa más que varias señales "
        f"ligeramente por debajo. Una sola señal extrema puede determinar la orientación."
    )
    if motivo_reanalisis and motivo_reanalisis.strip():
        descripcion += texto_reanalisis(motivo_reanalisis)
    return descripcion


EXPECTED_OUTPUT_ANALISIS = (
    "Nombre del cliente, lista de señales consultadas con sus cifras exactas, "
    "orientación global del caso y conclusión de 1-2 frases."
)


def analizar_cliente(nombre_cliente: str, cliente: dict, agente: Agent,
                     motivo_reanalisis: Optional[str] = None) -> Task:
    """Tarea del Analista (mismo nombre que en el notebook). `motivo_reanalisis` solo lo usa
    el Asistente del chat al repetir un análisis."""
    return Task(
        description=descripcion_tarea_analisis(nombre_cliente, cliente, motivo_reanalisis),
        expected_output=EXPECTED_OUTPUT_ANALISIS,
        output_pydantic=AnalisisCliente,
        agent=agente,
    )
