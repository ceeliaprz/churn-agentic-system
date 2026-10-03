"""Agente Recomendador

Dos variantes de agente (alto_valor / cartera_masiva), elegidas por el router según el
percentil de saldo. Los campos `ruta` y `variante_agente` de la salida no los decide el LLM:
vienen fijados por el router en Python.

"""
from __future__ import annotations

from typing import Optional

from crewai import LLM, Agent, Task

from config import settings
from backend.agents.llm import crear_llm_recomendador
from backend.agents.schemas import RecomendacionChurn

_BACKSTORY_COMUN = (
    'El nivel de riesgo oficial es la probabilidad que calculó el Predictor: no la '
    'cuestionas ni la vuelves a estimar. El Analista no emite un segundo veredicto de '
    'riesgo, solo evidencia (señales con sus cifras) sobre POR QUÉ; usa esas señales para '
    'decidir sobre qué palanca actuar, nunca para corregir el riesgo del Predictor.\n\n'
    'No inventas cifras: cualquier número que cites en tu justificación tiene que venir '
    'de las señales del Analista o de la tool de simulación reanálisis, si la usas. '
    'Siempre declaras una limitación: sensibilidad no causal si usaste la simulación, o '
    'que te apoyas en tasas históricas de segmentos similares (no en este cliente en '
    'particular) si no la usaste.\n\n'
    'Propones UNA acción de retención concreta, no una lista de opciones. Respondes en '
    'español.'
)

GOAL = ('Proponer una acción de retención concreta y justificada para un cliente con '
        'riesgo de churn, apoyada en las señales del Analista y, cuando el caso lo '
        'requiera, en la simulación reanálisis.')


def enfoque_variante(variante: str) -> str:
    if variante == 'alto_valor':
        return (
            'Trabajas la cartera de alto valor: clientes en el cuartil alto de saldo. Cada acción la '
            'ejecuta y vigila el gestor personal asignado a ESTE cliente concreto — nunca una campaña '
            'automática ni una oferta genérica. Redacta "accion_principal" y "justificacion" dando voz '
            'al gestor y transmitiendo seguimiento individual (por ejemplo: "el gestor contacta al '
            'cliente para..." o "el gestor propone y hace seguimiento de..."), no como una oferta '
            'estándar aplicable a cualquiera. El canal es siempre "gestor personal".'
        )
    elif variante == 'cartera_masiva':
        return (
            'Trabajas la cartera masiva: volumen alto, trato estandarizado. Cada acción se aplica igual '
            'a cualquier cliente con un perfil similar — nunca una gestión personalizada de alto coste ni '
            'un seguimiento individual. Redacta "accion_principal" y "justificacion" como una acción '
            'escalable y general (por ejemplo: "se incluye en campaña automatizada de..." o "contact '
            'center ofrece..."), sin dar a entender trato ni seguimiento individualizado. El canal es '
            '"campaña automatizada" o "contact center", según la acción.'
        )
    else:
        raise ValueError(f"Variante de Recomendador no reconocida: {variante}")


def crear_agente_recomendador(variante: str, tool_simulacion, llm: Optional[LLM] = None) -> Agent:
    """variante: 'alto_valor' o 'cartera_masiva'. Dos agentes con el mismo objetivo y las
    mismas herramientas, pero con un backstory distinto."""
    enfoque = enfoque_variante(variante)
    return Agent(
        role=f'Agente Recomendador de Retención — {variante}',
        goal=GOAL,
        backstory=f'{enfoque}\n\n{_BACKSTORY_COMUN}',
        tools=[tool_simulacion],
        verbose=settings.VERBOSE_AGENTES,
        llm=llm or crear_llm_recomendador(),
    )


def descripcion_tarea_recomendacion(
    nombre_cliente: str,
    customer_data: dict,
    prediccion: dict,
    senales: list,
    ruta_info: dict,
    conclusion_analista: Optional[str] = None,
) -> str:
    datos_str = ", ".join(f"{k}: {v}" for k, v in customer_data.items())
    senales_str = "\n".join(f"- {s}" for s in senales)
    bloque_conclusion = (f"\nConclusión redactada por el Analista: {conclusion_analista}\n"
                         if conclusion_analista else "")

    if ruta_info['ruta'] == 2 and ruta_info['variante'] == 'alto_valor':
        instruccion_ruta = (
            "Ruta: RIESGO ALTO, cartera de alto valor. El Predictor ya marca a este cliente "
            "como riesgo alto: no hace falta que justifiques el riesgo, solo la acción. DEBES "
            "invocar la tool de simulación reanálisis antes de proponer nada: en la cartera "
            "de alto valor la acción se justifica con una cifra concreta de impacto sobre ESTE "
            "cliente, no solo con las tasas del histórico. Prioriza la palanca con mayor delta, "
            "no necesariamente la primera que parezca plausible."
        )
        ruta_pydantic = "riesgo_alto"
    elif ruta_info['ruta'] == 2:
        instruccion_ruta = (
            "Ruta: RIESGO ALTO. El Predictor ya marca a este cliente como riesgo alto: no "
            "hace falta que justifiques el riesgo, solo la acción. Apóyate primero en las "
            "señales del Analista para decidir la palanca; usa la tool de simulación "
            "reanálisis solo si las señales no bastan para elegir entre dos palancas "
            "igual de plausibles."
        )
        ruta_pydantic = "riesgo_alto"
    else:
        instruccion_ruta = (
            "Ruta: REVISIÓN. El Predictor y la agregación determinista del Analista no "
            "coinciden con claridad. DEBES invocar la tool de simulación reanálisis antes"
            "de proponer nada: permite identificar qué palanca presenta una mayor "
            "sensibilidad en la predicción del modelo para ESTE cliente, "
            "en lugar de basarse únicamente en las tasas históricas del segmento."
        )
        ruta_pydantic = "revision"
    if ruta_info['variante'] == 'alto_valor':
        instruccion_estilo = (
            "Recuerda: eres el gestor personal de este cliente. Escribe 'accion_principal' y "
            "'justificacion' en ese papel — seguimiento individual, trato a medida — no como una "
            "oferta estándar aplicable a cualquiera."
        )
    else:
        instruccion_estilo = (
            "Recuerda: esta acción se aplica igual a cualquier cliente con perfil similar. Escribe "
            "'accion_principal' y 'justificacion' como una acción escalable y general, sin dar a "
            "entender trato o seguimiento individualizado."
        )
    return (
        f"Cliente: {nombre_cliente}.\n"
        f"Datos del cliente: {datos_str}\n\n"
        f"Predicción del Predictor (riesgo oficial): {prediccion['churn_prediction']} "
        f"(0=no abandona, 1=abandona), probabilidad {prediccion['churn_probability_pct']:.2f}%.\n\n"
        f"Señales del Analista (evidencia, no un segundo veredicto de riesgo):\n{senales_str}\n"
        f"{bloque_conclusion}\n"
        f"{instruccion_ruta}\n\n"
        f"Motivo de esta ruta: {ruta_info['motivo']}\n\n"
        f"Rellena 'ruta' con exactamente '{ruta_pydantic}' y 'variante_agente' con "
        f"exactamente '{ruta_info['variante']}'; no los decidas tú, ya vienen fijados "
        f"por el enrutado en Python.\n\n"
        "En 'resumen_cliente', describe en 1-2 frases y en lenguaje natural quién es este cliente a "
        "partir de sus datos (edad, antigüedad, saldo, productos, etc.), sin mencionar el riesgo ni "
        "la acción — solo contexto.\n\n"
        "En 'resumen_riesgo', cita la probabilidad del Predictor y enumera TODAS las señales con "
        "multiplicador ('x') mayor que 1, no solo la más alta; ignora las menores que 1. Si ninguna "
        "supera 1, dilo explícitamente. No reevalúes el riesgo, solo sintetiza lo ya calculado.\n\n"
        "En 'limitaciones', nunca la dejes vacía: si usaste la tool reanálisis, indica que mide "
        "sensibilidad, no causalidad; si no la usaste, que la recomendación se basa en tasas "
        "históricas del segmento, no en una estimación individual de este cliente.\n\n"
        "Si una señal >1 de 'resumen_riesgo' no es una palanca intervenible (edad, país, saldo, etc.), "
        "menciónala en 'justificacion', pero en una frase aparte de las palancas intervenibles y con otro "
        "verbo: usa 'el caso se ve agravado por...' o 'además, el cliente presenta...', nunca 'la campaña "
        "aborda' ni 'la acción se dirige a' seguido de una señal no intervenible — ni siquiera si esa "
        "misma frase también menciona señales intervenibles."
        "Si en cambio una señal SÍ es intervenible pero la descartas como palanca, explica en "
        "'justificacion' por qué (ej. muestra pequeña). No ignores ninguna señal en silencio.\n\n"
        f"{instruccion_estilo}\n\n"
        "Propón una acción de retención concreta y viable para este cliente, dirigida a la palanca o "
        "palancas identificadas. No interpretes el resultado de la simulación como causalidad ni des "
        "por hecho que el cliente debe modificar esa variable: solo indica qué cambio reduciría más la "
        "probabilidad de churn según el modelo, no que vaya a evitarlo.\n\n"
        "Formula la acción como seguimiento del gestor, nunca como una orden al cliente. Para "
        "'NumOfProducts', no propongas cerrar ni cancelar un producto: propón que se revise con el "
        "cliente su cartera, se detecten productos infrautilizados o necesidades de consolidación, y se "
        "valoren ajustes junto con él, la decisión final es suya. Para 'HasCrCard', orienta la acción a "
        "revisar sus necesidades de tarjeta y ofrecer una solución adecuada. Para 'IsActiveMember', "
        "orienta la acción a revisar su nivel de uso de los servicios y, si procede, fomentarlo.\n\n"
        "Si más de una de las tres palancas intervenibles (activación, tarjeta, productos) tiene señal "
        "relevante para este cliente, combínalas en una única acción de seguimiento que cubra todas, no "
        "elijas solo una y descartes el resto. 'palanca_objetivo' debe nombrar todas las palancas que "
        "uses. Si citas el impacto simulado de varias palancas, dalo por separado, nunca sumado, la "
        "simulación no prueba combinaciones."
    )


EXPECTED_OUTPUT_RECOMENDACION = (
    "Un objeto con la acción de retención propuesta, la palanca a la que apunta, el "
    "canal de ejecución, la justificación apoyada en las señales citadas y las "
    "limitaciones a declarar."
)


def crear_tarea_recomendacion(
    nombre_cliente: str,
    customer_data: dict,
    prediccion: dict,
    senales: list,
    ruta_info: dict,
    agente: Agent,
    conclusion_analista: Optional[str] = None,
) -> Task:
    return Task(
        description=descripcion_tarea_recomendacion(
            nombre_cliente, customer_data, prediccion, senales, ruta_info, conclusion_analista),
        expected_output=EXPECTED_OUTPUT_RECOMENDACION,
        agent=agente,
        output_pydantic=RecomendacionChurn,
    )
