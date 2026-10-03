"""Orquestación del sistema multiagente (notebook agéntico, celdas 61, 63 y 64).

Tres modos, de menor a mayor coste de Groq:

1. `ejecutar_caso_determinista`  — 0 tokens. Predictor = modelo en Python, Analista = las 4
   tools en Python, router, simulación y comparador tradicional. Sin recomendación LLM.
   Podría servir como fallback si no quedan tokens o se cae el servicio de GROQ.
2. `ejecutar_recomendador`       — solo el Recomendador con LLM, sobre un caso determinista
   (equivale a la prueba controlada notebook). ≈12k-41k tokens.
3. `ejecutar_pipeline_cliente`   — Predictor + Analista + Recomendador con LLM, cada uno en su
   propio Crew de una sola tarea, en secuencia, guardado por etapa y reanudable
   ≈25–35k tokens por cliente (reintentos incluidos).

Cada ejecución crea su propio registro de llamadas, sus tools y sus agentes: no hay estado
global compartido (en Colab había que reejecutar en cadena tools → Agent → tareas → Crew
al redefinir algo; aquí siempre se construyen frescos).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from backend.analytics.counterfactual import simular_palancas, texto_simulacion
from backend.analytics.reference import BaseReferencia, get_base_referencia
from backend.decision.router import enrutar_caso, enrutar_cliente
from backend.evaluation.baseline import recomendacion_tradicional
from backend.ml.predictor import ModeloChurn, get_modelo
from backend.pipeline.cases import construir_caso_desde_datos
from backend.pipeline.storage import AlmacenResultados

Evento = Callable[[str], None]


@dataclass
class ContextoSistema:
    """Dependencias compartidas (solo lectura) de todas las ejecuciones."""
    modelo: ModeloChurn
    base: BaseReferencia

    @classmethod
    def por_defecto(cls) -> "ContextoSistema":
        return cls(modelo=get_modelo(), base=get_base_referencia())


def _tokens(resultado) -> Optional[int]:
    uso = getattr(resultado, 'token_usage', None)
    return uso.total_tokens if uso else None


# ---------------------------------------------------------------------------
# Modo 1: determinista
# ---------------------------------------------------------------------------
def ejecutar_caso_determinista(nombre_cliente: str, customer_data: dict,
                               ctx: Optional[ContextoSistema] = None) -> dict:
    ctx = ctx or ContextoSistema.por_defecto()
    caso = construir_caso_desde_datos(nombre_cliente, customer_data, ctx.modelo, ctx.base)
    ruta_info = enrutar_caso(caso, ctx.base.df_train)
    simulacion = simular_palancas(ctx.modelo, ctx.base.prod_estable, customer_data)
    return {
        **caso,
        'ruta_info': ruta_info,
        'simulacion': simulacion,
        'simulacion_texto': texto_simulacion(simulacion),
        'recomendacion_tradicional': recomendacion_tradicional(nombre_cliente, ruta_info),
    }


# ---------------------------------------------------------------------------
# Modo 2: solo Recomendador con LLM
# ---------------------------------------------------------------------------
async def ejecutar_recomendador(nombre_cliente: str, customer_data: dict, prediccion: dict,
                                senales: list, ruta_info: dict,
                                ctx: Optional[ContextoSistema] = None,
                                conclusion_analista: Optional[str] = None,
                                on_evento: Optional[Evento] = print) -> Optional[dict]:
    """Devuelve {'recomendacion': dict, 'tokens_recomendador': int} o None si agota reintentos.
    No debe llamarse en Ruta 1 (el router cierra sin Recomendador por diseño)."""
    if ruta_info['ruta'] == 1:
        raise ValueError("Ruta 1: el caso se cierra sin Recomendador según el router.")

    from crewai import Crew, Process

    from backend.agents.recommender import crear_agente_recomendador, crear_tarea_recomendacion
    from backend.agents.tools import crear_tool_simulacion
    from backend.pipeline.retries import ejecutar_con_reintentos

    ctx = ctx or ContextoSistema.por_defecto()
    agente = crear_agente_recomendador(
        ruta_info['variante'], crear_tool_simulacion(ctx.modelo, ctx.base.prod_estable))
    tarea = crear_tarea_recomendacion(
        nombre_cliente=nombre_cliente, customer_data=customer_data, prediccion=prediccion,
        senales=senales, ruta_info=ruta_info, agente=agente,
        conclusion_analista=conclusion_analista,
    )
    crew = Crew(agents=[agente], tasks=[tarea], process=Process.sequential)
    resultado = await ejecutar_con_reintentos(crew, on_evento=on_evento)
    if resultado is None:
        return None
    return {'recomendacion': resultado.pydantic.model_dump(),
            'tokens_recomendador': _tokens(resultado)}


# ---------------------------------------------------------------------------
# Un agente cada vez (los usan el pipeline completo y el Asistente del chat)
# ---------------------------------------------------------------------------
async def ejecutar_predictor(nombre_cliente: str, customer_data: dict,
                             ctx: Optional[ContextoSistema] = None,
                             on_evento: Optional[Evento] = print) -> Optional[dict]:
    """Agente Predictor en su propio Crew de una tarea. None si agota los reintentos."""
    from crewai import Crew, Process

    from backend.agents.predictor import crear_agente_predictor, crear_tarea_prediccion
    from backend.agents.tools import crear_tool_predictor
    from backend.pipeline.retries import ejecutar_con_reintentos

    ctx = ctx or ContextoSistema.por_defecto()
    agente = crear_agente_predictor(crear_tool_predictor(ctx.modelo))
    tarea = crear_tarea_prediccion(nombre_cliente, customer_data, agente)
    crew = Crew(agents=[agente], tasks=[tarea], process=Process.sequential)
    resultado = await ejecutar_con_reintentos(crew, on_evento=on_evento)
    if resultado is None:
        return None
    return {'prediccion': resultado.pydantic.model_dump(), 'tokens_predictor': _tokens(resultado)}


async def ejecutar_analista(nombre_cliente: str, customer_data: dict,
                            ctx: Optional[ContextoSistema] = None,
                            on_evento: Optional[Evento] = print,
                            motivo_reanalisis: Optional[str] = None) -> Optional[dict]:
    """Agente Analista en su propio Crew de una tarea. Devuelve su salida y la orientación
    calculada con balance_de_senales sobre las tools que invocó de verdad.
    `motivo_reanalisis` (solo desde el chat) se añade a la tarea cuando se repite un análisis."""
    from crewai import Crew, Process

    from backend.agents.analyst import analizar_cliente, crear_agente_analista
    from backend.agents.tool_registry import RegistroLlamadas
    from backend.agents.tools import crear_tools_analista
    from backend.analytics.signals import balance_de_senales
    from backend.pipeline.retries import ejecutar_con_reintentos

    ctx = ctx or ContextoSistema.por_defecto()
    registro_llamadas = RegistroLlamadas()
    agente = crear_agente_analista(crear_tools_analista(ctx.base, registro_llamadas),
                                   ctx.base.referencias_texto)
    tarea = analizar_cliente(nombre_cliente, customer_data, agente, motivo_reanalisis)
    crew = Crew(agents=[agente], tasks=[tarea], process=Process.sequential)
    # El registro se vacía antes de CADA intento: si un rate limit obliga a repetir la tarea,
    # balance_de_senales solo debe ver las tools del intento que termina (si no, los lifts se
    # contarían dos veces y el producto podría cruzar los umbrales).
    resultado = await ejecutar_con_reintentos(crew, on_evento=on_evento,
                                              antes_de_cada_intento=registro_llamadas.iniciar)
    if resultado is None:
        return None
    analisis = resultado.pydantic
    return {
        'senales': analisis.senales,
        'conclusion_analista': analisis.conclusion,
        'orientacion_llm': analisis.orientacion,
        'orientacion_calculada': balance_de_senales(registro_llamadas.llamadas),
        'tokens_analista': _tokens(resultado),
        # Añadidos respecto al notebook (instrumentación, no cambian el flujo):
        'tools_llamadas': registro_llamadas.tools_llamadas(),
        'senales_tools': [ll['resultado'] for ll in registro_llamadas.llamadas],
    }


# ---------------------------------------------------------------------------
# Modo 3: pipeline completo con LLM
# ---------------------------------------------------------------------------
async def ejecutar_pipeline_cliente(clave: str, nombre_cliente: str, customer_data: dict,
                                    almacen: AlmacenResultados,
                                    ctx: Optional[ContextoSistema] = None,
                                    forzar: bool = True,
                                    on_evento: Optional[Evento] = print) -> Optional[dict]:
    """Ejecuta Predictor, Analista y, si el router lo requiere, el Recomendador, sobre un único
    cliente. Cada agente corre en su propio Crew de una sola tarea, en secuencia (el
    tier gratuito no permite paralelismo real ni resumir una tarea a medias — ver documento de
    estado). Si la clave ya está completa en el almacén, no gasta cuota (salvo forzar=True). Si
    ya tiene Predictor+Analista pero le falta el Recomendador (p.ej. porque agotó los reintentos
    la vez anterior), reintenta SOLO el Recomendador, reutilizando lo ya guardado.

    Igual que en el notebook, `forzar=True` por defecto (reejecuta todo). Para aprovechar la
    reanudación hay que pasar `forzar=False`."""
    ctx = ctx or ContextoSistema.por_defecto()
    avisar = on_evento or (lambda _msg: None)

    registro_previo = almacen.get(clave)
    # Un registro sin ruta_info (p. ej. el chat solo ejecutó el Predictor) se trata como no empezado
    if registro_previo is not None and not registro_previo.get('ruta_info'):
        registro_previo = None
    ya_completo = registro_previo is not None and (
        registro_previo['ruta_info']['ruta'] == 1 or registro_previo.get('recomendacion') is not None
    )

    if registro_previo is not None and ya_completo and not forzar:
        avisar(f"'{clave}' ya está completo en {almacen.ruta.name}, no se llama a Groq "
               f"(usa forzar=True para reejecutar todo desde cero).")
        return registro_previo

    if registro_previo is not None and not ya_completo and not forzar:
        avisar(f"'{clave}' ya tiene Predictor y Analista guardados; solo falta el Recomendador. "
               f"Reintentando solo esa etapa (no se repite Predictor ni Analista).")
        registro = registro_previo
        nombre_cliente = registro['cliente']
        customer_data = registro['datos']
        ruta_info = registro['ruta_info']
    else:
        # --- Etapa 1: Predictor ---
        avisar("Etapa 1/3: Predictor")
        pred = await ejecutar_predictor(nombre_cliente, customer_data, ctx, on_evento)
        if pred is None:
            avisar(f"⚠️ '{clave}': Predictor no completó tras los reintentos. Nada guardado.")
            return None

        # --- Etapa 2: Analista ---
        avisar("Etapa 2/3: Analista")
        analisis = await ejecutar_analista(nombre_cliente, customer_data, ctx, on_evento)
        if analisis is None:
            avisar(f"⚠️ '{clave}': Analista no completó tras los reintentos. Nada guardado "
                   f"(el coste del Predictor ya se ha gastado, pero no se guarda un caso a medias).")
            return None

        ruta_info = enrutar_cliente(
            churn_probability_pct=pred['prediccion']['churn_probability_pct'],
            orientacion_calculada=analisis['orientacion_calculada'],
            balance=customer_data['Balance'],
            df_train=ctx.base.df_train,
        )

        registro = {
            'cliente': nombre_cliente,
            'datos': customer_data,
            'prediccion': pred['prediccion'],
            'senales': analisis['senales'],
            'conclusion_analista': analisis['conclusion_analista'],
            'orientacion_llm': analisis['orientacion_llm'],
            'orientacion_calculada': analisis['orientacion_calculada'],
            'ruta_info': ruta_info,
            'tokens_predictor': pred['tokens_predictor'],
            'tokens_analista': analisis['tokens_analista'],
            'recomendacion': None,
            'tokens_recomendador': None,
            'tools_llamadas': analisis['tools_llamadas'],
            'senales_tools': analisis['senales_tools'],
        }

        avisar(f"[{clave}] {nombre_cliente} — Predictor {pred['prediccion']['churn_probability_pct']:.2f}% | "
               f"Analista: {analisis['orientacion_calculada']} | Ruta: {ruta_info['nombre_ruta']}")

        # Guarda ya Predictor+Analista, antes de intentar el Recomendador: si este falla,
        # no se pierde lo ya conseguido.
        almacen.guardar(clave, registro)

    # --- Etapa 3: Recomendador, solo si el router lo pide ---
    if ruta_info['ruta'] != 1:
        avisar("Etapa 3/3: Recomendador")
        salida = await ejecutar_recomendador(
            nombre_cliente=nombre_cliente, customer_data=customer_data,
            prediccion=registro['prediccion'], senales=registro['senales'],
            ruta_info=ruta_info, ctx=ctx,
            conclusion_analista=registro.get('conclusion_analista'),
            on_evento=on_evento,
        )
        if salida is None:
            avisar(f"⚠️ '{clave}': Recomendador no completó tras los reintentos. "
                   f"Predictor+Analista quedan guardados; vuelve a llamar a esta función más tarde "
                   f"con forzar=False para reintentar solo esta etapa.")
        else:
            registro['recomendacion'] = salida['recomendacion']
            registro['tokens_recomendador'] = salida['tokens_recomendador']
            avisar(f"  → Recomendación: {registro['recomendacion']['accion_principal']}")
    else:
        avisar("  → Ruta 1: se cierra sin Recomendador, por diseño.")

    almacen.guardar(clave, registro)
    avisar(f"Guardado en {almacen.ruta.name} (clave: '{clave}')")
    return registro
