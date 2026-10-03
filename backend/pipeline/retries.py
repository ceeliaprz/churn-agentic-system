"""Reintentos ante los límites del tier gratuito de Groq (notebook agéntico, celda 36).

- Rate limit (8.000 tokens/min): espera lo que sugiere el error (+5 s), nunca menos de 60 s.
- Nombre de tool corrupto por el formato harmony de gpt-oss (`actividad<|channel|>commentary`)
  o casing incorrecto de parámetros (`tool_use_failed`): reintento rápido a los 5 s.
- Cualquier otro error: se propaga.

Ojo (documentado en el estado del proyecto): reintentar un Crew repite TODA su tarea desde
cero, y `token_usage` se acumula sobre el mismo Crew entre reintentos.

Diferencia con el notebook: los mensajes se envían a `on_evento` (por defecto `print`) para
que el frontend pueda mostrarlos, y `ejecutar_sync` permite llamarlo fuera de Jupyter
(allí el `await` de nivel superior era válido; en un script o en Streamlit no).
"""
from __future__ import annotations

import asyncio
import contextvars
import re
from typing import Callable, Optional

from config import settings

Evento = Callable[[str], None]

# Destino de los avisos de LLMConEspera (backend/agents/llm.py) mientras corre un Crew: el LLM
# no conoce on_evento, así que se le pasa por contexto (asyncio.to_thread copia el contexto).
aviso_actual: contextvars.ContextVar[Optional[Evento]] = contextvars.ContextVar("aviso_actual",
                                                                                default=print)


def extraer_espera_sugerida(error_msg: str, default: float = 65.0) -> float:
    match = re.search(r"try again in ([\d.]+)s", error_msg)
    sugerida = float(match.group(1)) + 5 if match else default
    return max(sugerida, settings.ESPERA_MINIMA_RATE_LIMIT_S)   # nunca menos de una ventana completa


async def ejecutar_con_reintentos(crew, max_intentos: int = settings.MAX_INTENTOS,
                                  espera_default: float = settings.ESPERA_DEFAULT_S,
                                  on_evento: Optional[Evento] = print,
                                  antes_de_cada_intento: Optional[Callable[[], None]] = None):
    """`antes_de_cada_intento` se llama justo antes de cada kickoff (p. ej. para vaciar el
    registro de llamadas a tools): un reintento repite la tarea entera desde cero, así que lo
    registrado en un intento fallido no debe mezclarse con el intento que sí termina."""
    on_evento = on_evento or (lambda _msg: None)
    aviso_actual.set(on_evento)
    for intento in range(1, max_intentos + 1):
        try:
            if antes_de_cada_intento is not None:
                antes_de_cada_intento()
            return await crew.kickoff_async()
        except Exception as e:
            error_str = str(e)
            es_rate_limit = "RateLimitError" in str(type(e)) or "rate_limit" in error_str.lower()
            es_tool_corrupta = "tool_use_failed" in error_str or "<|channel|>" in error_str

            if es_rate_limit:
                espera = extraer_espera_sugerida(error_str, default=espera_default * intento)
                on_evento(f"⚠️ Rate limit, esperando {espera:.1f}s (intento {intento}/{max_intentos})...")
            elif es_tool_corrupta:
                espera = settings.ESPERA_TOOL_CORRUPTA_S
                on_evento(f"⚠️ Nombre de tool corrupto, reintento inmediato "
                          f"({intento}/{max_intentos})...")
            else:
                raise
            await asyncio.sleep(espera)
    return None


def ejecutar_sync(corutina):
    """Ejecuta una corrutina desde código síncrono (scripts, Streamlit)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(corutina)
    # Ya hay un bucle en marcha (p. ej. Jupyter): ejecutar en un hilo aparte
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, corutina).result()
