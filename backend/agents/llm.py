"""Configuración de los LLM (Groq vía LiteLLM) y workaround de CrewAI (notebook, celdas 6 y 10).

Importar este módulo:
1. Carga GROQ_API_KEY desde .env (sustituye a Colab Secrets).
2. Aplica el workaround del bug de `cache_breakpoint` con Groq en crewai==1.15.21:
   CrewAI importa `mark_cache_breakpoint` localmente dentro de las funciones que lo usan,
   así que basta con sustituir el atributo del módulo.

Diferencia con el notebook (2026-10-03): los agentes usan `LLMConEspera`, que ante un rate
limit de Groq espera lo que indica el error y repite SOLO esa llamada al LLM. En el notebook
el único reintento era el de `ejecutar_con_reintentos`, que repite la tarea entera desde cero:
si el Analista consulta las 4 herramientas, sus llamadas suman más de 8.000 tokens en menos de
un minuto, así que cada reintento volvía a chocar con el límite en el mismo punto (caso
Houghton desde el chat). Los prompts, las tools y la lógica no cambian.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Optional

from crewai import LLM

from config import settings
from backend.pipeline.retries import aviso_actual

settings.cargar_env()


def aplicar_workaround_cache() -> None:
    """Workaround para el bug de cache_breakpoint con Groq."""
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg


aplicar_workaround_cache()

# Registro del consumo real de tokens (lo muestra el frontend; ver backend/pipeline/consumo.py)
from backend.pipeline import consumo  # noqa: E402
consumo.activar()


def espera_para_reintentar_llamada(error: Exception) -> Optional[float]:
    """Segundos a esperar antes de repetir una llamada que falló, o None si no tiene sentido
    repetirla (no es un rate limit, la petición sola ya supera el límite por minuto, o se ha
    agotado el límite diario)."""
    texto = str(error)
    t = texto.lower()
    es_rate_limit = "RateLimitError" in str(type(error)) or "rate_limit" in t or "rate limit" in t
    if not es_rate_limit:
        return None
    if "request too large" in t or "reduce your message size" in t:
        return None                      # esperar no sirve: la petición sola supera 8.000 tokens
    if "tokens per day" in t or "(tpd)" in t:
        return None                      # límite diario: hay que esperar horas, no segundos
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", texto)
    if not match:
        return settings.ESPERA_MAXIMA_LLAMADA_S
    sugerida = int(match.group(1) or 0) * 60 + float(match.group(2))
    return min(sugerida + settings.MARGEN_ESPERA_LLAMADA_S, settings.ESPERA_MAXIMA_LLAMADA_S)


class LLMConEspera(LLM):
    """LLM de CrewAI que, ante un rate limit por minuto, espera y repite solo esa llamada.

    El límite de Groq funciona como un cubo que se va rellenando: el error dice cuántos
    segundos faltan para que quepa la petición (p. ej. "try again in 8.8s"). Esperar eso y
    repetir la misma llamada conserva lo que el agente ya ha hecho (las tools ya ejecutadas),
    en lugar de empezar la tarea de nuevo.
    """

    def call(self, *args, **kwargs):
        for intento in range(1, settings.MAX_ESPERAS_POR_LLAMADA + 1):
            try:
                return super().call(*args, **kwargs)
            except Exception as e:
                espera = espera_para_reintentar_llamada(e)
                if espera is None or intento == settings.MAX_ESPERAS_POR_LLAMADA:
                    raise
                self._avisar(espera, intento)
                time.sleep(espera)

    async def acall(self, *args, **kwargs):
        for intento in range(1, settings.MAX_ESPERAS_POR_LLAMADA + 1):
            try:
                return await super().acall(*args, **kwargs)
            except Exception as e:
                espera = espera_para_reintentar_llamada(e)
                if espera is None or intento == settings.MAX_ESPERAS_POR_LLAMADA:
                    raise
                self._avisar(espera, intento)
                await asyncio.sleep(espera)

    @staticmethod
    def _avisar(espera: float, intento: int) -> None:
        mensaje = (f"⏳ Límite por minuto de Groq: espero {espera:.0f} s y repito solo la última llamada "
                   f"del agente ({intento}/{settings.MAX_ESPERAS_POR_LLAMADA})...")
        avisar = aviso_actual.get() or print
        try:
            avisar(mensaje)
        except Exception:  # noqa: BLE001
            # El aviso es informativo: si falla (p. ej. Streamlit desde el hilo de CrewAI, que lanza
            # NoSessionContext sin mensaje), no debe tumbar la llamada ni el análisis.
            print(mensaje)

    def __copy__(self):
        # LLM.__copy__ construye un LLM normal; se conserva la subclase
        copia = super().__copy__()
        copia.__class__ = LLMConEspera
        return copia


def crear_llm_predictor() -> LLM:
    # baja temperatura: queremos un resultado consistente, no creativo
    return LLMConEspera(**settings.LLM_PREDICTOR)


def crear_llm_analista() -> LLM:
    # bajamos temperatura: decisiones consistentes, no creatividad
    return LLMConEspera(**settings.LLM_ANALISTA)


def crear_llm_recomendador() -> LLM:
    return LLMConEspera(**settings.LLM_RECOMENDADOR)
