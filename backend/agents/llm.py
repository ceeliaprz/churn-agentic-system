"""Configuración de los LLM (Groq vía LiteLLM) y workaround de CrewAI (notebook, celdas 6 y 10).

Importar este módulo:
1. Carga GROQ_API_KEY desde .env (sustituye a Colab Secrets).
2. Aplica el workaround del bug de `cache_breakpoint` con Groq en crewai==1.15.21:
   CrewAI importa `mark_cache_breakpoint` localmente dentro de las funciones que lo usan,
   así que basta con sustituir el atributo del módulo.
"""
from __future__ import annotations

from crewai import LLM

from config import settings

settings.cargar_env()


def aplicar_workaround_cache() -> None:
    """Workaround para el bug de cache_breakpoint con Groq."""
    import crewai.llms.cache as _crewai_cache
    _crewai_cache.mark_cache_breakpoint = lambda msg: msg


aplicar_workaround_cache()


def crear_llm_predictor() -> LLM:
    # baja temperatura: queremos un resultado consistente, no creativo
    return LLM(**settings.LLM_PREDICTOR)


def crear_llm_analista() -> LLM:
    # bajamos temperatura: decisiones consistentes, no creatividad
    return LLM(**settings.LLM_ANALISTA)


def crear_llm_recomendador() -> LLM:
    return LLM(**settings.LLM_RECOMENDADOR)
