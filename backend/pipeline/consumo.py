"""Registro del consumo real de tokens de Groq de este repositorio (chat, agentes y scripts).

Groq no ofrece ninguna API para consultar el consumo diario: sus cabeceras solo informan del
límite por minuto (TPM) y de las peticiones por día. Por eso se registra aquí cada llamada que
termina bien, con un callback de LiteLLM (todas las llamadas a Groq del repo pasan por LiteLLM:
las del Asistente y las de los agentes de CrewAI). Las llamadas rechazadas (429) no cuentan,
igual que en Groq.

Limitación: solo ve lo que se ejecuta con este repo en este ordenador. El notebook de Colab u
otros programas que usen la misma clave no aparecen; el total exacto de la cuenta está en
https://console.groq.com.

Formato: una línea JSON por llamada en data/results/ejecuciones/consumo_groq.jsonl
({"ts": "2026-10-03T17:35:52Z", "tokens": 2405, "modelo": "openai/gpt-oss-20b"}).
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from backend.pipeline.storage import ruta_ejecuciones

_lock = threading.Lock()
_ruta: Optional[Path] = None


def ruta_registro() -> Path:
    return _ruta or ruta_ejecuciones("consumo_groq.jsonl")


def usar_ruta(ruta: Optional[Path]) -> None:
    """Cambia el fichero de registro (por defecto, data/results/ejecuciones/consumo_groq.jsonl)."""
    global _ruta
    _ruta = Path(ruta) if ruta else None


def registrar(tokens: int, modelo: str = "", ahora: Optional[datetime] = None) -> None:
    if not tokens:
        return
    ahora = ahora or datetime.now(timezone.utc)
    linea = json.dumps({'ts': ahora.strftime("%Y-%m-%dT%H:%M:%SZ"), 'tokens': int(tokens),
                        'modelo': modelo}, ensure_ascii=False)
    ruta = ruta_registro()
    with _lock:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with open(ruta, "a", encoding="utf-8") as f:
            f.write(linea + "\n")


def _callback_litellm(kwargs, respuesta, _inicio, _fin) -> None:
    """success_callback de LiteLLM: anota las llamadas a Groq que han terminado bien."""
    try:
        proveedor = (kwargs.get('litellm_params') or {}).get('custom_llm_provider') or ''
        if proveedor != 'groq':
            return
        uso = getattr(respuesta, 'usage', None)
        registrar(getattr(uso, 'total_tokens', 0) or 0, kwargs.get('model') or '')
    except Exception:  # noqa: BLE001 — el registro nunca debe romper una llamada
        pass


def activar() -> None:
    """Engancha el registro a LiteLLM (idempotente)."""
    import litellm
    if _callback_litellm not in litellm.success_callback:
        litellm.success_callback.append(_callback_litellm)


def resumen(ahora: Optional[datetime] = None) -> dict:
    """Tokens de hoy (día UTC, como la consola de Groq) y de las últimas 24 horas."""
    ahora = ahora or datetime.now(timezone.utc)
    hoy = ahora.strftime("%Y-%m-%d")
    hace_24h = ahora - timedelta(hours=24)
    total_hoy = total_24h = llamadas_hoy = 0
    ruta = ruta_registro()
    if ruta.exists():
        with _lock, open(ruta, encoding="utf-8") as f:
            for linea in f:
                try:
                    r = json.loads(linea)
                    ts = datetime.strptime(r['ts'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                except (ValueError, KeyError):
                    continue
                if r['ts'].startswith(hoy):
                    total_hoy += r['tokens']
                    llamadas_hoy += 1
                if ts >= hace_24h:
                    total_24h += r['tokens']
    return {'hoy': total_hoy, 'ultimas_24h': total_24h, 'llamadas_hoy': llamadas_hoy,
            'dia_utc': hoy}
