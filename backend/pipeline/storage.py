"""Persistencia de resultados en JSON (sustituye los `json.dump` repartidos por el notebook).

Regla del repo: los resultados del notebook (data/results/notebook/) son de solo lectura;
las ejecuciones nuevas van a data/results/ejecuciones/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from config import settings


def leer_json(ruta: Path) -> dict:
    ruta = Path(ruta)
    if not ruta.exists():
        return {}
    with open(ruta, encoding='utf-8') as f:
        return json.load(f)


def escribir_json(ruta: Path, datos: dict) -> None:
    ruta = Path(ruta)
    if settings.RESULTS_NOTEBOOK_DIR.resolve() in ruta.resolve().parents:
        raise PermissionError(
            f"No se escribe en {settings.RESULTS_NOTEBOOK_DIR}: son los resultados que respaldan "
            f"la memoria. Usa {settings.RESULTS_EJECUCIONES_DIR}.")
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)


class AlmacenResultados:
    """Diccionario {clave: registro} respaldado por un JSON, guardado tras cada cambio
    (así una ejecución interrumpida no pierde las etapas ya completadas)."""

    def __init__(self, ruta: Path):
        self.ruta = Path(ruta)
        self.datos: dict = leer_json(self.ruta)

    def get(self, clave: str) -> Optional[dict]:
        return self.datos.get(clave)

    def guardar(self, clave: str, registro: dict) -> None:
        self.datos[clave] = registro
        escribir_json(self.ruta, self.datos)

    def claves(self) -> list:
        return list(self.datos)


def ruta_ejecuciones(nombre: str) -> Path:
    return settings.RESULTS_EJECUCIONES_DIR / nombre


def ruta_notebook(nombre: str) -> Path:
    return settings.RESULTS_NOTEBOOK_DIR / nombre
