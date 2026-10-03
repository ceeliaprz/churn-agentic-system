"""Fixtures comunes.

Los tests marcados `datos` necesitan data/ y models/ (se saltan si faltan). Los que comparan
con el notebook ejecutan con `exec` celdas concretas de notebooks/notebook_pipeline_agentico.ipynb,
sustituyendo CrewAI por stubs, para comprobar que el repo reproduce el código original.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from config import settings  # noqa: E402

ARCHIVOS_NECESARIOS = [
    settings.RUTA_MODELO, settings.RUTA_SCALER, settings.RUTA_FEATURE_ORDER,
    settings.RUTA_X_TRAIN_RAW, settings.RUTA_Y_TRAIN_RAW, settings.RUTA_X_TEST_RAW,
    settings.RUTA_PREDICCIONES_ORIGINALES,
]
HAY_DATOS = all(p.exists() for p in ARCHIVOS_NECESARIOS)
NOTEBOOK_AGENTICO = settings.NOTEBOOKS_DIR / "notebook_pipeline_agentico.ipynb"


def pytest_collection_modifyitems(config, items):
    saltar = pytest.mark.skip(reason="faltan datos/artefactos en data/ o models/")
    for item in items:
        if "datos" in item.keywords and not HAY_DATOS:
            item.add_marker(saltar)


@pytest.fixture(scope="session")
def modelo():
    from backend.ml.predictor import get_modelo
    return get_modelo()


@pytest.fixture(scope="session")
def base():
    from backend.analytics.reference import get_base_referencia
    return get_base_referencia()


@pytest.fixture(scope="session")
def x_test():
    from backend.analytics.reference import cargar_x_test
    return cargar_x_test()


@pytest.fixture(scope="session")
def casos(x_test, modelo, base):
    from backend.pipeline.cases import construir_casos_congelados
    return construir_casos_congelados(x_test, modelo, base)


def leer_json_notebook(nombre: str) -> dict:
    with open(settings.RESULTS_NOTEBOOK_DIR / nombre, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Acceso a celdas del notebook original
# ---------------------------------------------------------------------------
def celda_notebook(marcador: str, ruta: Path = NOTEBOOK_AGENTICO) -> str:
    """Código de la primera celda que contiene `marcador`, sin imports de crewai/google ni
    comandos de shell (`!pip`), listo para exec con stubs."""
    nb = json.loads(ruta.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and marcador in src:
            lineas = [l for l in src.splitlines()
                      if not l.strip().startswith(("from crewai", "import crewai", "!", "from google"))
                      and "display(" not in l]
            return "\n".join(lineas)
    raise LookupError(f"No se encontró en el notebook una celda con: {marcador!r}")


class _Registro:
    """Stubs de CrewAI: Agent/Task/LLM devuelven sus kwargs, @tool devuelve la función."""
    @staticmethod
    def kwargs(**kw):
        return kw

    @staticmethod
    def tool(_nombre):
        return lambda f: f


@pytest.fixture
def ns_stub():
    return {"Agent": _Registro.kwargs, "Task": _Registro.kwargs, "LLM": _Registro.kwargs,
            "Crew": _Registro.kwargs, "Process": type("P", (), {"sequential": "sequential"}),
            "tool": _Registro.tool, "__name__": "notebook"}


@pytest.fixture(autouse=True)
def consumo_aislado(tmp_path):
    """Los tests nunca escriben en el registro real de consumo de Groq."""
    from backend.pipeline import consumo
    consumo.usar_ruta(tmp_path / "consumo_groq.jsonl")
    yield
    consumo.usar_ruta(None)
