"""Persistencia: nunca se sobrescriben los resultados del notebook."""
import pytest

from config import settings
from backend.pipeline.storage import AlmacenResultados, escribir_json


def test_no_escribe_en_resultados_notebook():
    with pytest.raises(PermissionError):
        escribir_json(settings.RESULTS_NOTEBOOK_DIR / "prueba.json", {"a": 1})


def test_almacen_guarda_y_relee(tmp_path):
    ruta = tmp_path / "res.json"
    almacen = AlmacenResultados(ruta)
    almacen.guardar("x", {"cliente": "Ñúñez", "v": 1})
    assert AlmacenResultados(ruta).get("x") == {"cliente": "Ñúñez", "v": 1}
