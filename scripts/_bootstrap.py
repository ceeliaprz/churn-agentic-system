"""Permite ejecutar los scripts aunque el proyecto no esté instalado en el entorno
(p. ej. con un Python distinto del .venv de `uv sync`), añadiendo la raíz del repo a sys.path."""
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

# En Windows, la consola puede no ser UTF-8: evita UnicodeEncodeError con "→", "⚠️", "✅"...
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
