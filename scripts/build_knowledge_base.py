"""Genera knowledge_base/reference_stats.json a partir de df_train.

    python scripts/build_knowledge_base.py
"""
import json

import _bootstrap  # noqa: F401

from config import settings
from backend.analytics.reference import get_base_referencia


def main():
    base = get_base_referencia()
    with open(settings.RUTA_REFERENCE_STATS, "w", encoding="utf-8") as f:
        json.dump(base.resumen(), f, ensure_ascii=False, indent=2)
    print(f"Escrito {settings.RUTA_REFERENCE_STATS}")
    print(f"- Tasa base de churn: {base.base_churn:.2f}%  | prod. estable: {base.prod_estable}")
    print(base.referencias_texto)


if __name__ == "__main__":
    main()
