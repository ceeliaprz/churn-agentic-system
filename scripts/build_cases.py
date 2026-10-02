"""Reconstruye los 6 casos congelados (sin LLM), muestra el reparto de rutas y el baseline.

    python scripts/build_cases.py            # imprime y compara con data/results/notebook/
    python scripts/build_cases.py --guardar  # además guarda en data/results/ejecuciones/
"""
import argparse

import _bootstrap  # noqa: F401
import pandas as pd

from backend.analytics.reference import cargar_x_test, get_base_referencia
from backend.decision.router import construir_reparto
from backend.evaluation.baseline import generar_baseline
from backend.pipeline.cases import construir_casos_congelados
from backend.pipeline.storage import escribir_json, leer_json, ruta_ejecuciones, ruta_notebook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--guardar", action="store_true")
    args = parser.parse_args()

    base = get_base_referencia()
    casos = construir_casos_congelados(cargar_x_test())
    for nombre, caso in casos.items():
        print(f"\n=== {nombre} — {caso['cliente']} ===")
        print(f"Predicción: {caso['prediccion']['churn_prediction']} "
              f"({caso['prediccion']['churn_probability_pct']}%)  |  "
              f"Orientación: {caso['orientacion_calculada']}")
        for s in caso['senales']:
            print(f"  - {s}")

    pd.set_option('display.max_colwidth', 120)
    pd.set_option('display.width', 200)
    reparto = construir_reparto(casos, base.df_train)
    print("\n", reparto.drop(columns=['motivo']))
    print("\nDistribución de rutas:\n", reparto['ruta'].value_counts())

    baseline = generar_baseline(casos, base.df_train)
    print("\nCoincide con casos_recomendador.json del notebook:",
          casos == leer_json(ruta_notebook("casos_recomendador.json")))
    print("Coincide con baseline_tradicional.json del notebook:",
          baseline == leer_json(ruta_notebook("baseline_tradicional.json")))

    if args.guardar:
        escribir_json(ruta_ejecuciones("casos_recomendador.json"), casos)
        escribir_json(ruta_ejecuciones("baseline_tradicional.json"), baseline)
        print("Guardado en data/results/ejecuciones/")


if __name__ == "__main__":
    main()
