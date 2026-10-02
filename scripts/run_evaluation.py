"""Sección de evaluaciones: cobertura de riesgo sobre los 2.000 clientes del test set y su
validación con la etiqueta real (sin coste de Groq).

    python scripts/run_evaluation.py
"""
import _bootstrap  # noqa: F401

import pandas as pd

from config.settings import RUTA_Y_TEST_RAW, UMBRAL_PROB_BAJA
from backend.analytics.reference import cargar_x_test
from backend.evaluation.coverage import (comparar_cobertura_riesgo, resumen_cobertura,
                                         validar_rescatados)
from backend.pipeline.storage import ruta_ejecuciones


def main():
    resultado = comparar_cobertura_riesgo(cargar_x_test())
    r = resumen_cobertura(resultado)
    print(f"Clientes en el test set: {r['total']}")
    print(f"El modelo tradicional (umbral único en {UMBRAL_PROB_BAJA}%) cerraría sin acción: "
          f"{r['cerrados_tradicional']} ({r['pct_cerrados_tradicional']:.2f}%)")
    print(f"De ellos, el sistema completo los mantiene en revisión por desacuerdo del Analista: "
          f"{r['rescatados']} ({r['pct_rescatados_total']:.2f}% del total, "
          f"{r['pct_rescatados_de_cerrados']:.1f}% de los cerrados)")
    print("Reparto de rutas del sistema:", r['reparto_rutas'])

    salida = ruta_ejecuciones("comparacion_cobertura_riesgo.csv")
    salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(salida, index=False)
    print(f"Guardado en {salida}")

    # Validación con la etiqueta real: ¿los rescatados abandonan realmente más?
    y_test = pd.read_csv(RUTA_Y_TEST_RAW)['Exited']
    v = validar_rescatados(resultado, y_test)
    print("\nValidación con la etiqueta real (y_test):")
    print(v['tabla'].to_string(index=False))
    print(f"\np-valor (rescatados abandonan más que ambos_cierran, Fisher unilateral): "
          f"{v['p_valor']:.4f}")

    salida_val = ruta_ejecuciones("tabla_validacion_rescatados.csv")
    v['tabla'].to_csv(salida_val, index=False)
    print(f"Guardado en {salida_val}")


if __name__ == "__main__":
    main()
