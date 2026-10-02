"""Experimento de variabilidad de tools del Analista (sección 6.6).

    python scripts/run_variability.py                    # analiza los CSV del notebook (0 tokens)
    python scripts/run_variability.py --ejecutar --n 5   # reejecuta con Groq (consume mucha cuota)
"""
import argparse

import _bootstrap  # noqa: F401

from config import settings
from backend.evaluation.variability import (anadir_tokens_por_run, cargar_experimento,
                                            estabilidad_por_perfil, frecuencia_tools_por_perfil)


def analizar(ruta):
    df = anadir_tokens_por_run(cargar_experimento(ruta))
    print(f"\n##### {ruta.name}")
    print(f"Coste por ejecución — media: {df['tokens_run'].mean():.0f} tokens, "
          f"mín: {df['tokens_run'].min():.0f}, máx: {df['tokens_run'].max():.0f}")
    print("=== Estabilidad por perfil (Jaccard medio entre pares de runs) ===")
    for perfil, j in estabilidad_por_perfil(df).items():
        print(f"{perfil}: {j:.2f}" if j is not None else f"{perfil}: menos de 2 ejecuciones válidas")
    print("=== Tools más frecuentes por perfil ===")
    for perfil, info in frecuencia_tools_por_perfil(df).items():
        print(f"\n{perfil} (media {info['media_tools']:.1f} tools):")
        for t, n in info['frecuencias'].items():
            print(f"   {n}/{info['n_runs']}  {t}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ejecutar", action="store_true")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    if not args.ejecutar:
        for ruta in sorted(settings.RESULTS_NOTEBOOK_DIR.glob("experimento_variabilidad_*.csv")):
            analizar(ruta)
        return

    from backend.analytics.reference import cargar_x_test
    from backend.evaluation.variability import ejecutar_experimento
    from backend.pipeline.cases import seleccionar_perfiles_test
    from backend.pipeline.retries import ejecutar_sync
    from backend.pipeline.storage import ruta_ejecuciones

    perfiles = seleccionar_perfiles_test(cargar_x_test())
    perfiles.pop('desacuerdo_franco')  # el experimento original usó los 5 primeros
    ruta = ruta_ejecuciones("experimento_variabilidad.csv")
    ejecutar_sync(ejecutar_experimento(perfiles, ruta, n_repeticiones=args.n))
    analizar(ruta)


if __name__ == "__main__":
    main()
