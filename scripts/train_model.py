"""Regenera el modelo desde data/raw/Churn_Modelling.csv y lo compara con los artefactos oficiales.

    python scripts/train_model.py                 # entrena, guarda en artefactos_regenerados/ y verifica
    python scripts/train_model.py --sin-guardar   # solo verifica

Nunca sobrescribe models/ ni data/processed/ (decisión D1: los oficiales son los de Drive).
"""
import argparse
import json

import _bootstrap  # noqa: F401

from config import settings
from backend.ml.train import entrenar, guardar, verificar_contra_oficiales


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--salida", default=str(settings.ARTEFACTOS_REGENERADOS_DIR))
    parser.add_argument("--sin-guardar", action="store_true")
    args = parser.parse_args()

    print("Entrenando LightGBM con los hiperparámetros finales...")
    res = entrenar()
    if not args.sin_guardar:
        print(f"Artefactos regenerados en: {guardar(res, args.salida)}")

    verificacion = verificar_contra_oficiales(res)
    print(json.dumps(verificacion, indent=2))
    ok = all(verificacion[k] for k in ('mismo_split', 'mismo_feature_order', 'mismo_scaler',
                                       'misma_clase', 'misma_proba'))
    print("✅ Reproduce exactamente los artefactos oficiales" if ok
          else "⚠️ Hay diferencias con los artefactos oficiales")


if __name__ == "__main__":
    main()
