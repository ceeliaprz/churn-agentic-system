"""Ejecuta el sistema sobre un cliente del test set.

Elegir cliente (uno de):
    --perfil desacuerdo_franco      perfiles congelados: balance_cero, balance_alto, multi_producto,
                                    inactivo_alemania, perfil_limpio, desacuerdo_franco
    --indice 1198                   posición en X_test_raw.csv
    --customer-id 15749540

Modo:
    (por defecto)   determinista, 0 tokens
    --llm           pipeline completo con Groq (≈25–35k tokens/cliente; límite 200k/día)
    --solo-recomendador   señales deterministas + Recomendador con Groq

    --forzar        con --llm: reejecuta aunque ya esté guardado (por defecto se reanuda)

Los resultados con LLM se guardan en data/results/ejecuciones/resultados_pipeline.json.
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401

from config import settings
from backend.analytics.reference import cargar_x_test
from backend.pipeline.cases import PERFILES, datos_cliente, seleccionar_perfiles_test
from backend.pipeline.orchestrator import ejecutar_caso_determinista
from backend.pipeline.storage import AlmacenResultados, ruta_ejecuciones


def elegir_fila(args, X):
    """Devuelve (clave, fila). La clave es siempre customer_<CustomerId>, la misma que usa el
    chat, así que ambos comparten los resultados guardados (y no se paga dos veces)."""
    if args.perfil:
        fila = seleccionar_perfiles_test(X)[args.perfil]
    elif args.indice is not None:
        fila = X.iloc[args.indice]
    else:
        filas = X[X['CustomerId'] == args.customer_id]
        if filas.empty:
            sys.exit(f"CustomerId {args.customer_id} no está en el test set")
        fila = filas.iloc[0]
    return f"customer_{int(fila['CustomerId'])}", fila


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    grupo = parser.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--perfil", choices=PERFILES)
    grupo.add_argument("--indice", type=int)
    grupo.add_argument("--customer-id", type=int)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--solo-recomendador", action="store_true")
    parser.add_argument("--forzar", action="store_true")
    args = parser.parse_args()

    clave, fila = elegir_fila(args, cargar_x_test())
    nombre, datos = fila['Surname'], datos_cliente(fila)

    det = ejecutar_caso_determinista(nombre, datos)
    print(f"\n=== {clave} — {nombre} ===")
    print(f"Predictor: {det['prediccion']['churn_probability_pct']}%  |  "
          f"Orientación calculada: {det['orientacion_calculada']}")
    for s in det['senales']:
        print(f"  - {s}")
    print(f"Ruta: {det['ruta_info']['nombre_ruta']} ({det['ruta_info']['variante']}) — "
          f"{det['ruta_info']['motivo']}")
    print(det['simulacion_texto'])
    print(f"Recomendación tradicional: {det['recomendacion_tradicional']['accion_principal']}")

    if not (args.llm or args.solo_recomendador):
        return

    if not settings.hay_api_key():
        sys.exit("Falta GROQ_API_KEY: copia .env.example como .env y pon tu clave.")

    from backend.pipeline.orchestrator import ejecutar_pipeline_cliente, ejecutar_recomendador
    from backend.pipeline.retries import ejecutar_sync

    if args.solo_recomendador:
        if det['ruta_info']['ruta'] == 1:
            print("Ruta 1: el router cierra el caso sin Recomendador.")
            return
        salida = ejecutar_sync(ejecutar_recomendador(
            nombre, datos, det['prediccion'], det['senales'], det['ruta_info']))
        almacen = AlmacenResultados(ruta_ejecuciones("resultados_recomendador.json"))
        if salida:
            almacen.guardar(clave, salida['recomendacion'])
            print(json.dumps(salida, ensure_ascii=False, indent=2))
        return

    almacen = AlmacenResultados(ruta_ejecuciones("resultados_pipeline.json"))
    registro = ejecutar_sync(ejecutar_pipeline_cliente(clave, nombre, datos, almacen,
                                                       forzar=args.forzar))
    if registro:
        print(json.dumps(registro, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
