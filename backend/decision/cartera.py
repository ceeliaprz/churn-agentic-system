"""Vista de cartera: la capa determinista del sistema aplicada a los 2.000 clientes del test set.

Parte de la cobertura ya calculada (`data/results/notebook/comparacion_cobertura_riesgo.csv`,
la misma de la memoria): probabilidad del modelo, orientación calculada con las 4 herramientas
del Analista en Python y ruta del router. No llama a ningún LLM.

Si un cliente ya tiene un análisis real del Agente Analista guardado (desde el chat o con
`scripts/run_pipeline.py`), se usa ese análisis en lugar del cálculo con las 4 herramientas,
y se indica en `origen`. Así la vista nunca contradice lo que el sistema ya ha analizado.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import settings
from backend.decision.router import enrutar_cliente

NOMBRES_RUTA_CSV = {'cierre_sin_recomendador': 1, 'recomendador_riesgo_alto': 2,
                    'recomendador_revision': 3}
PAISES = {'france': 'France', 'francia': 'France', 'germany': 'Germany', 'alemania': 'Germany',
          'spain': 'Spain', 'españa': 'Spain', 'espana': 'Spain'}

ORIGEN_CALCULADO = "cálculo con las 4 herramientas (sin agente)"


def cargar_cobertura(ruta=None) -> pd.DataFrame:
    ruta = ruta or settings.RESULTS_NOTEBOOK_DIR / "comparacion_cobertura_riesgo.csv"
    return pd.read_csv(ruta)


def _variantes(balances: pd.Series, df_train: pd.DataFrame) -> np.ndarray:
    """Mismo cálculo que router.percentil_balance_numerico, vectorizado para toda la cartera."""
    ordenados = np.sort(df_train['Balance'].to_numpy())
    percentil = np.searchsorted(ordenados, balances.to_numpy(), side='left') / len(ordenados) * 100
    percentil = np.where(balances.to_numpy() <= 0, 0.0, percentil)
    return np.where(percentil >= settings.UMBRAL_VALOR_ALTO, 'alto_valor', 'cartera_masiva')


def construir_vista_cartera(cobertura: pd.DataFrame, x_test: pd.DataFrame, df_train: pd.DataFrame,
                            registros: Optional[dict] = None) -> pd.DataFrame:
    """Una fila por cliente del test set con probabilidad, orientación, ruta, variante y origen.

    `cobertura` va en el mismo orden de filas que `x_test` (se comprueba). `registros` son los
    resultados guardados del pipeline/chat ({clave: registro}); los que tienen análisis del
    Agente Analista sustituyen al cálculo con las 4 herramientas.
    """
    if len(cobertura) != len(x_test) or not (cobertura['cliente'].values == x_test['Surname'].values).all():
        raise ValueError("La cobertura no está alineada con X_test_raw")

    vista = pd.DataFrame({
        'customer_id': x_test['CustomerId'].astype(int).values,
        'apellido': x_test['Surname'].values,
        'pais': x_test['Geography'].values,
        'edad': x_test['Age'].astype(int).values,
        'saldo': x_test['Balance'].astype(float).values,
        'productos': x_test['NumOfProducts'].astype(int).values,
        'activo': x_test['IsActiveMember'].astype(int).values,
        'probabilidad_pct': cobertura['probabilidad_pct'].astype(float).values,
        'orientacion_calculada': cobertura['orientacion_calculada'].values,
        'ruta': cobertura['ruta_sistema'].map(NOMBRES_RUTA_CSV).astype(int).values,
        'origen': ORIGEN_CALCULADO,
        'herramientas_agente': None,   # nº de herramientas del análisis real, si lo hay
    })
    vista['variante'] = np.where(vista['ruta'] == 1, None, _variantes(x_test['Balance'], df_train))

    for clave, reg in (registros or {}).items():
        # Clave del almacén compartido por el chat y la CLI: "customer_<CustomerId>"
        if not str(clave).startswith('customer_') or not str(clave)[9:].isdigit():
            continue
        if not reg.get('orientacion_calculada'):
            continue
        filas = vista.index[vista['customer_id'] == int(str(clave)[9:])]
        if filas.empty:
            continue
        ri = reg.get('ruta_info')
        if not ri:
            # Analizado desde el chat sin pasar por el Predictor: la ruta se calcula con la
            # probabilidad del modelo de la vista (la misma que daría la tool del Predictor)
            ri = enrutar_cliente(churn_probability_pct=float(vista.loc[filas[0], 'probabilidad_pct']),
                                 orientacion_calculada=reg['orientacion_calculada'],
                                 balance=float(vista.loc[filas[0], 'saldo']), df_train=df_train)
        n_tools = len(reg.get('tools_llamadas') or [])
        vista.loc[filas, ['orientacion_calculada', 'ruta', 'variante', 'origen', 'herramientas_agente']] = [
            reg['orientacion_calculada'], int(ri['ruta']), ri.get('variante'),
            f"análisis del Agente Analista ({n_tools} de 4 herramientas)", n_tools]
        if reg.get('prediccion'):
            vista.loc[filas, 'probabilidad_pct'] = float(reg['prediccion']['churn_probability_pct'])
    return vista


def filtrar_cartera(vista: pd.DataFrame, pais: Optional[str] = None, ruta: Optional[int] = None,
                    variante: Optional[str] = None) -> pd.DataFrame:
    sel = vista
    if pais:
        sel = sel[sel['pais'] == PAISES.get(str(pais).strip().lower(), str(pais).strip())]
    if ruta:
        sel = sel[sel['ruta'] == int(ruta)]
    if variante:
        sel = sel[sel['variante'] == str(variante).strip()]
    return sel.sort_values('probabilidad_pct', ascending=False)
