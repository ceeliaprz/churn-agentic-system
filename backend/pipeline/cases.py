"""Perfiles congelados y construcción determinista de casos (notebook agéntico, celdas 40, 47 y 48).

`construir_caso` reproduce en Python, sin LLM ni coste, lo que produciría el pipeline para
un cliente: la predicción del modelo y las 4 señales del Analista. Es la base del modo
determinista del frontend y de la evaluación de la sección 9.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

from backend.analytics.reference import BaseReferencia, get_base_referencia
from backend.analytics.segments import todas_las_senales
from backend.analytics.signals import balance_de_senales
from backend.ml.features import COLUMNAS_ID
from backend.ml.predictor import ModeloChurn, get_modelo

# Fila del test set (posición en X_test_raw.csv) del sexto perfil, localizado en el notebook
# recorriendo X_test_raw con construir_caso. Es el cliente "Gray".
INDICE_DESACUERDO_FRANCO = 1198

PERFILES = ['balance_cero', 'balance_alto', 'multi_producto',
            'inactivo_alemania', 'perfil_limpio', 'desacuerdo_franco']

# Cliente esperado por perfil (para detectar si cambia el test set)
CLIENTE_ESPERADO = {
    'balance_cero': 'Hsiung', 'balance_alto': 'Lo', 'multi_producto': 'Gardiner',
    'inactivo_alemania': "O'Donnell", 'perfil_limpio': 'Sun', 'desacuerdo_franco': 'Gray',
}


def primero(df_filtrado, fallback):
    """Devuelve la primera fila que cumpla el filtro, o el fallback si no hay ninguna."""
    return df_filtrado.iloc[0] if not df_filtrado.empty else fallback


def seleccionar_perfiles_test(X_test_raw: pd.DataFrame) -> dict:
    """Los 6 perfiles de prueba del notebook: 5 variados (celda 40) + desacuerdo franco (celda 47)."""
    perfiles_test = {
        # Saldo en el extremo bajo
        'balance_cero': primero(X_test_raw[X_test_raw['Balance'] == 0], X_test_raw.iloc[0]),

        # Saldo en el extremo alto
        'balance_alto': X_test_raw.sort_values('Balance', ascending=False).iloc[0],

        # Productos fuera de la zona estable
        'multi_producto': primero(X_test_raw[X_test_raw['NumOfProducts'] >= 3], X_test_raw.iloc[0]),

        # Eje de actividad y país de riesgo
        'inactivo_alemania': primero(
            X_test_raw[(X_test_raw['IsActiveMember'] == 0) & (X_test_raw['Geography'] == 'Germany')],
            X_test_raw.iloc[0]),

        # Sin nada llamativo en ningún eje: debería disparar pocas herramientas
        'perfil_limpio': primero(
            X_test_raw[(X_test_raw['IsActiveMember'] == 1)
                       & (X_test_raw['NumOfProducts'] == 2)
                       & (X_test_raw['Age'].between(28, 38))
                       & (X_test_raw['Geography'] != 'Germany')
                       & (X_test_raw['Balance'] > 0)],
            X_test_raw.iloc[0]),
    }
    # añadimos otro perfil que da desacuerdo entre predictor y analista
    perfiles_test['desacuerdo_franco'] = X_test_raw.loc[INDICE_DESACUERDO_FRANCO]
    return perfiles_test


def _a_nativo(v):
    """numpy → tipos nativos de Python (para JSON y para que el LLM vea 585, no np.int64(585))."""
    return v.item() if hasattr(v, 'item') else v


def datos_cliente(fila: pd.Series) -> dict:
    """Datos crudos del cliente sin identificadores (como `fila.drop(labels=[...]).to_dict()`)."""
    return {k: _a_nativo(v) for k, v in fila.drop(labels=COLUMNAS_ID, errors='ignore').to_dict().items()}


def construir_caso_desde_datos(nombre_cliente: str, customer_data: dict,
                               modelo: Optional[ModeloChurn] = None,
                               base: Optional[BaseReferencia] = None) -> dict:
    """Predicción + 4 señales + orientación calculada, sin LLM. Mismo formato que
    casos_recomendador.json."""
    modelo = modelo or get_modelo()
    base = base or get_base_referencia()

    pred = modelo.predict_churn(customer_data)
    senales_con_nombre = todas_las_senales(base, customer_data)
    senales = [resultado for _, resultado in senales_con_nombre]

    return {
        'cliente': nombre_cliente,
        'datos': customer_data,
        'prediccion': {
            'churn_prediction': int(pred['churn_prediction']),
            'churn_probability_pct': round(pred['churn_probability'] * 100, 2),
        },
        'senales': senales,
        'orientacion_calculada': balance_de_senales(senales),
    }


def construir_caso(fila: pd.Series, modelo: Optional[ModeloChurn] = None,
                   base: Optional[BaseReferencia] = None) -> dict:
    """Reproduce en Python lo que produciría el pipeline para un cliente: la predicción del
    modelo y las señales del analista. Determinista y sin coste, sirve como entrada fija para
    desarrollar el recomendador sin reejecutar predictor y analista en cada iteración."""
    return construir_caso_desde_datos(fila['Surname'], datos_cliente(fila), modelo, base)


def construir_casos_congelados(X_test_raw: pd.DataFrame, modelo: Optional[ModeloChurn] = None,
                               base: Optional[BaseReferencia] = None) -> dict:
    perfiles = seleccionar_perfiles_test(X_test_raw)
    return {nombre: construir_caso(fila, modelo, base) for nombre, fila in perfiles.items()}
