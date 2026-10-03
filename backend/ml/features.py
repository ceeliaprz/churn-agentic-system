"""Preprocesado del dataset.

Mimso preprocesado en los dos notebooks:
- `ml_churn_model.ipynb`, celdas 9, 26, 32 y 36 (con `pd.get_dummies` y mapeos por texto).
- `notebook_pipeline_agentico.ipynb`, celda 13, dentro de `predict_churn` (reescrito a mano).

Aquí se unifica en `transformar_features`, que usan tanto `train.py` como `predictor.py`.
El orden de columnas resultante es exactamente el del entrenamiento original (importa:
LightGBM con `colsample_bytree` muestrea columnas por posición).
"""
from __future__ import annotations

import pandas as pd

# Columnas que debe tener cada fila (cliente) de entrada
REQUIRED_COLS = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember',
                 'EstimatedSalary']

# Columnas que no aportan señal predictiva (identificadores) y target
COLUMNAS_ID = ['RowNumber', 'CustomerId', 'Surname']
TARGET = 'Exited'

# Columnas numéricas que necesitan escalado
NUM_COLS = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']

# Valores válidos de las columnas categóricas
VALID_GEOGRAPHY = ['France', 'Germany', 'Spain']
VALID_GENDER = ['Male', 'Female']

# Orden de columnas que produce transformar_features (y que guarda feature_order.pkl)
FEATURE_ORDER_ESPERADO = [
    'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'HasZeroBalance', 'ProductRisk',
    'Geography_Germany', 'Geography_Spain',
]


def validar_cliente(customer_data: dict) -> None:
    missing = [c for c in REQUIRED_COLS if c not in customer_data]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para predecir: {missing}")

    if customer_data['Geography'] not in VALID_GEOGRAPHY:
        raise ValueError(f"Geography inválida: {customer_data['Geography']}")

    if customer_data['Gender'] not in VALID_GENDER:
        raise ValueError(f"Gender inválido: {customer_data['Gender']}")


def transformar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza, features derivadas y encoding (sin escalado).
    1. Eliminar identificadores y target si vienen incluidos.
    2. Features derivadas: HasZeroBalance y ProductRisk.
    3. Encoding: Gender (Male=0, Female=1) y one-hot de Geography con France como referencia.
    """
    df = df.copy()

    # 1. Eliminar columnas irrelevantes si vienen incluidas (identificadores, target)
    df = df.drop(columns=COLUMNAS_ID + [TARGET], errors='ignore')

    # 2. Features derivadas
    df['HasZeroBalance'] = (df['Balance'] == 0).astype(int)
    df['ProductRisk'] = df['NumOfProducts'].map({1: 0, 2: 0, 3: 1, 4: 1})

    # 3. Encoding
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    df['Geography_Germany'] = (df['Geography'] == 'Germany').astype(int)
    df['Geography_Spain'] = (df['Geography'] == 'Spain').astype(int)
    df = df.drop(columns=['Geography'])

    return df


def escalar(df: pd.DataFrame, scaler, feature_order: list) -> pd.DataFrame:
    """Escalado con el scaler YA ajustado en train (solo transform) y reordenación de
    columnas exactamente como en entrenamiento."""
    df = df.copy()
    df[NUM_COLS] = scaler.transform(df[NUM_COLS])
    return df[feature_order]
