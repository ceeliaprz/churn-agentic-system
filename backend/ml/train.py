"""Entrenamiento del modelo final (ml_churn_model.ipynb)

El EDA, la validación cruzada de los 4 modelos, el test de Wilcoxon y la búsqueda de
hiperparámetros NO se repiten aquí: esa selección ya está hecha y justificada en el notebook
(se conserva en notebooks/). Este módulo reproduce solo el camino que produce los artefactos:
split estratificado → escalado → LightGBM con los hiperparámetros elegidos → guardado.

Los artefactos oficiales son los copiados de Drive (models/ y
data/processed/). Por eso `entrenar_y_guardar` escribe por defecto en
`artefactos_regenerados/` y `verificar_contra_oficiales` comprueba que salen iguales.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import settings
from backend.ml.features import NUM_COLS, TARGET, transformar_features

TEST_SIZE = 0.2
RANDOM_STATE = 42

# Mejores hiperparámetros encontrados con RandomizedSearchCV (celda 70-71)
HIPERPARAMETROS_LGBM = dict(
    n_estimators=600,
    num_leaves=127,
    max_depth=6,
    learning_rate=0.01,
    min_child_samples=30,
    subsample=0.7,
    subsample_freq=10,
    colsample_bytree=0.6,
    reg_alpha=0.5,
    reg_lambda=0.0,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    verbose=-1,
)


@dataclass
class ResultadoEntrenamiento:
    model: LGBMClassifier
    scaler: StandardScaler
    feature_order: list
    X_train_raw: pd.DataFrame
    X_test_raw: pd.DataFrame
    y_train_raw: pd.Series
    y_test_raw: pd.Series
    predicciones: pd.DataFrame
    metricas: dict


def entrenar(ruta_csv: Path = settings.RUTA_CHURN_CSV) -> ResultadoEntrenamiento:
    df = pd.read_csv(ruta_csv)

    # Split sobre el dataset crudo (el que usan los agentes). El notebook hacía dos splits
    # con la misma semilla y estratificación (uno sobre bank_df y otro sobre el crudo) y
    # comprobaba con assert que los índices coincidían; aquí se hace uno solo y se deriva
    # la versión procesada de él, lo que garantiza la misma partición.
    X_agent = df.drop(columns=[TARGET])
    y_agent = df[TARGET]
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_agent, y_agent, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_agent
    )

    X_train = transformar_features(X_train_raw)
    X_test = transformar_features(X_test_raw)

    # fit SOLO con train, para evitar data leakage
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUM_COLS] = scaler.fit_transform(X_train[NUM_COLS])
    X_test_scaled[NUM_COLS] = scaler.transform(X_test[NUM_COLS])
    feature_order = list(X_train_scaled.columns)

    model = LGBMClassifier(**HIPERPARAMETROS_LGBM)
    model.fit(X_train_scaled, y_train_raw)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    predicciones = pd.DataFrame({
        'index': X_test_raw.index,
        'pred_original': y_test_raw.values,
        'pred_proba': y_proba,
        'pred_clase': y_pred,
    })
    metricas = {
        'roc_auc': float(roc_auc_score(y_test_raw, y_proba)),
        'pr_auc': float(average_precision_score(y_test_raw, y_proba)),
        'classification_report': classification_report(y_test_raw, y_pred, output_dict=True),
    }
    return ResultadoEntrenamiento(model, scaler, feature_order, X_train_raw, X_test_raw,
                                  y_train_raw, y_test_raw, predicciones, metricas)


def guardar(res: ResultadoEntrenamiento, salida: Path = settings.ARTEFACTOS_REGENERADOS_DIR) -> Path:
    salida = Path(salida)
    (salida / "models").mkdir(parents=True, exist_ok=True)
    (salida / "processed").mkdir(parents=True, exist_ok=True)

    joblib.dump(res.model, salida / "models" / "best_model_lgbm.pkl")
    joblib.dump(res.scaler, salida / "models" / "scaler.pkl")
    joblib.dump(res.feature_order, salida / "models" / "feature_order.pkl")

    res.X_train_raw.to_csv(salida / "processed" / "X_train_raw.csv", index=False)
    res.y_train_raw.to_csv(salida / "processed" / "y_train_raw.csv", index=False)
    res.X_test_raw.to_csv(salida / "processed" / "X_test_raw.csv", index=False)
    res.y_test_raw.to_csv(salida / "processed" / "y_test_raw.csv", index=False)
    res.predicciones.to_csv(salida / "processed" / "predicciones_best_lgbm.csv", index=False)
    return salida


def verificar_contra_oficiales(res: ResultadoEntrenamiento, tolerancia: float = 1e-6) -> dict:
    """Compara lo regenerado con los artefactos oficiales de Drive."""
    oficiales_X_test = pd.read_csv(settings.RUTA_X_TEST_RAW)
    oficiales_X_train = pd.read_csv(settings.RUTA_X_TRAIN_RAW)
    oficiales_pred = pd.read_csv(settings.RUTA_PREDICCIONES_ORIGINALES)
    feature_order_oficial = list(joblib.load(settings.RUTA_FEATURE_ORDER))
    scaler_oficial = joblib.load(settings.RUTA_SCALER)

    mismo_split = (
        res.X_test_raw.reset_index(drop=True).equals(oficiales_X_test)
        and res.X_train_raw.reset_index(drop=True).equals(oficiales_X_train)
    )
    diff_proba = np.abs(res.predicciones['pred_proba'].values - oficiales_pred['pred_proba'].values)
    return {
        'mismo_split': bool(mismo_split),
        'mismo_feature_order': res.feature_order == feature_order_oficial,
        'mismo_scaler': bool(np.allclose(res.scaler.mean_, scaler_oficial.mean_)
                             and np.allclose(res.scaler.scale_, scaler_oficial.scale_)),
        'misma_clase': bool((res.predicciones['pred_clase'].values
                             == oficiales_pred['pred_clase'].values).all()),
        'max_diff_proba': float(diff_proba.max()),
        'misma_proba': bool(diff_proba.max() < tolerancia),
        'roc_auc': res.metricas['roc_auc'],
        'pr_auc': res.metricas['pr_auc'],
    }
