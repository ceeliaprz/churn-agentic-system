"""Carga de artefactos y predicción de churn (notebook agéntico, celdas 8 y 13).

`ModeloChurn` agrupa los tres artefactos que en el notebook eran variables globales
(`model`, `scaler`, `FEATURE_ORDER`). `predict_churn(customer_data)` se mantiene como
función de módulo con la misma firma y la misma salida que en el notebook, usando un
modelo cargado una sola vez.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from config import settings
from backend.ml.features import escalar, transformar_features, validar_cliente


@dataclass
class ModeloChurn:
    model: object
    scaler: object
    feature_order: list

    @classmethod
    def cargar(cls,
               ruta_modelo: Path = settings.RUTA_MODELO,
               ruta_scaler: Path = settings.RUTA_SCALER,
               ruta_feature_order: Path = settings.RUTA_FEATURE_ORDER) -> "ModeloChurn":
        return cls(
            model=joblib.load(ruta_modelo),
            scaler=joblib.load(ruta_scaler),
            feature_order=list(joblib.load(ruta_feature_order)),
        )

    def preparar(self, df_crudo: pd.DataFrame) -> pd.DataFrame:
        """Datos crudos (como en el CSV de Kaggle) → matriz lista para el modelo."""
        return escalar(transformar_features(df_crudo), self.scaler, self.feature_order)

    def predict_churn(self, customer_data: dict) -> dict:
        """Recibe los datos crudos de un cliente (como en el CSV original) y devuelve
        la predicción de churn del modelo de ML.

        customer_data debe incluir las claves:
        CreditScore, Geography, Gender, Age, Tenure, Balance,
        NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
        """
        validar_cliente(customer_data)
        df = self.preparar(pd.DataFrame([customer_data]))

        pred = self.model.predict(df)[0]
        proba = self.model.predict_proba(df)[0][1]

        return {
            'churn_prediction': int(pred),
            'churn_probability': float(proba),
        }

    def predict_lote(self, df_crudo: pd.DataFrame) -> pd.DataFrame:
        """Predicción vectorizada de muchos clientes a la vez (para la evaluación).
        Mismo resultado que llamar a predict_churn fila a fila, pero mucho más rápido."""
        X = self.preparar(df_crudo)
        return pd.DataFrame({
            'churn_prediction': self.model.predict(X).astype(int),
            'churn_probability': self.model.predict_proba(X)[:, 1],
        }, index=df_crudo.index)


@lru_cache(maxsize=1)
def get_modelo() -> ModeloChurn:
    """Modelo por defecto (artefactos de models/), cargado una sola vez por proceso."""
    return ModeloChurn.cargar()


def predict_churn(customer_data: dict) -> dict:
    """Atajo con la misma firma que en el notebook, usando el modelo por defecto."""
    return get_modelo().predict_churn(customer_data)
