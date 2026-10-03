"""Base histórica de referencia del Analista (notebook agéntico).

En el notebook, `df_train`, `BASE_CHURN`, `Q_BALANCE_POS`, `referencias_texto`, las anclas
(`_paises_altos`, `_prod_estable`...) y los docstrings `DOC_CONTEXTO` / `DOC_PRODUCTOS`
eran variables globales que las tools leían directamente. Aquí se agrupan en un único objeto
`BaseReferencia`, construido una vez a partir de df_train y pasado explícitamente a quien lo
necesite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import settings


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def cargar_df_train(ruta_x: Path = settings.RUTA_X_TRAIN_RAW,
                    ruta_y: Path = settings.RUTA_Y_TRAIN_RAW) -> pd.DataFrame:
    """Combina X_train_raw e y_train_raw en df_train."""
    X_train_raw = pd.read_csv(ruta_x)
    y_train_raw = pd.read_csv(ruta_y)
    assert len(X_train_raw) == len(y_train_raw)

    df_train = X_train_raw.copy()
    if 'Exited' not in df_train.columns:
        df_train['Exited'] = (y_train_raw['Exited'] if 'Exited' in y_train_raw.columns
                              else y_train_raw.iloc[:, 0])
    return df_train


def cargar_x_test(ruta: Path = settings.RUTA_X_TEST_RAW) -> pd.DataFrame:
    return pd.read_csv(ruta)


# ---------------------------------------------------------------------------
# Estadísticos de referencia
# ---------------------------------------------------------------------------
def calcular_referencias(df: pd.DataFrame) -> dict:
    """Calcula estadísticos descriptivos de df_train para dar contexto objetivo
    al analista, evitando que el LLM tenga que 'adivinar' si un valor
    es alto/bajo sin referencia. No decide nada, solo describe la distribución
    de la base para que el agente pueda juzgar qué datos del cliente merecen
    una consulta."""
    referencias = {}
    for col in ['Balance', 'Age']:
        referencias[col] = {
            'media': df[col].mean(),
            'std': df[col].std(),
            'q25': df[col].quantile(0.25),
            'q75': df[col].quantile(0.75),
        }
    return referencias


def _enumerar(items: list) -> str:
    if not items:
        return "ninguno"
    if len(items) == 1:
        return str(items[0])
    return ", ".join(str(i) for i in items[:-1]) + f" y {items[-1]}"


def _niveles_por_lift(tasas: dict, tamanos: dict, base: float, minimo_n: int = 100):
    """Separa los niveles de una variable en los que quedan por encima y por debajo de la
    media global, ignorando los de muestra insuficiente."""
    altos, bajos = [], []
    for nivel, tasa in tasas.items():
        if tamanos[nivel] < minimo_n:
            continue
        lift = tasa / base
        if lift >= settings.LIFT_ENCIMA:
            altos.append(nivel)
        elif lift <= settings.LIFT_DEBAJO:
            bajos.append(nivel)
    return altos, bajos


@dataclass
class BaseReferencia:
    """Todo lo que el Analista, el router y la simulación necesitan de la base histórica."""
    df_train: pd.DataFrame
    base_churn: float                     # BASE_CHURN: tasa media de abandono (%)
    q_balance_pos: np.ndarray             # Q_BALANCE_POS: cuartiles del saldo positivo
    referencias: dict                     # referencias_train
    referencias_texto: str
    paises_altos: list = field(default_factory=list)
    paises_bajos: list = field(default_factory=list)
    edad_altas: list = field(default_factory=list)
    edad_bajas: list = field(default_factory=list)
    prod_estable: int = 2                 # _prod_estable
    prod_altos: list = field(default_factory=list)
    doc_contexto: str = ""
    doc_productos: str = ""

    def resumen(self) -> dict:
        """Versión serializable (para knowledge_base/reference_stats.json y el frontend)."""
        df = self.df_train
        banda_edad = (df['Age'] // 10 * 10)

        def tasas(agrupador) -> dict:
            g = df.groupby(agrupador)['Exited']
            return {str(k): {'n': int(n), 'churn_pct': round(float(t) * 100, 4),
                             'lift': round(float(t) * 100 / self.base_churn, 4)}
                    for k, n, t in zip(g.size().index, g.size().values, g.mean().values)}

        return {
            'n_train': int(len(df)),
            'base_churn_pct': round(float(self.base_churn), 6),
            'q_balance_pos': [round(float(q), 4) for q in self.q_balance_pos],
            'referencias': {col: {k: round(float(v), 4) for k, v in r.items()}
                            for col, r in self.referencias.items()},
            'tasas_por_segmento': {
                'Geography': tasas('Geography'),
                'FranjaEdad': tasas(banda_edad.rename('FranjaEdad')),
                'NumOfProducts': tasas('NumOfProducts'),
                'IsActiveMember_HasCrCard': tasas(
                    df['IsActiveMember'].astype(str) + "_" + df['HasCrCard'].astype(str)),
                'HasZeroBalance': tasas((df['Balance'] == 0).astype(int).rename('HasZeroBalance')),
            },
            'anclas': {
                'paises_altos': self.paises_altos, 'paises_bajos': self.paises_bajos,
                'edad_altas': self.edad_altas, 'edad_bajas': self.edad_bajas,
                'prod_estable': self.prod_estable,
                'prod_altos': [int(p) for p in self.prod_altos],
            },
            'docstrings': {'Contexto': self.doc_contexto, 'Productos': self.doc_productos},
            'referencias_texto': self.referencias_texto,
        }


def construir_base_referencia(df_train: pd.DataFrame) -> BaseReferencia:
    """Reproduce la celda 27 del notebook sobre el df_train dado."""
    referencias_train = calcular_referencias(df_train)
    referencias_texto = "\n".join(
        f"- {col}: media={r['media']:.1f}, rango típico (Q25-Q75)=[{r['q25']:.1f} - {r['q75']:.1f}]"
        for col, r in referencias_train.items() if col in ('Balance', 'Age')
    )

    base_churn = df_train['Exited'].mean() * 100
    q_balance_pos = df_train.loc[df_train['Balance'] > 0, 'Balance'].quantile([.25, .50, .75]).values

    # --- País ---
    paises_altos, paises_bajos = _niveles_por_lift(
        (df_train.groupby('Geography')['Exited'].mean() * 100).to_dict(),
        df_train.groupby('Geography').size().to_dict(), base_churn,
        minimo_n=settings.MINIMO_N_ANCLAS)

    # --- Edad, en tramos de diez años ---
    banda_edad = (df_train['Age'] // 10 * 10)
    edad_altas, edad_bajas = _niveles_por_lift(
        (df_train.groupby(banda_edad)['Exited'].mean() * 100).to_dict(),
        df_train.groupby(banda_edad).size().to_dict(), base_churn,
        minimo_n=settings.MINIMO_N_ANCLAS)
    edad_altas = [f"{int(b)}-{int(b)+9}" for b in sorted(edad_altas)]
    edad_bajas = [f"{int(b)}-{int(b)+9}" for b in sorted(edad_bajas)]

    # --- Productos ---
    tasas_prod = df_train.groupby('NumOfProducts')['Exited'].mean() * 100
    prod_estable = int(tasas_prod.idxmin())
    prod_altos, _ = _niveles_por_lift(
        tasas_prod.to_dict(), df_train.groupby('NumOfProducts').size().to_dict(),
        base_churn, minimo_n=settings.MINIMO_N_ANCLAS_PRODUCTOS)

    doc_contexto = (
        "Devuelve la tasa histórica de abandono del país del cliente y de su franja de edad "
        "(tramos de diez años), ambas comparadas con la media global. "
        f"Los mercados no se comportan igual: {_enumerar(paises_altos)} se sitúa por encima de la "
        f"media y {_enumerar(paises_bajos)} por debajo. Por edad, las franjas de "
        f"{_enumerar(edad_altas)} años quedan por encima y las de {_enumerar(edad_bajas)} años por "
        "debajo. Consúltala cuando el cliente pertenezca a alguno de los grupos que se sitúan por "
        "encima. El banco no puede intervenir sobre estas variables: "
        "el resultado dimensiona la gravedad del caso, no señala una palanca."
    )

    doc_productos = (
        "Devuelve la tasa histórica de abandono de los clientes con el mismo número de productos, "
        "comparada con la media global. La relación no es monótona y contradice la intuición de que "
        f"más productos implican más fidelidad: contratar {prod_estable} productos es el punto más "
        f"estable de la escala, mientras que {_enumerar(prod_altos)} producto(s) se sitúan por "
        "encima de la media."
    )

    return BaseReferencia(
        df_train=df_train, base_churn=base_churn, q_balance_pos=q_balance_pos,
        referencias=referencias_train, referencias_texto=referencias_texto,
        paises_altos=paises_altos, paises_bajos=paises_bajos,
        edad_altas=edad_altas, edad_bajas=edad_bajas,
        prod_estable=prod_estable, prod_altos=prod_altos,
        doc_contexto=doc_contexto, doc_productos=doc_productos,
    )


@lru_cache(maxsize=1)
def get_base_referencia() -> BaseReferencia:
    """Base por defecto (data/processed/), construida una sola vez por proceso."""
    return construir_base_referencia(cargar_df_train())
