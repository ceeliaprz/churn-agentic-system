"""Preprocesado (sin datos): orden de columnas y validaciones."""
import pandas as pd
import pytest

from backend.ml.features import FEATURE_ORDER_ESPERADO, transformar_features, validar_cliente

CLIENTE = {'CreditScore': 585, 'Geography': 'Germany', 'Gender': 'Female', 'Age': 36,
           'Tenure': 7, 'Balance': 0.0, 'NumOfProducts': 3, 'HasCrCard': 1,
           'IsActiveMember': 0, 'EstimatedSalary': 94283.09}


def test_orden_columnas_igual_que_entrenamiento():
    df = transformar_features(pd.DataFrame([{**CLIENTE, 'RowNumber': 1, 'CustomerId': 2,
                                              'Surname': 'X', 'Exited': 1}]))
    assert list(df.columns) == FEATURE_ORDER_ESPERADO


def test_features_derivadas_y_encoding():
    fila = transformar_features(pd.DataFrame([CLIENTE])).iloc[0]
    assert fila['HasZeroBalance'] == 1
    assert fila['ProductRisk'] == 1
    assert fila['Gender'] == 1
    assert (fila['Geography_Germany'], fila['Geography_Spain']) == (1, 0)


@pytest.mark.parametrize("cambio, mensaje", [
    ({'Geography': 'Italy'}, "Geography inválida"),
    ({'Gender': 'X'}, "Gender inválido"),
])
def test_validaciones(cambio, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        validar_cliente({**CLIENTE, **cambio})


def test_faltan_columnas():
    datos = dict(CLIENTE)
    del datos['Age']
    with pytest.raises(ValueError, match="Faltan columnas"):
        validar_cliente(datos)
