"""balance_de_senales y router (sin datos, con un df_train sintético)."""
import pandas as pd
import pytest

from backend.analytics.signals import balance_de_senales, lift_combinado
from backend.decision.router import enrutar_cliente, percentil_balance_numerico

# Señales reales de Hsiung (casos_recomendador.json): 0.36 × 1.33 × 0.67 × 0.78 × 0.54 ≈ 0.135
SENALES_HSIUNG = [
    "Clientes con 2 producto(s) (n=3672): churn 7.4% vs 20.4% base → x0.36, muy por debajo de la media",
    "Clientes inactivos y con tarjeta (n=2762): churn 27.1% vs 20.4% base → x1.33, por encima de la media",
    "Clientes con saldo 0 € (n=2891): churn 13.6% vs 20.4% base → x0.67, por debajo de la media",
    "[País] Clientes de France (n=3993): churn 15.9% vs 20.4% base → x0.78, por debajo de la media\n"
    "[Edad] Clientes de 30-39 años (n=3486): churn 10.9% vs 20.4% base → x0.54, muy por debajo de la media",
]


def test_balance_hsiung_por_debajo():
    assert lift_combinado(SENALES_HSIUNG) == pytest.approx(0.36 * 1.33 * 0.67 * 0.78 * 0.54)
    assert balance_de_senales(SENALES_HSIUNG) == "riesgo_por_debajo_de_la_media"


def test_balance_acepta_registro_de_dicts():
    registro = [{'tool': 't', 'resultado': s} for s in SENALES_HSIUNG]
    assert balance_de_senales(registro) == balance_de_senales(SENALES_HSIUNG)


@pytest.mark.parametrize("senales, esperado", [
    ([], "riesgo_en_linea_con_la_media"),
    (["→ x4.03"], "riesgo_por_encima_de_la_media"),
    (["→ x1.00"], "riesgo_en_linea_con_la_media"),
    (["→ x0.36", "→ x1.26", "→ x1.25"], "riesgo_por_debajo_de_la_media"),  # O'Donnell ≈ 0.567
])
def test_balance_casos(senales, esperado):
    assert balance_de_senales(senales) == esperado


@pytest.fixture
def df_train_sintetico():
    return pd.DataFrame({'Balance': [0.0] * 25 + [float(i) for i in range(1, 76)]})


def test_percentil(df_train_sintetico):
    assert percentil_balance_numerico(0, df_train_sintetico) == 0.0
    assert percentil_balance_numerico(50.5, df_train_sintetico) == pytest.approx(75.0)


@pytest.mark.parametrize("prob, orientacion, balance, ruta, variante", [
    (6.07, "riesgo_por_debajo_de_la_media", 0.0, 1, None),        # Hsiung
    (95.13, "riesgo_por_encima_de_la_media", 1000.0, 2, "alto_valor"),
    (97.59, "riesgo_por_debajo_de_la_media", 10.0, 2, "cartera_masiva"),  # manda el Predictor
    (32.83, "riesgo_por_debajo_de_la_media", 10.0, 3, "cartera_masiva"),  # zona intermedia
    (17.37, "riesgo_por_encima_de_la_media", 10.0, 3, "cartera_masiva"),  # desacuerdo franco
    (17.37, "riesgo_en_linea_con_la_media", 10.0, 3, "cartera_masiva"),   # Gray en el pipeline real
])
def test_rutas(df_train_sintetico, prob, orientacion, balance, ruta, variante):
    info = enrutar_cliente(prob, orientacion, balance, df_train=df_train_sintetico)
    assert info['ruta'] == ruta
    assert info['variante'] == variante
