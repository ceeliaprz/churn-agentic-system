"""Configuración central del repositorio.

Sustituye a lo que en Colab hacían el montaje de Drive (`os.chdir(WORK_DIR)`), los
Secrets de Colab (`userdata.get('GROQ_API_KEY')`) y las constantes sueltas repartidas por
las celdas. Todas las rutas son relativas a la raíz del repo, así que funciona igual en
cualquier ordenador.

Los umbrales de negocio no se definen aquí: se leen de knowledge_base/business_rules.yaml,
que es la fuente única de verdad.
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]

DATA_DIR = RAIZ / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
# Resultados producidos por los notebooks: respaldan las cifras de la memoria. Solo lectura.
RESULTS_NOTEBOOK_DIR = RESULTS_DIR / "notebook"
# Ejecuciones nuevas desde scripts/ o el frontend. Nunca se escribe en RESULTS_NOTEBOOK_DIR.
RESULTS_EJECUCIONES_DIR = RESULTS_DIR / "ejecuciones"

MODELS_DIR = RAIZ / "models"
KB_DIR = RAIZ / "knowledge_base"
NOTEBOOKS_DIR = RAIZ / "notebooks"
ARTEFACTOS_REGENERADOS_DIR = RAIZ / "artefactos_regenerados"

RUTA_CHURN_CSV = RAW_DIR / "Churn_Modelling.csv"
RUTA_X_TRAIN_RAW = PROCESSED_DIR / "X_train_raw.csv"
RUTA_Y_TRAIN_RAW = PROCESSED_DIR / "y_train_raw.csv"
RUTA_X_TEST_RAW = PROCESSED_DIR / "X_test_raw.csv"
RUTA_Y_TEST_RAW = PROCESSED_DIR / "y_test_raw.csv"
RUTA_PREDICCIONES_ORIGINALES = PROCESSED_DIR / "predicciones_best_lgbm.csv"

RUTA_MODELO = MODELS_DIR / "best_model_lgbm.pkl"
RUTA_SCALER = MODELS_DIR / "scaler.pkl"
RUTA_FEATURE_ORDER = MODELS_DIR / "feature_order.pkl"

RUTA_REGLAS = KB_DIR / "business_rules.yaml"
RUTA_REFERENCE_STATS = KB_DIR / "reference_stats.json"

# ---------------------------------------------------------------------------
# Reglas de negocio (desde knowledge_base/business_rules.yaml)
# ---------------------------------------------------------------------------
with open(RUTA_REGLAS, encoding="utf-8") as _f:
    REGLAS: dict = yaml.safe_load(_f)

UMBRAL_PROB_ALTA: float = float(REGLAS["router"]["umbral_prob_alta"])
UMBRAL_PROB_BAJA: float = float(REGLAS["router"]["umbral_prob_baja"])
UMBRAL_VALOR_ALTO: float = float(REGLAS["router"]["umbral_valor_alto"])

LIFT_MUY_ENCIMA: float = float(REGLAS["lift"]["muy_por_encima"])
LIFT_ENCIMA: float = float(REGLAS["lift"]["por_encima"])
LIFT_DEBAJO: float = float(REGLAS["lift"]["por_debajo"])
LIFT_MUY_DEBAJO: float = float(REGLAS["lift"]["muy_por_debajo"])
N_MUESTRA_PEQUENA: int = int(REGLAS["lift"]["n_muestra_pequena"])

BALANCE_UMBRAL_ENCIMA: float = float(REGLAS["balance_senales"]["umbral_encima"])
BALANCE_UMBRAL_DEBAJO: float = float(REGLAS["balance_senales"]["umbral_debajo"])

MINIMO_N_ANCLAS: int = int(REGLAS["anclas_docstrings"]["minimo_n_general"])
MINIMO_N_ANCLAS_PRODUCTOS: int = int(REGLAS["anclas_docstrings"]["minimo_n_productos"])

# ---------------------------------------------------------------------------
# LLM (Groq vía LiteLLM). Valores del notebook, secciones 5.2, 6.3 y 7.4.
# ---------------------------------------------------------------------------
MODELO_LLM = "groq/openai/gpt-oss-20b"

LLM_PREDICTOR = {"model": MODELO_LLM, "temperature": 0.1}
LLM_ANALISTA = {"model": MODELO_LLM, "temperature": 0.0, "seed": 42, "reasoning_effort": "medium"}
LLM_RECOMENDADOR = {"model": MODELO_LLM, "temperature": 0.2, "seed": 42}

MAX_ITER_ANALISTA = 8
VERBOSE_AGENTES = True

# Asistente conversacional del frontend (backend/chat). Llamadas directas a Groq con
# function calling vía LiteLLM (sin CrewAI: su prompt es mucho más largo y el tier gratuito
# solo da 8.000 tokens/min). reasoning_effort bajo: su trabajo es elegir herramientas y resumir.
LLM_CHAT = {"model": MODELO_LLM, "temperature": 0.2, "reasoning_effort": "low"}
CHAT_MAX_PASOS = 6            # llamadas al LLM por mensaje del usuario (bucle de herramientas)
CHAT_MAX_MENSAJES = 24        # mensajes de historial que se reenvían (se recorta por turnos)
CHAT_MAX_RESULTADOS_BUSQUEDA = 10
# Estimaciones orientativas por agente y cliente, para avisar al usuario antes de ejecutarlo.
# Salen de la ejecución real de Lo con el repo (2.276 / 12.928 / 9.688, reintentos incluidos) y
# de las diferencias entre clientes consecutivos de la tabla de 8.1 (Predictor ≈2,2k; Analista
# 6–14k, 50k con muchos reintentos; Recomendador 7–15k). No usar los valores brutos de 8.1:
# estaban acumulados entre clientes (ver NOTAS_CONTINUACION.md).
TOKENS_ESTIMADOS_PREDICTOR = 2_500
TOKENS_ESTIMADOS_ANALISTA = 15_000
TOKENS_ESTIMADOS_RECOMENDADOR = 15_000

# ---------------------------------------------------------------------------
# Reintentos (tier gratuito de Groq: 8.000 tokens/min, 200.000/día)
# ---------------------------------------------------------------------------
MAX_INTENTOS = 6
ESPERA_DEFAULT_S = 65
ESPERA_MINIMA_RATE_LIMIT_S = 60.0
ESPERA_TOOL_CORRUPTA_S = 5
# Pipeline completo por cliente. En el notebook (8.1) figuraba 76.788, pero ese valor estaba
# acumulado entre clientes; restando consecutivos sale ≈34k, y la ejecución real de Lo dio 24.892.
TOKENS_MEDIOS_POR_CLIENTE = 34_000
LIMITE_TOKENS_DIA = 200_000
LIMITE_TOKENS_MINUTO = 8_000


def cargar_env() -> None:
    """Carga .env (si existe) en las variables de entorno. Sustituye a Colab Secrets."""
    try:
        from dotenv import load_dotenv
    except ImportError:  # python-dotenv es opcional si la variable ya está definida
        return
    load_dotenv(RAIZ / ".env")


def hay_api_key() -> bool:
    cargar_env()
    clave = os.environ.get("GROQ_API_KEY", "")
    return bool(clave) and clave != "pon_aqui_tu_clave_de_groq"
