# Sistema agéntico de churn bancario

TFM: *Diseño de un sistema de IA agéntica para la toma de decisiones empresariales basada en datos* (bank customer churn).

Tres agentes especializados (CrewAI + Groq, `gpt-oss-20b`) colaboran sobre un modelo LightGBM:

- **Predictor**: invoca el modelo de ML y devuelve la probabilidad de churn (el riesgo oficial).
- **Analista**: caracteriza al cliente frente al histórico con 4 tools (productos, actividad × tarjeta, percentil de saldo, contexto demográfico).
- **Recomendador**: propone una acción de retención explicable. Tiene dos variantes (`alto_valor` y `cartera_masiva`) y una tool de simulación contrafactual.

Un **router en Python** decide la ruta: cierre, riesgo alto o revisión. Para ello usa la probabilidad del Predictor y la agregación determinista de las señales del Analista (`balance_de_senales`), nunca el juicio del LLM.

Este repositorio es la versión organizada de `notebooks/ml_churn_model.ipynb` y `notebooks/notebook_pipeline_agentico.ipynb`. Reproduce exactamente sus resultados, y los tests lo comprueban. Añade además una interfaz de **chat** (Streamlit) en la que un agente **Asistente** decide, pregunta a pregunta, qué parte del sistema usar.

## Estructura

```
pyproject.toml, uv.lock  Dependencias (uv) y configuración de pytest; .python-version fija Python 3.12
config/settings.py       Rutas, modelos LLM, reintentos (umbrales: knowledge_base/business_rules.yaml)
data/raw/                Churn_Modelling.csv (Kaggle)
data/processed/          Split train/test y predicciones originales del notebook de ML
data/results/notebook/   Resultados del notebook (respaldan la memoria; solo lectura)
data/results/ejecuciones/  Ejecuciones nuevas desde scripts/ o el frontend (no se versionan)
models/                  best_model_lgbm.pkl, scaler.pkl, feature_order.pkl
knowledge_base/          Reglas de negocio, estadísticos de referencia, diccionario de variables
backend/
  ml/                    Preprocesado único, predicción, reentrenamiento
  analytics/             Base de referencia, tools del Analista (lógica pura), señales, simulación
  decision/              Router de 3 rutas
  agents/                CrewAI: LLM, esquemas, tools, Predictor, Analista, Recomendador
  pipeline/              Casos congelados, orquestador (3 modos), reintentos, persistencia
  evaluation/            Comparador tradicional, cobertura de riesgo, variabilidad
  chat/                  Agente Asistente (function calling): capa encima de los 3 agentes
frontend/                Chat en Streamlit (app.py + components/trazas.py)
scripts/                 Puntos de entrada por línea de comandos
tests/                   Tests de regresión contra el notebook
notebooks/               Notebooks originales (histórico, EDA, selección de modelo)
docs/                    pip freeze original de Colab
.env.example             Plantilla del fichero .env (clave de Groq)
```

## Instalación (Windows) con uv

El proyecto usa **[uv](https://docs.astral.sh/uv/)** como gestor de dependencias y de entornos (sustituye a pip + venv).
uv descarga **Python 3.12** por su cuenta (versión fijada en `.python-version`): no hace falta instalar Python aparte.

1. Instalar uv, una sola vez, en PowerShell:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   También sirve `winget install --id=astral-sh.uv -e`. Después, cierra y vuelve a abrir la terminal y comprueba con `uv --version`.
2. Crear el entorno e instalar todo, con las versiones exactas de `uv.lock`:
   ```powershell
   cd churn-agentic-system
   uv sync
   ```
   `uv sync` crea `.venv\`, descarga Python 3.12 si falta, instala las dependencias (incluido pytest) e instala el propio proyecto en modo editable (`backend` y `config`).

Para usar los modos con LLM (el chat y el pipeline con `--llm`) hace falta una clave de Groq propia: ver la sección siguiente. El modo determinista y los tests funcionan sin clave.

## Clave de la API de Groq (necesaria para el chat y los agentes)

Los agentes usan el modelo `openai/gpt-oss-20b` a través de [Groq](https://groq.com). La clave es personal y **no se incluye en el repositorio**: cada persona tiene que crear la suya. El plan gratuito es suficiente.

1. Crea una cuenta (o entra) en la consola de Groq: <https://console.groq.com>.
2. En el menú, ve a **API Keys** (<https://console.groq.com/keys>) y pulsa **Create API Key**. Ponle un nombre, por ejemplo `tfm-churn`.
3. Copia la clave, que empieza por `gsk_`. Groq solo la muestra una vez; si la pierdes, crea otra.
4. En la carpeta raíz del proyecto (la misma que `pyproject.toml`), crea el fichero `.env` a partir de la plantilla:
   ```powershell
   copy .env.example .env
   ```
   y sustituye el valor de ejemplo por tu clave:
   ```
   GROQ_API_KEY=gsk_tu_clave_aqui
   ```
5. Comprueba que funciona: al abrir el chat (`uv run streamlit run frontend/app.py`), la barra lateral debe mostrar «Groq configurado».

El fichero `.env` está en `.gitignore`: nunca se sube a git. No compartas la clave ni la pegues en el código.

**Límites del plan gratuito:** 8.000 tokens por minuto y 200.000 por día (los límites vigentes de tu cuenta están en <https://console.groq.com/settings/limits>). Un cliente completo (los tres agentes) gasta unos 25–35k tokens, así que caben unos 5–6 al día. Si se alcanza el límite por minuto, el sistema espera y reintenta solo. La barra lateral del chat muestra los tokens gastados hoy desde este ordenador; el total exacto de la cuenta se ve en la consola de Groq.

**PyCharm:** *Settings → Project → Python Interpreter → Add Interpreter → Add Local Interpreter → Select existing* y elige `churn-agentic-system\.venv\Scripts\python.exe`, que es el entorno que crea `uv sync`.

## Uso

`uv run` ejecuta cualquier comando dentro del entorno del proyecto, sin necesidad de activarlo, y antes comprueba que está sincronizado con `uv.lock`:

| Comando | Qué hace |
|---|---|
| `uv run pytest` | Tests de regresión, sin coste de Groq (≈30 s) |
| `uv run streamlit run frontend/app.py` | Chat en http://localhost:8501 |
| `uv run python scripts/run_pipeline.py --perfil desacuerdo_franco` | Un cliente en modo determinista (0 tokens) |
| `uv run python scripts/run_pipeline.py --perfil balance_alto --llm` | Pipeline completo con Groq (se reanuda si se corta) |
| `uv run python scripts/run_pipeline.py --customer-id 15749540 --solo-recomendador` | Señales en Python + Recomendador con Groq |
| `uv run python scripts/build_cases.py` | 6 casos congelados, reparto de rutas y baseline |
| `uv run python scripts/run_evaluation.py` | Cobertura de riesgo sobre los 2.000 clientes y su validación con `y_test` |
| `uv run python scripts/run_variability.py` | Análisis del experimento de variabilidad de tools |
| `uv run python scripts/train_model.py` | Reentrena y compara con los artefactos oficiales |
| `uv run python scripts/build_knowledge_base.py` | Regenera `knowledge_base/reference_stats.json` |

### Gestión de dependencias

| Quiero… | Comando |
|---|---|
| Añadir una librería | `uv add nombre` (actualiza `pyproject.toml` y `uv.lock`) |
| Añadir una librería solo de desarrollo | `uv add --dev nombre` |
| Quitar una librería | `uv remove nombre` |
| Reinstalar exactamente lo del lock | `uv sync` |
| Instalar sin las herramientas de desarrollo | `uv sync --no-dev` |
| Generar un requirements.txt (p. ej. para Colab) | `uv export --no-hashes --no-dev -o requirements.txt` |

No edites `uv.lock` a mano, y súbelo a git junto con `pyproject.toml` y `.python-version`.

## El chat

```powershell
uv run streamlit run frontend/app.py
```

Es una única pantalla de conversación, una **capa encima del pipeline del notebook**. El Asistente es un LLM (Groq, llamado directamente con LiteLLM) cuyas herramientas son **los tres agentes CrewAI**, una búsqueda de clientes y una vista de la cartera. En cada pregunta decide qué agentes activar:

| Herramienta | Qué activa | Para preguntas como… | Coste aprox. |
|---|---|---|---|
| `buscar_cliente` | Búsqueda en el test set | "¿Qué clientes se apellidan Gray?" | 0 |
| `consultar_predictor` | **Agente Predictor** | "¿Qué riesgo tiene Gray?" | ≈2,5k |
| `consultar_analista` | **Agente Analista** | "¿Por qué tiene ese riesgo?" | ≈15k, pide confirmación |
| `recomendar` | Predictor y Analista si faltan → **router (Python)** → **Agente Recomendador** | "¿Qué hacemos con Gray?" | hasta ≈32k, pide confirmación |
| `resumen_cartera` | Vista de los 2.000 clientes con el router aplicado (sin agentes) | "¿Qué clientes de Alemania están en revisión?" | 0 |

- Cada agente se ejecuta **una sola vez por cliente**. Sus resultados se guardan por etapas en `data/results/ejecuciones/resultados_pipeline.json`, el mismo archivo y formato que `scripts/run_pipeline.py --llm`.
- Además, cada mensaje consume unos 3–8k tokens del propio Asistente.
- Se necesita `GROQ_API_KEY`.
- Cada respuesta muestra *Herramientas usadas*, con la salida exacta de cada agente, y la recomendación aparece como tarjeta.

## Modos del pipeline (línea de comandos) y coste

| Modo | Qué usa LLM | Tokens aprox. |
|---|---|---|
| Determinista | Nada. Modelo, tools, router y simulación en Python | 0 |
| Solo Recomendador | Recomendador (las señales se calculan en Python) | ≈7k–15k |
| Pipeline completo | Predictor + Analista + Recomendador | ≈25–35k (ejecución real de Lo: 24.892). El notebook reportaba ≈77k, pero era el contador acumulado de CrewAI entre clientes |
| Chat | Asistente (3–8k por pregunta) + los agentes que active | variable |

Límites del tier gratuito de Groq: 8.000 tokens por minuto y 200.000 por día, lo que da unos 5–6 clientes completos al día.

## Qué validan los tests

| Test | Cifra del notebook |
|---|---|
| `test_predict_churn_30_de_30`, `test_predicciones_2000_de_2000` | Predicciones idénticas a `predicciones_best_lgbm.csv` |
| `test_reentrenar_reproduce_artefactos` | Mismo split, scaler y probabilidades; ROC-AUC 0,8655, PR-AUC 0,7163 |
| `test_casos_congelados_identicos` | `casos_recomendador.json` idéntico |
| `test_reparto_6_casos` | 2 cierre, 2 riesgo alto, 2 revisión, con sus variantes |
| `test_baseline_tradicional_identico` | `baseline_tradicional.json` idéntico |
| `test_baseline_tradicional_no_usa_analista` | El comparador decide solo con la probabilidad (Gray se cierra sin acción) |
| `test_validacion_rescatados_con_y_test` | 81 / 10 / 12,3 % frente a 972 / 42 / 4,3 %, p = 0,0045, y la tabla idéntica |
| `test_cobertura_riesgo` | 2.000 / 1.053 / 81 (4,05 %) y el CSV idéntico |
| `test_equivalencia_celdas.py`, `test_prompts_notebook.py` | Ejecutan las celdas originales del notebook y comparan tools, docstrings, prompts, backstories y esquemas |
| `test_orchestrator_sin_groq.py` | Encadenado de etapas, guardado y reanudación con un LLM falso |
| `test_chat.py` | Herramientas del chat (agentes falsos con los datos de los casos congelados), guardado por etapas, confirmación, reintentos y recorte con un LLM falso |
| `test_chat_litellm.py` | Protocolo real LiteLLM → Groq contra un servidor falso local (sin red ni cuota) |
| `test_frontend.py` | El chat: aviso sin clave, ejemplos, pregunta con herramienta, confirmación del Recomendador |
