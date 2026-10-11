# Sistema agéntico de churn bancario

TFM: *Diseño de un sistema de IA agéntica para la toma de decisiones empresariales basada en datos* (bank customer churn).

Tres agentes (CrewAI + Groq, `gpt-oss-20b`) trabajan sobre un modelo LightGBM:

- **Predictor**: invoca el modelo de ML y devuelve la probabilidad de churn (el riesgo oficial).
- **Analista**: compara al cliente con el histórico mediante 4 tools (productos, actividad × tarjeta, percentil de saldo, contexto demográfico).
- **Recomendador**: propone una acción de retención explicable. Tiene dos variantes (`alto_valor` y `cartera_masiva`) y una tool de simulación contrafactual.

Un **router en Python** decide la ruta (cierre, riesgo alto o revisión) a partir de la probabilidad del Predictor y de la agregación determinista de las señales del Analista (`balance_de_senales`), nunca del juicio del LLM.

El repositorio organiza el código de `notebooks/ml_churn_model.ipynb` y `notebooks/notebook_pipeline_agentico.ipynb` y reproduce sus resultados (`scripts/build_cases.py` y `scripts/train_model.py` los comparan con los del notebook al ejecutarse). Encima añade un **chat** (Streamlit) con un agente **Asistente** que decide, en cada pregunta, qué agentes activar.

## Estructura

```
config/settings.py         Rutas, modelo LLM y reintentos (umbrales en knowledge_base/business_rules.yaml)
data/raw/                  Churn_Modelling.csv (Kaggle)
data/processed/            Split train/test y predicciones del notebook de ML
data/results/notebook/     Resultados del notebook (respaldan la memoria; solo lectura)
data/results/ejecuciones/  Ejecuciones nuevas desde scripts/ o el chat (no se versionan)
models/                    best_model_lgbm.pkl, scaler.pkl, feature_order.pkl
knowledge_base/            Reglas de negocio, estadísticos de referencia, diccionario de variables
backend/
  ml/                      Preprocesado, predicción, reentrenamiento
  analytics/               Base de referencia, lógica de las tools del Analista, señales, simulación
  decision/                Router de 3 rutas y vista de cartera
  agents/                  CrewAI: LLM, esquemas, tools, Predictor, Analista, Recomendador
  pipeline/                Casos congelados, orquestador, reintentos, persistencia, consumo de tokens
  evaluation/              Comparador de reglas fijas, cobertura de riesgo, variabilidad
  chat/                    Agente Asistente (function calling) sobre los 3 agentes
frontend/                  Chat en Streamlit (app.py + components/trazas.py)
scripts/                   Puntos de entrada por línea de comandos
notebooks/                 Notebooks originales (EDA, selección de modelo, experimentos)
docs/                      pip freeze original de Colab
```

## Instalación (Windows)

El proyecto usa **[uv](https://docs.astral.sh/uv/)**, que también descarga Python 3.12 (fijado en `.python-version`), así que no hace falta instalar Python aparte.

1. Instalar uv en PowerShell (una sola vez) y reabrir la terminal:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
   También sirve `winget install --id=astral-sh.uv -e`. Comprobar con `uv --version`.
2. Crear el entorno con las versiones exactas de `uv.lock`:
   ```powershell
   cd churn-agentic-system
   uv sync
   ```
   Crea `.venv\` e instala las dependencias y el propio proyecto (`backend` y `config`). En PyCharm, el intérprete es `.venv\Scripts\python.exe`.

## Clave de la API de Groq

El chat y los agentes necesitan una clave de Groq propia (el plan gratuito es suficiente). El modo determinista funciona sin clave.

1. En <https://console.groq.com/keys>, pulsa **Create API Key** y copia la clave (empieza por `gsk_`; Groq solo la muestra una vez).
2. En la raíz del proyecto, crea `.env` a partir de la plantilla y pon tu clave:
   ```powershell
   copy .env.example .env
   ```
   ```
   GROQ_API_KEY=gsk_tu_clave_aqui
   ```
3. Al abrir el chat, la barra lateral debe mostrar «Groq configurado».

`.env` está en `.gitignore` y nunca se sube a git.

**Límites del plan gratuito:** 8.000 tokens por minuto y 200.000 por día (los de tu cuenta, en <https://console.groq.com/settings/limits>). Si se alcanza el límite por minuto, el sistema espera y reintenta solo. La barra lateral muestra los tokens gastados hoy desde este ordenador.

## Uso

| Comando | Qué hace |
|---|---|
| `uv run streamlit run frontend/app.py` | Chat en http://localhost:8501 |
| `uv run python scripts/run_pipeline.py --perfil desacuerdo_franco` | Un cliente en modo determinista (0 tokens) |
| `uv run python scripts/run_pipeline.py --perfil balance_alto --llm` | Pipeline completo con Groq (se reanuda si se corta) |
| `uv run python scripts/run_pipeline.py --customer-id 15749540 --solo-recomendador` | Señales en Python + Recomendador con Groq |
| `uv run python scripts/build_cases.py` | 6 casos congelados, reparto de rutas y comparador de reglas fijas |
| `uv run python scripts/run_evaluation.py` | Cobertura de riesgo sobre los 2.000 clientes y su validación con `y_test` |
| `uv run python scripts/run_variability.py` | Análisis del experimento de variabilidad de tools |
| `uv run python scripts/train_model.py` | Reentrena y compara con los artefactos oficiales |
| `uv run python scripts/build_knowledge_base.py` | Regenera `knowledge_base/reference_stats.json` |

Para añadir o quitar librerías: `uv add nombre` / `uv remove nombre` (actualizan `pyproject.toml` y `uv.lock`). No edites `uv.lock` a mano.

## El chat

Una única pantalla de conversación encima del pipeline. El Asistente es un LLM (Groq, llamado directamente con LiteLLM) cuyas herramientas son los tres agentes, una búsqueda de clientes y una vista de la cartera:

| Herramienta | Qué activa | Ejemplo de pregunta | Coste aprox. |
|---|---|---|---|
| `buscar_cliente` | Búsqueda en el conjunto de test | "¿Qué clientes se apellidan Gray?" | 0 |
| `consultar_predictor` | Agente Predictor | "¿Qué riesgo tiene Gray?" | ≈2,5k |
| `consultar_analista` | Agente Analista (puede repetir el análisis indicando qué faltó) | "¿Por qué tiene ese riesgo?" | ≈15k, pide confirmación |
| `recomendar` | Predictor y Analista si faltan → router → Agente Recomendador | "¿Qué hacemos con Gray?" | hasta ≈32k, pide confirmación |
| `resumen_cartera` | Rutas de los 2.000 clientes de test, sin agentes | "¿Qué clientes de Alemania están en revisión?" | 0 |

- Los resultados de cada agente se guardan en `data/results/ejecuciones/resultados_pipeline.json` (el mismo archivo que `run_pipeline.py --llm`) y se reutilizan: un agente no se vuelve a ejecutar para el mismo cliente salvo que se pida un reanálisis.
- Cada mensaje consume además unos 3–8k tokens del propio Asistente.
- Cada respuesta muestra *Herramientas usadas* con la salida de cada agente, y la recomendación aparece como tarjeta.

## Modos y coste

| Modo | Qué usa LLM | Tokens aprox. |
|---|---|---|
| Determinista | Nada (modelo, tools, router y simulación en Python) | 0 |
| Solo Recomendador | Recomendador (señales calculadas en Python) | ≈7–15k |
| Pipeline completo | Predictor + Analista + Recomendador | ≈34k de media por cliente (unos 5–6 clientes al día en el plan gratuito) |
| Chat | Asistente (3–8k por pregunta) + los agentes que active | variable |
