# Diccionario de variables — Churn Modelling (Kaggle)

Origen: celda 7 de `notebooks/ml_churn_model.ipynb`. 10.000 clientes de un banco europeo.

## Identificadores (no se usan para predecir)

| Variable | Descripción |
|---|---|
| RowNumber | Número de fila del CSV original |
| CustomerId | Identificador del cliente |
| Surname | Apellido (se usa como nombre del cliente en los agentes) |

## Demográficas (no intervenibles por el banco)

| Variable | Descripción |
|---|---|
| Geography | País: France, Germany, Spain. Francia es el principal (≈5.000 clientes) |
| Gender | Male / Female |
| Age | Edad (18–92, media ≈39) |
| EstimatedSalary | Salario anual estimado (≈11,58 € a 199.992,48 €) |

## Relación con el banco

| Variable | Descripción | ¿Palanca intervenible? |
|---|---|---|
| CreditScore | Puntuación crediticia (350–850, media ≈650) | No |
| Tenure | Antigüedad en años (0–10) | No |
| Balance | Saldo en cuenta. ≥25 % de clientes con saldo 0 | No |
| NumOfProducts | Productos contratados (1–4). Relación no monótona: 2 es el punto más estable; 3–4 casi siempre abandonan (muestras pequeñas) | **Sí** |
| HasCrCard | Tiene tarjeta de crédito (1/0). ≈70,5 % sí | **Sí** |
| IsActiveMember | Miembro activo comercialmente (1/0). ≈51,5 % activos | **Sí** |

## Objetivo

| Variable | Descripción |
|---|---|
| Exited | 1 = abandona (churn), 0 = permanece. Desbalanceo ≈80/20 |

## Variables derivadas (creadas en el preprocesado)

| Variable | Definición | Motivo (EDA) |
|---|---|---|
| HasZeroBalance | `Balance == 0` | Los clientes con saldo 0 tienen casi la mitad de churn |
| ProductRisk | 1–2 productos → 0 (Low); 3–4 → 1 (High) | Agrupa los niveles de riesgo con pocas muestras |
| Geography_Germany, Geography_Spain | One-hot de Geography (France es la referencia) | Encoding sin orden |

Gender se codifica Male = 0, Female = 1. Las numéricas (CreditScore, Age, Tenure, Balance,
EstimatedSalary) se escalan con el `StandardScaler` ajustado solo en train.
