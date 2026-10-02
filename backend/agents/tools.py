"""
Nombres, parámetros y docstrings idénticos al notebook: son lo que ve el LLM para decidir
qué tool usar y con qué argumentos (Groq no acepta `dict` genéricos: parámetros escalares).

Las tools se construyen con factorías que reciben sus dependencias (modelo, base de
referencia, registro) en lugar de leer variables globales.
"""
from __future__ import annotations

from crewai.tools import tool

from backend.analytics import segments
from backend.analytics.counterfactual import simular_palancas, texto_simulacion
from backend.analytics.reference import BaseReferencia
from backend.agents.tool_registry import RegistroLlamadas
from backend.ml.predictor import ModeloChurn

# ---------------------------------------------------------------------------
# Docstrings literales
# ---------------------------------------------------------------------------
DOC_PREDICTOR = """
    ÚTIL ÚNICAMENTE PARA: calcular la probabilidad de abandono (churn) de un cliente
    bancario, usando un modelo de Machine Learning ya entrenado (LightGBM).

    Debes llamar SIEMPRE a esta tool cuando necesites predecir el riesgo de churn
    de un cliente. No inventes ni estimes el resultado por tu cuenta bajo ningún concepto.

    Recibe los datos crudos del cliente, usando EXACTAMENTE las
    mismas claves que se te han dado en la descripción de la tarea (CreditScore,
    Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
    IsActiveMember, EstimatedSalary), sin traducirlas ni cambiarles el nombre.
    Devuelve la predicción (0=no churn, 1=churn) junto con la probabilidad estimada.
    """

DOC_SIMULACION = """
    ÚTIL ÚNICAMENTE PARA: estimar qué palanca de retención movería más la probabilidad de
    churn de ESTE cliente concreto. Úsala cuando el riesgo sea alto o cuando el Predictor y
    el Analista no coincidan: en ambos casos conviene saber sobre qué variable actuar antes
    de proponer nada. No la uses con un cliente de riesgo bajo.
    Re-calcula la probabilidad del modelo cambiando, una a una, las variables sobre las que
    el banco puede intervenir realmente (activar al cliente, ofrecerle tarjeta, o llevarlo
    al número de productos más estable). Devuelve SOLO el cambio en puntos porcentuales de
    cada intervención frente a la situación actual del cliente, de mayor a menor impacto —
    nunca la probabilidad absoluta, que es competencia exclusiva del Predictor.
    """


def texto_prediccion(resultado: dict) -> str:
    """Texto que devuelve la tool del Predictor. El % se calcula aquí fuera para evitar
    errores del LLM."""
    probabilidad_pct = resultado['churn_probability'] * 100
    return (
        f"Predicción: {resultado['churn_prediction']} "
        f"(0=no abandona, 1=abandona) | "
        f"Probabilidad de churn: {probabilidad_pct:.2f}%"
    )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------
def crear_tool_predictor(modelo: ModeloChurn):
    def predecir_churn_tool(CreditScore: int, Geography: str, Gender: str, Age: int,
                            Tenure: int, Balance: float, NumOfProducts: int, HasCrCard: int,
                            IsActiveMember: int, EstimatedSalary: float) -> str:
        customer_data = {
            'CreditScore': CreditScore,
            'Geography': Geography,
            'Gender': Gender,
            'Age': Age,
            'Tenure': Tenure,
            'Balance': Balance,
            'NumOfProducts': NumOfProducts,
            'HasCrCard': HasCrCard,
            'IsActiveMember': IsActiveMember,
            'EstimatedSalary': EstimatedSalary,
        }
        return texto_prediccion(modelo.predict_churn(customer_data))

    predecir_churn_tool.__doc__ = DOC_PREDICTOR
    return tool("Predecir Churn de Cliente")(predecir_churn_tool)


# ---------------------------------------------------------------------------
# Analista
# ---------------------------------------------------------------------------
def crear_tools_analista(base: BaseReferencia, registro: RegistroLlamadas) -> list:
    """Las 4 tools del Analista ligadas a una base de referencia y a un registro de llamadas.
    Orden de la lista igual que en el notebook (celda 32)."""

    def consultar_actividad_vinculacion(is_active_member: int, has_crcard: int) -> str:
        return segments.consultar_actividad_vinculacion(base, is_active_member, has_crcard)

    def calcular_percentil_balance(balance: float) -> str:
        return segments.calcular_percentil_balance(base, balance)

    def _consultar_riesgo_productos(num_productos: int) -> str:
        return segments.consultar_riesgo_productos(base, num_productos)

    def _consultar_contexto_demografico(geography: str, age: int) -> str:
        return segments.consultar_contexto_demografico(base, geography, age)

    consultar_actividad_vinculacion.__doc__ = segments.DOC_ACTIVIDAD
    calcular_percentil_balance.__doc__ = segments.DOC_BALANCE
    # Los docstrings se asignan antes de decorar porque se construyen con f-strings a partir
    # de df_train, y un f-string no puede ser docstring literal.
    _consultar_riesgo_productos.__doc__ = base.doc_productos
    _consultar_contexto_demografico.__doc__ = base.doc_contexto

    tool_actividad = tool("Actividad")(
        registro.registrar(segments.NOMBRE_REGISTRO_ACTIVIDAD)(consultar_actividad_vinculacion))
    tool_balance = tool("Balance")(
        registro.registrar(segments.NOMBRE_REGISTRO_BALANCE)(calcular_percentil_balance))
    tool_productos = tool("Productos")(
        registro.registrar(segments.NOMBRE_REGISTRO_PRODUCTOS)(_consultar_riesgo_productos))
    tool_contexto = tool("Contexto")(
        registro.registrar(segments.NOMBRE_REGISTRO_CONTEXTO)(_consultar_contexto_demografico))

    return [tool_balance, tool_productos, tool_actividad, tool_contexto]


# ---------------------------------------------------------------------------
# Recomendador
# ---------------------------------------------------------------------------
def crear_tool_simulacion(modelo: ModeloChurn, prod_estable: int):
    def simular_reanalisis_tool(
        CreditScore: int, Geography: str, Gender: str, Age: int, Tenure: int,
        Balance: float, NumOfProducts: int, HasCrCard: int,
        IsActiveMember: int, EstimatedSalary: float
    ) -> str:
        base_cliente = {
            'CreditScore': CreditScore, 'Geography': Geography, 'Gender': Gender,
            'Age': Age, 'Tenure': Tenure, 'Balance': Balance,
            'NumOfProducts': NumOfProducts, 'HasCrCard': HasCrCard,
            'IsActiveMember': IsActiveMember, 'EstimatedSalary': EstimatedSalary,
        }
        return texto_simulacion(simular_palancas(modelo, prod_estable, base_cliente))

    simular_reanalisis_tool.__doc__ = DOC_SIMULACION
    return tool("Simulación reanalisis")(simular_reanalisis_tool)
