"""Lógica pura de las 4 tools del Analista (notebook agéntico, celda 30).

Cada función recibe la `BaseReferencia` de forma explícita en lugar de leer las globales
`df_train` y `BASE_CHURN`. Los textos que devuelven son idénticos carácter a carácter a los
del notebook: son lo que lee el LLM y lo que parsea `balance_de_senales`.

Los envoltorios `@tool` de CrewAI están en backend/agents/tools.py.
"""
from __future__ import annotations

from config import settings
from backend.analytics.reference import BaseReferencia


# --- Formato común de salida de las tools ---
def _formato_segmento(descripcion: str, n: int, tasa: float, base_churn: float) -> str:
    """Devuelve siempre el segmento comparado con la base global.
    Un porcentaje solo no es interpretable sin saber cuál es la media;
    el cociente frente a la base sí lo es."""
    if n == 0:
        return f"{descripcion}: no hay clientes con este perfil en la base histórica."
    lift = tasa / base_churn
    if lift >= settings.LIFT_MUY_ENCIMA:
        lectura = "muy por encima de la media"
    elif lift >= settings.LIFT_ENCIMA:
        lectura = "por encima de la media"
    elif lift <= settings.LIFT_MUY_DEBAJO:
        lectura = "muy por debajo de la media"
    elif lift <= settings.LIFT_DEBAJO:
        lectura = "por debajo de la media"
    else:
        lectura = "en línea con la media"
    aviso = " [muestra pequeña: interpretar con cautela]" if n < settings.N_MUESTRA_PEQUENA else ""
    return (f"{descripcion} (n={n}): churn {tasa:.1f}% vs {base_churn:.1f}% base "
            f"→ x{lift:.2f}, {lectura}{aviso}")


# --- Tools del Analista Palanca ---
def consultar_riesgo_productos(base: BaseReferencia, num_productos: int) -> str:
    df_train = base.df_train
    try:
        seg = df_train[df_train['NumOfProducts'] == num_productos]
        return _formato_segmento(f"Clientes con {num_productos} producto(s)",
                                 len(seg), seg['Exited'].mean() * 100, base.base_churn)
    except Exception as e:
        return f"Error al consultar el número de productos: {e}"


def consultar_actividad_vinculacion(base: BaseReferencia, is_active_member: int, has_crcard: int) -> str:
    df_train = base.df_train
    try:
        seg = df_train[(df_train['IsActiveMember'] == is_active_member) &
                       (df_train['HasCrCard'] == has_crcard)]
        estado = "activos" if is_active_member == 1 else "inactivos"
        tarjeta = "con tarjeta" if has_crcard == 1 else "sin tarjeta"
        return _formato_segmento(f"Clientes {estado} y {tarjeta}",
                                 len(seg), seg['Exited'].mean() * 100, base.base_churn)
    except Exception as e:
        return f"Error al evaluar la vinculación: {e}"


# --- Tools del Analista Contexto ---
def calcular_percentil_balance(base: BaseReferencia, balance: float) -> str:
    df_train = base.df_train
    try:
        if balance == 0:
            seg = df_train[df_train['Balance'] == 0]
            return _formato_segmento("Clientes con saldo 0 €", len(seg),
                                     seg['Exited'].mean() * 100, base.base_churn)

        percentil = (df_train['Balance'] < balance).mean() * 100
        q1, q2, q3 = base.q_balance_pos
        if balance <= q1:
            seg = df_train[(df_train['Balance'] > 0) & (df_train['Balance'] <= q1)]
            tramo = "primer cuartil de saldo positivo"
        elif balance <= q2:
            seg = df_train[(df_train['Balance'] > q1) & (df_train['Balance'] <= q2)]
            tramo = "segundo cuartil de saldo positivo"
        elif balance <= q3:
            seg = df_train[(df_train['Balance'] > q2) & (df_train['Balance'] <= q3)]
            tramo = "tercer cuartil de saldo positivo"
        else:
            seg = df_train[df_train['Balance'] > q3]
            tramo = "cuarto cuartil de saldo positivo"

        return (f"Saldo de {balance:,.2f} € → percentil {percentil:.1f} de la base. "
                + _formato_segmento(f"Clientes del {tramo}", len(seg),
                                    seg['Exited'].mean() * 100, base.base_churn))
    except Exception as e:
        return f"Error al calcular el percentil de balance: {e}"


def consultar_contexto_demografico(base: BaseReferencia, geography: str, age: int) -> str:
    df_train = base.df_train
    try:
        seg_geo = df_train[df_train['Geography'] == geography]
        inf = (int(age) // 10) * 10
        sup = inf + 9
        seg_edad = df_train[(df_train['Age'] >= inf) & (df_train['Age'] <= sup)]

        linea_pais = _formato_segmento(f"Clientes de {geography}", len(seg_geo),
                                       seg_geo['Exited'].mean() * 100, base.base_churn)
        linea_edad = _formato_segmento(f"Clientes de {inf}-{sup} años", len(seg_edad),
                                       seg_edad['Exited'].mean() * 100, base.base_churn)
        return f"[País] {linea_pais}\n[Edad] {linea_edad}"
    except Exception as e:
        return f"Error al consultar el contexto demográfico: {e}"


# ---------------------------------------------------------------------------
# Docstrings literales de las tools cuya descripción no depende de df_train.
# (Las de "Productos" y "Contexto" se generan en BaseReferencia.doc_productos/doc_contexto.)
# ---------------------------------------------------------------------------
DOC_ACTIVIDAD = """Devuelve la tasa histórica de abandono del segmento formado por el estado de actividad
    (IsActiveMember) y si tiene tarjeta (HasCrCard), comparada con la media global. La
    actividad separa la base en dos grupos con tasas muy distintas; en clientes activos el
    resultado queda cerca de la media global."""

DOC_BALANCE = """Sitúa el saldo del cliente en la distribución real de la base y devuelve la tasa histórica
    de abandono de su tramo de saldo, comparada con la media global. Una cifra en euros no es
    interpretable sin saber qué posición ocupa; los extremos (saldo nulo o muy alto) son los que
    más se separan de la media."""

# Nombres con los que el registro de llamadas identifica cada tool (celda 30)
NOMBRE_REGISTRO_PRODUCTOS = "Riesgo por número de productos"
NOMBRE_REGISTRO_ACTIVIDAD = "Score de actividad y abandono por segmento"
NOMBRE_REGISTRO_BALANCE = "Percentil de balance"
NOMBRE_REGISTRO_CONTEXTO = "Contexto demográfico"


def todas_las_senales(base: BaseReferencia, customer_data: dict) -> list[tuple[str, str]]:
    """Invoca las 4 tools en Python, sin LLM, en el mismo orden que `construir_caso`
    (celda 48). Devuelve pares (nombre_registro, resultado)."""
    return [
        (NOMBRE_REGISTRO_PRODUCTOS,
         consultar_riesgo_productos(base, int(customer_data['NumOfProducts']))),
        (NOMBRE_REGISTRO_ACTIVIDAD,
         consultar_actividad_vinculacion(base, int(customer_data['IsActiveMember']),
                                         int(customer_data['HasCrCard']))),
        (NOMBRE_REGISTRO_BALANCE,
         calcular_percentil_balance(base, float(customer_data['Balance']))),
        (NOMBRE_REGISTRO_CONTEXTO,
         consultar_contexto_demografico(base, customer_data['Geography'],
                                        int(customer_data['Age']))),
    ]
