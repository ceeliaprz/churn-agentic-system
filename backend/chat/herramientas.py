"""Herramientas del Asistente: buscar un cliente y activar los agentes del notebook.

El Asistente no calcula nada por su cuenta: sus herramientas son los tres agentes CrewAI del
TFM (Predictor, Analista, Recomendador) y el router de Python, exactamente como en el
pipeline de la sección 8.1. Lo nuevo es que el Asistente decide, según la pregunta, qué
agentes activar:

    buscar_cliente       apellido / CustomerId / perfil → cliente del test set   (0 tokens)
    consultar_predictor  Agente Predictor → probabilidad de churn (riesgo oficial) (≈2,5k)
    consultar_analista   Agente Analista → señales, conclusión, orientación       (≈15k, confirma)
                         repetir=true + motivo → reanálisis con una indicación de qué faltó
    recomendar           lo que falte de Predictor/Analista → router → Recomendador (confirma)
    resumen_cartera      vista determinista de los 2.000 clientes: filtrar y priorizar  (0 tokens)

Cada agente se ejecuta una sola vez por cliente: los resultados se guardan por etapas en
data/results/ejecuciones/resultados_pipeline.json (el mismo archivo y formato que usa
`scripts/run_pipeline.py --llm`), así que ni el chat ni la línea de comandos pagan dos veces
por lo mismo.
"""
from __future__ import annotations

import json
from typing import Callable, Optional

import pandas as pd

from config import settings
from backend.decision.router import enrutar_cliente
from backend.pipeline.cases import datos_cliente, seleccionar_perfiles_test
from backend.pipeline.orchestrator import ContextoSistema
from backend.pipeline.storage import AlmacenResultados, ruta_ejecuciones

Evento = Optional[Callable[[str], None]]

NOMBRES_RUTA = {1: "Ruta 1 — cierre sin Recomendador", 2: "Ruta 2 — riesgo alto",
                3: "Ruta 3 — revisión"}
ORIENTACION_CORTA = {'riesgo_por_encima_de_la_media': 'por encima',
                     'riesgo_en_linea_con_la_media': 'en línea',
                     'riesgo_por_debajo_de_la_media': 'por debajo'}
ORIGEN_CORTO_CALCULADO = 'cálculo 4 herramientas'


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Ejecución de los agentes (puntos únicos que los tests sustituyen por falsos)
# ---------------------------------------------------------------------------
def correr_predictor(nombre, datos, ctx, on_evento):
    from backend.pipeline.orchestrator import ejecutar_predictor
    from backend.pipeline.retries import ejecutar_sync
    return ejecutar_sync(ejecutar_predictor(nombre, datos, ctx, on_evento))


def correr_analista(nombre, datos, ctx, on_evento, motivo=None):
    from backend.pipeline.orchestrator import ejecutar_analista
    from backend.pipeline.retries import ejecutar_sync
    return ejecutar_sync(ejecutar_analista(nombre, datos, ctx, on_evento,
                                           motivo_reanalisis=motivo))


def correr_recomendador(nombre, datos, prediccion, senales, ruta_info, ctx, conclusion, on_evento):
    from backend.pipeline.orchestrator import ejecutar_recomendador
    from backend.pipeline.retries import ejecutar_sync
    return ejecutar_sync(ejecutar_recomendador(nombre, datos, prediccion, senales, ruta_info, ctx,
                                               conclusion_analista=conclusion, on_evento=on_evento))


# ---------------------------------------------------------------------------
# Esquemas de function calling (parámetros escalares: Groq no acepta dict genéricos)
# ---------------------------------------------------------------------------
def _funcion(nombre: str, descripcion: str, propiedades: dict, requeridos: list) -> dict:
    return {"type": "function", "function": {
        "name": nombre, "description": descripcion,
        "parameters": {"type": "object", "properties": propiedades, "required": requeridos}}}


_ID = {"type": "integer", "description": "CustomerId del cliente (obtenlo con buscar_cliente)"}

ESQUEMAS = [
    _funcion(
        "buscar_cliente",
        "Busca clientes del conjunto de test (2.000 clientes) por apellido (coincidencia parcial), "
        "por CustomerId o por nombre de perfil congelado (balance_cero, balance_alto, multi_producto, "
        "inactivo_alemania, perfil_limpio, desacuerdo_franco). Devuelve customer_id y datos básicos. "
        "Úsala antes de cualquier agente si solo conoces el apellido. No consume tokens.",
        {"consulta": {"type": "string", "description": "Apellido, CustomerId o nombre de perfil"}},
        ["consulta"]),
    _funcion(
        "consultar_predictor",
        "Activa el Agente Predictor: invoca el modelo LightGBM y devuelve la predicción y la "
        "probabilidad de churn, que es el riesgo oficial. Úsalo cuando pregunten por el riesgo o la "
        "probabilidad de abandono.",
        {"customer_id": _ID}, ["customer_id"]),
    _funcion(
        "consultar_analista",
        "Activa el Agente Analista: caracteriza al cliente frente al histórico con sus herramientas "
        "(productos, actividad, saldo, contexto demográfico) y devuelve señales con multiplicador, "
        "conclusión y orientación. Úsalo cuando pregunten POR QUÉ o qué perfil tiene el cliente. "
        "Con repetir=true vuelve a ejecutarlo aunque ya haya un análisis guardado. Si repites porque "
        "falta información, indica en 'motivo' qué no se consultó y por qué puede importar para este "
        "cliente: el Analista lo recibe como indicación y sigue decidiendo qué herramientas usa. "
        "Consume unos 15.000 tokens.",
        {"customer_id": _ID,
         "repetir": {"type": "boolean", "description": "Rehacer el análisis aunque exista uno guardado"},
         "motivo": {"type": "string",
                    "description": "Solo al repetir: qué faltó en el análisis anterior y por qué puede "
                                   "importar (p. ej. herramientas no consultadas que encajan con el "
                                   "perfil del cliente). Se le pasa al Analista."}},
        ["customer_id"]),
    _funcion(
        "recomendar",
        "Obtiene una acción de retención: ejecuta el Predictor y el Analista si todavía no se han "
        "ejecutado para este cliente, aplica el router de 3 rutas y, si el router no cierra el caso, "
        "activa al Agente Recomendador con la variante que corresponda. No hace falta llamar antes a "
        "los otros agentes. La interfaz pedirá confirmación del coste.",
        {"customer_id": _ID}, ["customer_id"]),
    _funcion(
        "resumen_cartera",
        "Vista de toda la cartera (los 2.000 clientes de test) sin activar ningún agente: reparto por "
        "rutas y lista de clientes ordenada de mayor a menor probabilidad de churn, con filtros "
        "opcionales por país, ruta y variante. Úsala para preguntas sobre muchos clientes, conteos o "
        "priorización (p. ej. 'qué clientes de Alemania están en revisión', 'a quién contacto "
        "primero'). La probabilidad es la del modelo y la ruta la del router; la orientación es la "
        "calculada con las 4 herramientas del Analista, salvo en los clientes que el Agente Analista ya "
        "ha analizado, donde se usa su análisis (campo 'origen'). No consume tokens de agentes.",
        {"pais": {"type": "string", "description": "France, Germany o Spain (opcional)"},
         "ruta": {"type": "integer", "description": "1 (cierre), 2 (riesgo alto) o 3 (revisión) (opcional)"},
         "variante": {"type": "string", "description": "alto_valor o cartera_masiva (opcional)"},
         "limite": {"type": "integer",
                    "description": f"Solo si el usuario pide un número concreto de clientes; si no, no lo "
                                   f"pases (por defecto {settings.CHAT_CARTERA_LIMITE_DEFECTO}, "
                                   f"máximo {settings.CHAT_CARTERA_LIMITE_MAX})"}},
        []),
]

HERRAMIENTAS_CON_CONFIRMACION = {"consultar_analista", "recomendar"}


# ---------------------------------------------------------------------------
class HerramientasChat:
    def __init__(self, ctx: Optional[ContextoSistema] = None, x_test: Optional[pd.DataFrame] = None,
                 almacen: Optional[AlmacenResultados] = None):
        from backend.analytics.reference import cargar_x_test
        self.ctx = ctx or ContextoSistema.por_defecto()
        self.x_test = x_test if x_test is not None else cargar_x_test()
        self._perfiles = {p: int(f.name) for p, f in seleccionar_perfiles_test(self.x_test).items()}
        self.almacen = almacen or AlmacenResultados(ruta_ejecuciones("resultados_pipeline.json"))
        self._cobertura = None   # cobertura de los 2.000 clientes; se carga la primera vez que se usa

    # --- estado por cliente ------------------------------------------------------
    def _fila(self, customer_id) -> Optional[pd.Series]:
        filas = self.x_test[self.x_test['CustomerId'] == int(customer_id)]
        return None if filas.empty else filas.iloc[0]

    @staticmethod
    def _clave(customer_id) -> str:
        return f"customer_{int(customer_id)}"

    def _registro(self, fila: pd.Series) -> dict:
        """Registro del cliente con el formato del pipeline (8.1); vacío si aún no hay nada."""
        previo = self.almacen.get(self._clave(fila['CustomerId']))
        return dict(previo) if previo else {'cliente': fila['Surname'], 'datos': datos_cliente(fila)}

    def _guardar(self, fila: pd.Series, registro: dict) -> None:
        if registro.get('prediccion') and registro.get('orientacion_calculada'):
            registro['ruta_info'] = enrutar_cliente(
                churn_probability_pct=registro['prediccion']['churn_probability_pct'],
                orientacion_calculada=registro['orientacion_calculada'],
                balance=registro['datos']['Balance'], df_train=self.ctx.base.df_train)
        self.almacen.guardar(self._clave(fila['CustomerId']), registro)

    @staticmethod
    def _router(registro: dict) -> Optional[dict]:
        ri = registro.get('ruta_info')
        if not ri:
            return None
        return {'ruta': ri['ruta'], 'nombre': NOMBRES_RUTA[ri['ruta']], 'variante': ri['variante'],
                'motivo': ri['motivo']}

    def _no_existe(self, customer_id) -> str:
        return _json({'error': f"No hay ningún cliente con CustomerId {customer_id} en el test set. "
                               "Usa buscar_cliente."})

    # --- herramientas ------------------------------------------------------------
    def buscar_cliente(self, consulta: str) -> str:
        consulta = str(consulta).strip()
        X = self.x_test
        if consulta in self._perfiles:
            coincidencias = X.loc[[self._perfiles[consulta]]]
        elif consulta.isdigit():
            coincidencias = X[X['CustomerId'] == int(consulta)]
        else:
            coincidencias = X[X['Surname'].str.contains(consulta, case=False, na=False, regex=False)]
        filas = [{
            'customer_id': int(f['CustomerId']), 'apellido': f['Surname'], 'pais': f['Geography'],
            'edad': int(f['Age']), 'saldo': float(f['Balance']), 'productos': int(f['NumOfProducts']),
            'activo': int(f['IsActiveMember']),
            'perfil_congelado': next((p for p, i in self._perfiles.items() if i == int(idx)), None),
        } for idx, f in coincidencias.head(settings.CHAT_MAX_RESULTADOS_BUSQUEDA).iterrows()]
        return _json({'total_coincidencias': int(len(coincidencias)), 'clientes': filas})

    def resumen_cartera(self, pais: Optional[str] = None, ruta: Optional[int] = None,
                        variante: Optional[str] = None, limite: Optional[int] = None) -> str:
        from backend.decision.cartera import ORIGEN_CALCULADO, cargar_cobertura, construir_vista_cartera, filtrar_cartera
        if self._cobertura is None:
            self._cobertura = cargar_cobertura()
        vista = construir_vista_cartera(self._cobertura, self.x_test, self.ctx.base.df_train,
                                        self.almacen.datos)
        try:
            ruta = int(ruta) if ruta not in (None, "", 0) else None
        except (TypeError, ValueError):
            return _json({'error': f"Ruta no válida: {ruta}. Usa 1, 2 o 3."})
        if ruta is not None and ruta not in NOMBRES_RUTA:
            return _json({'error': f"Ruta no válida: {ruta}. Usa 1, 2 o 3."})
        sel = filtrar_cartera(vista, pais=pais, ruta=ruta, variante=variante)
        try:
            limite = int(limite) if limite else settings.CHAT_CARTERA_LIMITE_DEFECTO
        except (TypeError, ValueError):
            limite = settings.CHAT_CARTERA_LIMITE_DEFECTO
        limite = max(1, min(limite, settings.CHAT_CARTERA_LIMITE_MAX))
        # Salida compacta: cada lista se reenvía al LLM en los mensajes siguientes y el límite de
        # Groq es de 8.000 tokens por minuto (incluida la petición entera).
        clientes = []
        for f in sel.head(limite).itertuples():
            c = {'customer_id': int(f.customer_id), 'apellido': f.apellido}
            if not pais:
                c['pais'] = f.pais
            c.update({'edad': int(f.edad), 'saldo': int(round(float(f.saldo))), 'productos': int(f.productos),
                      'prob_pct': float(f.probabilidad_pct),
                      'orientacion': ORIENTACION_CORTA.get(f.orientacion_calculada, f.orientacion_calculada)})
            if not ruta:
                c['ruta'] = int(f.ruta)
            c['variante'] = f.variante
            c['origen'] = (ORIGEN_CORTO_CALCULADO if f.origen == ORIGEN_CALCULADO
                           else f"Agente Analista ({int(f.herramientas_agente)}/4)")
            clientes.append(c)
        return _json({
            'filtros': {'pais': pais or None, 'ruta': ruta, 'variante': variante or None},
            'total_coincidencias': int(len(sel)),
            'reparto_rutas': {f"Ruta {r}": int(n) for r, n in sel['ruta'].value_counts().sort_index().items()},
            'clientes_con_analisis_real': int((vista['origen'] != ORIGEN_CALCULADO).sum()),
            'clientes': clientes,
            'nota': ("Sin agentes. Orden: mayor probabilidad del modelo. Muestra siempre la columna origen. "
                     "'cálculo 4 herramientas': orientación calculada en Python con las 4 herramientas del "
                     "Analista; 'Agente Analista (n/4)': análisis real del agente con n herramientas. "
                     "Ruta 1 cierre, 2 riesgo alto, 3 revisión."),
        })

    def consultar_predictor(self, customer_id: int, on_evento: Evento = None) -> str:
        fila = self._fila(customer_id)
        if fila is None:
            return self._no_existe(customer_id)
        reg = self._registro(fila)
        tokens, desde_guardado = 0, 'prediccion' in reg
        if not desde_guardado:
            pred = correr_predictor(reg['cliente'], reg['datos'], self.ctx, on_evento)
            if pred is None:
                return _json({'error': "El Agente Predictor no completó tras los reintentos (límite de Groq)."})
            reg.update(pred)
            tokens = pred['tokens_predictor'] or 0
            self._guardar(fila, reg)
        return _json({'agente': 'Predictor', 'cliente': reg['cliente'], 'customer_id': int(customer_id),
                      'churn_prediction': reg['prediccion']['churn_prediction'],
                      'churn_probability_pct': reg['prediccion']['churn_probability_pct'],
                      'router': self._router(reg), 'resultado_guardado': desde_guardado,
                      'tokens_agentes': tokens})

    def consultar_analista(self, customer_id: int, repetir: bool = False, motivo: Optional[str] = None,
                           on_evento: Evento = None) -> str:
        fila = self._fila(customer_id)
        if fila is None:
            return self._no_existe(customer_id)
        reg = self._registro(fila)
        motivo = (motivo or '').strip() or None
        repetir = bool(repetir or motivo)          # dar un motivo implica pedir un reanálisis
        tokens, desde_guardado = 0, 'senales' in reg and not repetir
        anterior = None
        if not desde_guardado:
            if 'senales' in reg:
                # Se conserva un resumen del análisis previo para poder compararlos
                anterior = {'herramientas_invocadas': reg.get('tools_llamadas'),
                            'orientacion_calculada': reg.get('orientacion_calculada'),
                            'orientacion_llm': reg.get('orientacion_llm')}
            analisis = correr_analista(reg['cliente'], reg['datos'], self.ctx, on_evento, motivo=motivo)
            if analisis is None:
                return _json({'error': "El Agente Analista no completó tras los reintentos (límite de Groq)."})
            reg.update(analisis)
            reg['motivo_reanalisis'] = motivo
            reg['analisis_anterior'] = anterior
            reg.pop('recomendacion', None)  # un análisis nuevo invalida la recomendación anterior
            tokens = analisis['tokens_analista'] or 0
            self._guardar(fila, reg)
        return _json({'agente': 'Analista', 'cliente': reg['cliente'], 'customer_id': int(customer_id),
                      'senales': reg['senales'], 'conclusion': reg['conclusion_analista'],
                      'orientacion_llm': reg['orientacion_llm'],
                      'orientacion_calculada': reg['orientacion_calculada'],
                      'herramientas_invocadas': reg.get('tools_llamadas'),
                      'router': self._router(reg), 'resultado_guardado': desde_guardado,
                      'motivo_reanalisis': reg.get('motivo_reanalisis'),
                      'analisis_anterior': reg.get('analisis_anterior'),
                      'tokens_agentes': tokens,
                      'nota': "El router usa orientacion_calculada (producto de multiplicadores), "
                              "nunca orientacion_llm."})

    def recomendar(self, customer_id: int, on_evento: Evento = None) -> str:
        fila = self._fila(customer_id)
        if fila is None:
            return self._no_existe(customer_id)
        reg = self._registro(fila)
        tokens, etapas = 0, []
        avisar = on_evento or (lambda _m: None)

        if 'prediccion' not in reg:
            avisar("Agente Predictor…")
            pred = correr_predictor(reg['cliente'], reg['datos'], self.ctx, on_evento)
            if pred is None:
                return _json({'error': "El Agente Predictor no completó tras los reintentos (límite de Groq)."})
            reg.update(pred)
            tokens += pred['tokens_predictor'] or 0
            etapas.append('Predictor')
            self._guardar(fila, reg)
        if 'senales' not in reg:
            avisar("Agente Analista…")
            analisis = correr_analista(reg['cliente'], reg['datos'], self.ctx, on_evento)
            if analisis is None:
                return _json({'error': "El Agente Analista no completó tras los reintentos (límite de Groq). "
                                       "La predicción queda guardada."})
            reg.update(analisis)
            tokens += analisis['tokens_analista'] or 0
            etapas.append('Analista')
            self._guardar(fila, reg)

        router = self._router(reg)
        base = {'cliente': reg['cliente'], 'customer_id': int(customer_id), 'router': router,
                'etapas_ejecutadas': etapas}
        if reg['ruta_info']['ruta'] == 1:
            return _json({**base, 'recomendacion': None, 'tokens_agentes': tokens,
                          'resultado': "El router cierra el caso (Ruta 1): Predictor y Analista coinciden en "
                                       "riesgo bajo, así que por diseño no se activa el Recomendador."})
        if not reg.get('recomendacion'):
            avisar("Agente Recomendador…")
            salida = correr_recomendador(reg['cliente'], reg['datos'], reg['prediccion'], reg['senales'],
                                         reg['ruta_info'], self.ctx, reg.get('conclusion_analista'), on_evento)
            if salida is None:
                return _json({**base, 'tokens_agentes': tokens,
                              'error': "El Agente Recomendador no completó tras los reintentos (límite de "
                                       "Groq). Predictor y Analista quedan guardados."})
            reg.update(salida)
            tokens += salida['tokens_recomendador'] or 0
            etapas.append('Recomendador')
            self._guardar(fila, reg)
        return _json({**base, 'variante': reg['ruta_info']['variante'],
                      'recomendacion': reg['recomendacion'],
                      'resultado_guardado': not etapas, 'tokens_agentes': tokens})

    # --- confirmación y despacho -------------------------------------------------
    def coste_estimado(self, nombre: str, argumentos: dict) -> Optional[dict]:
        """None si la llamada no consume tokens de agentes (o no aplica); si no, qué se va a
        ejecutar y cuánto cuesta aproximadamente, para que la interfaz pida confirmación."""
        if nombre not in HERRAMIENTAS_CON_CONFIRMACION:
            return None
        try:
            fila = self._fila(argumentos['customer_id'])
        except (KeyError, TypeError, ValueError):
            return None
        if fila is None:
            return None
        reg = self._registro(fila)
        s = settings
        if nombre == 'consultar_analista':
            if 'senales' in reg and not (argumentos.get('repetir') or argumentos.get('motivo')):
                return None
            agentes = [('Analista', s.TOKENS_ESTIMADOS_ANALISTA)]
        else:
            if reg.get('recomendacion') or (reg.get('ruta_info') and reg['ruta_info']['ruta'] == 1):
                return None
            agentes = [a for a, falta in (
                (('Predictor', s.TOKENS_ESTIMADOS_PREDICTOR), 'prediccion' not in reg),
                (('Analista', s.TOKENS_ESTIMADOS_ANALISTA), 'senales' not in reg),
                (('Recomendador', s.TOKENS_ESTIMADOS_RECOMENDADOR), True)) if falta]
        return {'cliente': reg['cliente'], 'customer_id': int(fila['CustomerId']),
                'agentes': [a for a, _ in agentes],
                'tokens_estimados': sum(t for _, t in agentes),
                'nota': ("El Recomendador solo se activa si el router no cierra el caso (Ruta 1)."
                         if nombre == 'recomendar' else "")}

    def ejecutar(self, nombre: str, argumentos: dict, on_evento: Evento = None) -> str:
        metodos = {'buscar_cliente': self.buscar_cliente, 'consultar_predictor': self.consultar_predictor,
                   'consultar_analista': self.consultar_analista, 'recomendar': self.recomendar,
                   'resumen_cartera': self.resumen_cartera}
        if nombre not in metodos:
            return _json({'error': f"Herramienta desconocida: {nombre}"})
        extra = {} if nombre in ('buscar_cliente', 'resumen_cartera') else {'on_evento': on_evento}
        try:
            return metodos[nombre](**argumentos, **extra)
        except TypeError as e:
            return _json({'error': f"Argumentos no válidos para {nombre}: {e}"})
        except Exception as e:  # una herramienta nunca debe tumbar la conversación
            return _json({'error': f"Error al ejecutar {nombre}: {type(e).__name__}: {e}"})
