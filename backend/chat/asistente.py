"""Asistente conversacional: una capa encima del sistema multiagente del notebook.

El Asistente es un LLM con function calling cuyas herramientas son esos agentes (backend/chat/herramientas.py):
en cada pregunta decide qué agentes activar ("¿qué riesgo tiene?" → solo el Predictor;
"¿por qué?" → el Analista; "¿qué hacemos?" → recomendar, que completa la cadena y aplica el
router en Python), si hace falta repetir un análisis y cuándo responder.

Bucle por mensaje del usuario:
    LLM → (tool_calls → ejecutar → resultados → LLM)* → respuesta final
con un máximo de `CHAT_MAX_PASOS` llamadas al LLM. Las herramientas que activan agentes caros
(Analista, Recomendador) PAUSAN el bucle y devuelven `pendiente` para que la interfaz pida
confirmación del coste; `confirmar()` lo reanuda.

El Asistente llama a Groq directamente con LiteLLM (CrewAI también usa LiteLLM por debajo):
así cada llamada lleva solo el historial y los esquemas de las herramientas, y la conversación
multi-turno no tiene que reconstruirse como un Crew nuevo en cada mensaje.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from config import settings
from backend.chat.herramientas import ESQUEMAS, HerramientasChat
from backend.pipeline.retries import extraer_espera_sugerida

Evento = Optional[Callable[[str], None]]

RESULTADO_RECORTADO = ('{"recortado": true, "nota": "Resultado de un turno anterior recortado para no '
                       'superar el límite de Groq. Vuelve a llamar a la herramienta si lo necesitas."}')


def es_peticion_demasiado_grande(texto_error: str) -> bool:
    t = texto_error.lower()
    return "request too large" in t or "reduce your message size" in t


def completar_litellm(**kwargs):
    """Llamada al LLM (Groq vía LiteLLM)."""
    import litellm
    from backend.pipeline import consumo
    consumo.activar()                     # registra los tokens de hoy (barra lateral)
    return litellm.completion(**kwargs)


def prompt_sistema() -> str:
    s = settings
    return f"""Eres el asistente de retención de clientes de un banco. Das acceso conversacional a un sistema multiagente de predicción y retención de churn (TFM) sobre el dataset Churn Modelling (2.000 clientes de test).

Tus herramientas activan los agentes y algunas herramientas sueltas del sistema:
- buscar_cliente: encuentra al cliente (gratis). Úsala primero si solo te dan un apellido; si hay varias coincidencias, enséñalas en una lista corta y pregunta cuál.
- consultar_predictor: Agente Predictor (modelo LightGBM). Su probabilidad es el riesgo oficial.
- consultar_analista: Agente Analista. Explica el porqué con señales frente al histórico; cada señal lleva un multiplicador x = churn del segmento / churn medio.
- recomendar: completa lo que falte (Predictor, Analista), aplica el router en Python y, si procede, activa al Agente Recomendador.
- resumen_cartera: vista de los 2.000 clientes sin activar agentes (gratis): reparto por rutas y clientes ordenados por probabilidad, con filtros por país, ruta y variante. Para preguntas sobre muchos clientes, conteos o a quién priorizar.

Cómo decide el sistema (para explicarlo, no para calcularlo): el router cierra el caso sin recomendación (Ruta 1) si la probabilidad es < {s.UMBRAL_PROB_BAJA:.0f}% y la orientación calculada del Analista está por debajo de la media; manda a Ruta 2 (riesgo alto) si la probabilidad es ≥ {s.UMBRAL_PROB_ALTA:.0f}%; todo lo demás va a Ruta 3 (revisión), es decir, probabilidad entre {s.UMBRAL_PROB_BAJA:.0f}% y {s.UMBRAL_PROB_ALTA:.0f}%, o probabilidad < {s.UMBRAL_PROB_BAJA:.0f}% con orientación calculada en línea o por encima de la media. La variante del Recomendador es alto_valor (gestor personal) si el percentil de saldo ≥ {s.UMBRAL_VALOR_ALTO:.0f}; si no, cartera_masiva (campaña automatizada o contact center). El router usa la orientación calculada (producto de multiplicadores), nunca el juicio del LLM.

Reglas:
1. Activa solo los agentes que la pregunta necesita: riesgo → Predictor; porqué o perfil → Analista; qué hacer → recomendar (no llames antes a los otros para eso: recomendar ya lo hace). Llama a recomendar solo si el usuario pide qué hacer con el cliente o una acción de retención; para preguntas sobre la ruta, el riesgo o el porqué usa consultar_predictor y consultar_analista, que también devuelven la ruta del router. Si un agente ya se ejecutó para ese cliente en esta conversación, usa lo que devolvió en lugar de volver a llamarlo.
1b. Si le pediste al usuario una aclaración (p. ej. cuál de varios clientes) y responde, retoma su petición original con ese cliente: si preguntó qué hacer, llama a recomendar. Antes de decir que un agente no se ha ejecutado, mira los campos resultado_guardado y analisis_guardado de las herramientas.
2. Toda cifra que des debe salir de una herramienta llamada en esta conversación. Nunca estimes ni recuerdes cifras.
3. No cambies ni recalcules la ruta, la variante ni la orientación: vienen del sistema.
4. Nunca propongas, sugieras ni anticipes acciones de retención, ni siquiera como ejemplo o en condicional (nada de "típicamente", "se podría" ni "próximos pasos"). Las acciones solo las propone el Agente Recomendador con la herramienta recomendar. Si no lo has activado en esta conversación, como mucho termina ofreciendo activarlo: "Si quieres una acción concreta, puedo activar al Agente Recomendador".
5. Si el Analista consultó pocas herramientas y el usuario pide más evidencia, puedes repetir el análisis (consultar_analista con repetir=true). Indica en motivo qué herramientas no consultó y por qué pueden importar para este cliente, según sus datos (p. ej. país o edad de un grupo con más abandono). Al presentar el nuevo análisis, empieza siempre con una comparación breve (1–2 frases) con analisis_anterior: qué herramientas se consultaron antes y ahora, y si la orientación calculada y la ruta cambian o se mantienen.
6. Si una herramienta devuelve un error o el usuario cancela, explícalo y propón cómo seguir.
7. Para preguntas sobre varios clientes o sobre la cartera usa resumen_cartera, no los agentes. No pases limite salvo que el usuario pida un número concreto de clientes. Al mostrar la lista, incluye siempre la columna Origen (cálculo con las 4 herramientas o análisis real del Agente Analista) y di cuántos clientes hay en total. Si después piden el porqué o qué hacer con un cliente concreto de la lista, usa los agentes con su customer_id.
8. Cuando recomendar devuelve una recomendación, la interfaz ya la muestra completa en una tarjeta: no la repitas ni la pases a tabla; resúmela en 1–2 frases (ruta y acción principal) y no añadas notas sobre otras acciones. Responde en español, breve y claro, en markdown, sin etiquetas HTML (nada de <br>). Al explicar señales, cita el multiplicador (p. ej. x1.33)."""


@dataclass
class Turno:
    """Resultado de un mensaje del usuario (o de una confirmación)."""
    texto: str = ""
    llamadas: list = field(default_factory=list)   # [{'nombre', 'argumentos', 'resultado'}]
    pendiente: Optional[dict] = None               # acción que espera confirmación
    tokens: int = 0


class AsistenteChat:
    def __init__(self, herramientas: Optional[HerramientasChat] = None,
                 completar: Optional[Callable] = None):
        self.herramientas = herramientas or HerramientasChat()
        self._completar = completar
        self.mensajes: list[dict] = []          # historial sin el mensaje de sistema
        self.pendiente: Optional[dict] = None
        self.tokens_llm = 0
        self.tokens_agentes = 0          # tokens de los agentes CrewAI activados desde el chat

    @property
    def tokens_totales(self) -> int:
        return self.tokens_llm + self.tokens_agentes

    # --- API pública ---------------------------------------------------------
    def responder(self, texto_usuario: str, on_evento: Evento = None) -> Turno:
        if self.pendiente:
            self._resolver_pendiente(aceptar=False, motivo="El usuario escribió otro mensaje en "
                                                           "lugar de confirmar la ejecución.")
        inicio = len(self.mensajes)
        self.mensajes.append({"role": "user", "content": texto_usuario})
        try:
            return self._bucle(Turno(), on_evento)
        except Exception:
            # Si falló antes de avanzar, se retira el mensaje para poder repetirlo sin duplicarlo
            if len(self.mensajes) == inicio + 1:
                self.mensajes.pop()
            raise

    def confirmar(self, aceptar: bool, on_evento: Evento = None) -> Turno:
        if not self.pendiente:
            return Turno(texto="No hay ninguna acción pendiente de confirmar.")
        turno = Turno(llamadas=list(self.pendiente['llamadas']))
        turno.llamadas += self._resolver_pendiente(aceptar, on_evento=on_evento)
        return self._bucle(turno, on_evento)

    def reiniciar(self) -> None:
        """«Nueva conversación»: borra el historial y pone a cero los tokens de la conversación.
        El consumo del día (backend/pipeline/consumo.py) no se toca."""
        self.mensajes.clear()
        self.pendiente = None
        self.tokens_llm = 0
        self.tokens_agentes = 0

    # --- internos ------------------------------------------------------------
    @staticmethod
    def _tokens_estimados(mensajes: list[dict]) -> int:
        return sum(len(json.dumps(m, ensure_ascii=False)) for m in mensajes) // settings.CHAT_CARACTERES_POR_TOKEN

    def _mensajes_para_llm(self, agresivo: bool = False) -> list[dict]:
        """Historial que se envía al LLM, recortado para no superar el límite de Groq.

        1. Como mucho CHAT_MAX_MENSAJES mensajes, empezando siempre en un mensaje de usuario
           (para no romper los pares tool_call/tool).
        2. Si supera el presupuesto de tokens, los resultados de herramientas de turnos antiguos
           se sustituyen por un aviso corto. Se conservan enteros los del último turno y los del
           anterior (para preguntas como "analiza al primero de la lista"); con agresivo=True,
           solo los del último.
        3. Si aún lo supera, se quitan turnos completos, del más antiguo al más reciente.
        """
        mensajes = list(self.mensajes)
        if len(mensajes) > settings.CHAT_MAX_MENSAJES:
            corte = len(mensajes) - settings.CHAT_MAX_MENSAJES
            usuarios = [i for i, m in enumerate(mensajes) if m['role'] == 'user']
            candidatos = [i for i in usuarios if i >= corte]
            inicio = candidatos[0] if candidatos else (usuarios[-1] if usuarios else 0)
            mensajes = mensajes[inicio:]

        presupuesto = settings.CHAT_PRESUPUESTO_TOKENS_HISTORIAL
        if agresivo or self._tokens_estimados(mensajes) > presupuesto:
            usuarios = [i for i, m in enumerate(mensajes) if m['role'] == 'user']
            conservar_desde = (usuarios[-1] if agresivo or len(usuarios) < 2 else usuarios[-2]) if usuarios else 0
            mensajes = [
                {**m, 'content': RESULTADO_RECORTADO}
                if m['role'] == 'tool' and i < conservar_desde and len(m.get('content') or '') > len(RESULTADO_RECORTADO)
                else m
                for i, m in enumerate(mensajes)]
        while self._tokens_estimados(mensajes) > presupuesto:
            usuarios = [i for i, m in enumerate(mensajes) if m['role'] == 'user']
            if len(usuarios) <= 1:
                break
            mensajes = mensajes[usuarios[1]:]
        return [{"role": "system", "content": prompt_sistema()}] + mensajes

    def _llamar_llm(self, on_evento: Evento):
        completar = self._completar or completar_litellm
        avisar = on_evento or (lambda _m: None)
        ultimo_error = None
        agresivo = False
        for intento in range(1, settings.MAX_INTENTOS + 1):
            try:
                respuesta = completar(messages=self._mensajes_para_llm(agresivo=agresivo), tools=ESQUEMAS,
                                      tool_choice="auto", **settings.LLM_CHAT)
                uso = getattr(respuesta, 'usage', None)
                tokens = getattr(uso, 'total_tokens', None) or 0
                self.tokens_llm += tokens
                return respuesta, tokens
            except Exception as e:
                ultimo_error = e
                texto = str(e)
                if es_peticion_demasiado_grande(texto):
                    # Esperar no sirve: la petición sola ya supera el límite por minuto.
                    if agresivo:
                        raise RuntimeError(
                            "La conversación es demasiado larga para el límite de Groq (8.000 tokens por "
                            "minuto). Pulsa «Nueva conversación» y vuelve a preguntar.") from e
                    agresivo = True
                    avisar("⚠️ La petición era demasiado grande para Groq: recorto el historial y reintento...")
                    continue
                if "RateLimitError" in str(type(e)) or "rate_limit" in texto.lower():
                    espera = extraer_espera_sugerida(texto, default=settings.ESPERA_DEFAULT_S)
                    avisar(f"⚠️ Límite de Groq alcanzado: espero {espera:.0f} s "
                           f"(intento {intento}/{settings.MAX_INTENTOS})...")
                    time.sleep(espera)
                elif "tool_use_failed" in texto or "<|channel|>" in texto:
                    avisar(f"⚠️ El modelo generó una llamada a herramienta mal formada; reintento "
                           f"({intento}/{settings.MAX_INTENTOS})...")
                    time.sleep(2)
                else:
                    raise
        raise RuntimeError(f"El LLM no respondió tras {settings.MAX_INTENTOS} intentos: {ultimo_error}")

    @staticmethod
    def _extraer(respuesta):
        mensaje = respuesta.choices[0].message
        contenido = getattr(mensaje, 'content', None)
        tool_calls = []
        for tc in (getattr(mensaje, 'tool_calls', None) or []):
            funcion = tc.function if hasattr(tc, 'function') else tc['function']
            tool_calls.append({
                'id': tc.id if hasattr(tc, 'id') else tc['id'],
                'nombre': funcion.name if hasattr(funcion, 'name') else funcion['name'],
                'argumentos_json': (funcion.arguments if hasattr(funcion, 'arguments')
                                    else funcion['arguments']) or "{}",
            })
        return contenido, tool_calls

    def _necesita_confirmacion(self, nombre: str, argumentos: dict) -> Optional[dict]:
        return self.herramientas.coste_estimado(nombre, argumentos)

    def _ejecutar(self, tc: dict, on_evento: Evento) -> dict:
        try:
            argumentos = json.loads(tc['argumentos_json'])
            if not isinstance(argumentos, dict):
                raise ValueError("los argumentos no son un objeto")
        except (json.JSONDecodeError, ValueError) as e:
            resultado = json.dumps({'error': f"Argumentos JSON no válidos: {e}"}, ensure_ascii=False)
            argumentos = {}
        else:
            resultado = self.herramientas.ejecutar(tc['nombre'], argumentos, on_evento=on_evento)
        self._anadir_resultado(tc, resultado)
        return {'nombre': tc['nombre'], 'argumentos': argumentos, 'resultado': resultado}

    def _anadir_resultado(self, tc: dict, resultado: str) -> None:
        self.mensajes.append({"role": "tool", "tool_call_id": tc['id'], "name": tc['nombre'],
                              "content": resultado})
        try:
            self.tokens_agentes += int(json.loads(resultado).get('tokens_agentes') or 0)
        except (ValueError, AttributeError, TypeError):
            pass

    def _bucle(self, turno: Turno, on_evento: Evento) -> Turno:
        for _ in range(settings.CHAT_MAX_PASOS):
            respuesta, tokens = self._llamar_llm(on_evento)
            turno.tokens += tokens
            contenido, tool_calls = self._extraer(respuesta)

            if not tool_calls:
                self.mensajes.append({"role": "assistant", "content": contenido or ""})
                turno.texto = contenido or ""
                return turno

            self.mensajes.append({"role": "assistant", "content": contenido or None, "tool_calls": [
                {"id": tc['id'], "type": "function",
                 "function": {"name": tc['nombre'], "arguments": tc['argumentos_json']}}
                for tc in tool_calls]})

            for i, tc in enumerate(tool_calls):
                try:
                    argumentos = json.loads(tc['argumentos_json'])
                except json.JSONDecodeError:
                    argumentos = None
                previa = (self._necesita_confirmacion(tc['nombre'], argumentos)
                          if isinstance(argumentos, dict) else None)
                if previa:
                    self.pendiente = {'tool_call': tc, 'argumentos': argumentos, 'previa': previa,
                                      'resto': tool_calls[i + 1:], 'llamadas': list(turno.llamadas)}
                    turno.pendiente = previa
                    turno.texto = contenido or ""
                    return turno
                turno.llamadas.append(self._ejecutar(tc, on_evento))

        turno.texto = ("He llegado al límite de pasos para esta pregunta. ¿Puedes concretarla un poco "
                       "más o dividirla en partes?")
        self.mensajes.append({"role": "assistant", "content": turno.texto})
        return turno

    def _resolver_pendiente(self, aceptar: bool, motivo: str = "El usuario ha cancelado la ejecución.",
                            on_evento: Evento = None) -> list:
        """Da respuesta a la tool_call pendiente (y a las que venían detrás) para que el historial
        quede consistente. Devuelve las llamadas ejecutadas."""
        p, self.pendiente = self.pendiente, None
        llamadas = []
        if aceptar:
            llamadas.append(self._ejecutar(p['tool_call'], on_evento))
        else:
            resultado = json.dumps({'cancelado': True, 'motivo': motivo,
                                    'nota': "No se ha invocado al Recomendador ni se han gastado tokens."},
                                   ensure_ascii=False)
            self._anadir_resultado(p['tool_call'], resultado)
            llamadas.append({'nombre': p['tool_call']['nombre'], 'argumentos': p['argumentos'],
                             'resultado': resultado})
        for tc in p['resto']:
            try:
                args_tc = json.loads(tc['argumentos_json'])
            except json.JSONDecodeError:
                args_tc = None
            con_coste = isinstance(args_tc, dict) and self._necesita_confirmacion(tc['nombre'], args_tc)
            if con_coste:
                resultado = json.dumps({'cancelado': True, 'motivo': "Solo se activa un agente con coste "
                                        "por confirmación; pídelo de nuevo si lo necesitas."}, ensure_ascii=False)
                self._anadir_resultado(tc, resultado)
                llamadas.append({'nombre': tc['nombre'], 'argumentos': {}, 'resultado': resultado})
            elif aceptar:
                llamadas.append(self._ejecutar(tc, on_evento))
            else:
                resultado = json.dumps({'cancelado': True, 'motivo': motivo}, ensure_ascii=False)
                self._anadir_resultado(tc, resultado)
                llamadas.append({'nombre': tc['nombre'], 'argumentos': {}, 'resultado': resultado})
        return llamadas
