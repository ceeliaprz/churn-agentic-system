"""LLM falso con guion, compatible con la forma de las respuestas de LiteLLM."""
import json
from types import SimpleNamespace


def llamada(nombre: str, argumentos: dict | str, id_: str = None) -> SimpleNamespace:
    args = argumentos if isinstance(argumentos, str) else json.dumps(argumentos)
    return SimpleNamespace(id=id_ or f"call_{nombre}", type="function",
                           function=SimpleNamespace(name=nombre, arguments=args))


def respuesta(texto: str = None, llamadas: list = None, tokens: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=texto, tool_calls=llamadas or None))],
        usage=SimpleNamespace(total_tokens=tokens))


class LLMFalso:
    """Devuelve las respuestas del guion en orden y guarda lo que recibió en cada llamada.
    Un elemento del guion puede ser una excepción (se lanza) o una función(mensajes) → respuesta."""

    def __init__(self, guion: list):
        self.guion = list(guion)
        self.recibido = []

    def __call__(self, **kwargs):
        self.recibido.append(kwargs)
        if not self.guion:
            raise AssertionError("El LLM falso se ha quedado sin respuestas")
        siguiente = self.guion.pop(0)
        if isinstance(siguiente, Exception):
            raise siguiente
        if callable(siguiente):
            return siguiente(kwargs['messages'])
        return siguiente
