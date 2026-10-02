"""Registro de llamadas a tools (notebook agéntico, celda 29).

Instrumentación, no entra en el prompt: permite contrastar qué herramientas usó realmente
el agente frente a las que dice haber usado, y es la entrada de `balance_de_senales`.

Diferencia con el notebook: allí `registro_llamadas` era una lista global compartida. Aquí
cada ejecución crea su propio `RegistroLlamadas`, y las tools se construyen ligadas a él
(ver tools.py). Así dos sesiones del frontend no se mezclan.
"""
from __future__ import annotations

from functools import wraps


class RegistroLlamadas:
    def __init__(self):
        self.llamadas: list[dict] = []

    def registrar(self, nombre: str):
        """Envuelve una tool para dejar constancia de su llamada y de su resultado.
        @wraps conserva firma y docstring, de modo que CrewAI sigue generando el mismo
        esquema de parámetros que si la función no estuviera decorada."""
        def decorador(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                resultado = func(*args, **kwargs)
                self.llamadas.append({'tool': nombre, 'resultado': resultado})
                return resultado
            return wrapper
        return decorador

    def iniciar(self) -> None:
        """Vacía el registro. Llámala justo antes de cada kickoff."""
        self.llamadas.clear()

    def tools_llamadas(self) -> list:
        """Nombres de las tools realmente invocadas, en orden y sin repetidos."""
        vistas = []
        for llamada in self.llamadas:
            if llamada['tool'] not in vistas:
                vistas.append(llamada['tool'])
        return vistas
