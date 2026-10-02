"""El asistente habla el protocolo real de LiteLLM → Groq (servidor falso local, sin red ni cuota).

Levanta un servidor HTTP compatible con la API de OpenAI/Groq que primero pide una
herramienta y después responde, y comprueba lo que LiteLLM le envía: modelo, esquemas de
herramientas, reasoning_effort y el mensaje `tool` emparejado con su `tool_call_id`.
"""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("litellm")
pytestmark = [pytest.mark.datos, pytest.mark.crewai]

PETICIONES = []


class _GroqFalso(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        cuerpo = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        PETICIONES.append(cuerpo)
        if len(PETICIONES) == 1:
            mensaje = {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "buscar_cliente", "arguments": json.dumps({"consulta": "Gray"})}}]}
            fin = "tool_calls"
        else:
            mensaje = {"role": "assistant", "content": "Hay dos clientes que se apellidan Gray."}
            fin = "stop"
        datos = json.dumps({"id": "x", "object": "chat.completion", "created": 0,
                            "model": cuerpo["model"], "service_tier": "on_demand",
                            "system_fingerprint": "fp_falso",
                            "choices": [{"index": 0, "message": mensaje, "finish_reason": fin}],
                            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(datos)))
        self.end_headers()
        self.wfile.write(datos)


@pytest.fixture
def servidor():
    PETICIONES.clear()
    srv = HTTPServer(("127.0.0.1", 0), _GroqFalso)
    hilo = threading.Thread(target=srv.serve_forever, daemon=True)
    hilo.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_conversacion_real_con_litellm(servidor):
    import litellm

    from backend.chat.asistente import AsistenteChat
    from backend.chat.herramientas import HerramientasChat

    def completar(**kwargs):
        return litellm.completion(api_base=servidor, api_key="gsk_falsa", **kwargs)

    asistente = AsistenteChat(herramientas=HerramientasChat(), completar=completar)
    turno = asistente.responder("¿Qué clientes se apellidan Gray?")

    assert turno.texto == "Hay dos clientes que se apellidan Gray."
    assert [ll['nombre'] for ll in turno.llamadas] == ["buscar_cliente"]
    assert turno.tokens == 30

    primera, segunda = PETICIONES
    assert primera["model"] == "openai/gpt-oss-20b"
    assert primera["reasoning_effort"] == "low"
    assert {t["function"]["name"] for t in primera["tools"]} == {"buscar_cliente", "consultar_predictor", "consultar_analista", "recomendar"}
    tool_msg = segunda["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
    assert json.loads(tool_msg["content"])["total_coincidencias"] == 2
    assert segunda["messages"][-2]["tool_calls"][0]["id"] == "call_1"
