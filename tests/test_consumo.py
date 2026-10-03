"""Registro del consumo diario de tokens de Groq (backend/pipeline/consumo.py)."""
from datetime import datetime, timezone
from types import SimpleNamespace


def _t(texto):
    return datetime.strptime(texto, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def test_resumen_por_dia_y_24h():
    from backend.pipeline import consumo
    consumo.registrar(1000, ahora=_t("2026-10-02 23:30"))
    consumo.registrar(2405, ahora=_t("2026-10-03 08:00"))
    consumo.registrar(500, ahora=_t("2026-10-03 17:00"))
    consumo.registrar(0, ahora=_t("2026-10-03 17:01"))           # no se anota
    r = consumo.resumen(ahora=_t("2026-10-03 18:00"))
    assert r['hoy'] == 2905 and r['llamadas_hoy'] == 2           # día UTC
    assert r['ultimas_24h'] == 3905


def test_callback_solo_cuenta_groq():
    from backend.pipeline import consumo
    resp = SimpleNamespace(usage=SimpleNamespace(total_tokens=321))
    consumo._callback_litellm({'model': 'openai/gpt-oss-20b',
                               'litellm_params': {'custom_llm_provider': 'groq'}}, resp, None, None)
    consumo._callback_litellm({'model': 'gpt-4o',
                               'litellm_params': {'custom_llm_provider': 'openai'}}, resp, None, None)
    consumo._callback_litellm({}, None, None, None)                 # nunca lanza
    assert consumo.resumen()['hoy'] == 321


def test_activar_es_idempotente():
    import litellm
    from backend.pipeline import consumo
    consumo.activar()
    consumo.activar()
    assert litellm.success_callback.count(consumo._callback_litellm) == 1


def test_llamada_real_de_litellm_queda_registrada():
    """Con mock_response, LiteLLM no sale a la red pero ejecuta el callback igual."""
    import time
    import litellm
    from backend.pipeline import consumo
    consumo.activar()
    litellm.completion(model="groq/openai/gpt-oss-20b", api_key="gsk_falsa",
                       messages=[{"role": "user", "content": "hola"}], mock_response="hola")
    for _ in range(50):                      # LiteLLM ejecuta el callback en otro hilo
        if consumo.resumen()['llamadas_hoy']:
            break
        time.sleep(0.1)
    assert consumo.resumen()['llamadas_hoy'] == 1
