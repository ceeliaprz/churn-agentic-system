"""Presentación de las llamadas a herramientas dentro del chat (trazabilidad de cada cifra)."""
from __future__ import annotations

import json

import streamlit as st

ETIQUETAS = {
    'buscar_cliente': "🔎 Buscar cliente",
    'consultar_predictor': "🔵 Agente Predictor",
    'consultar_analista': "🟢 Agente Analista",
    'recomendar': "🟣 Recomendar (Predictor/Analista si faltan → router → Agente Recomendador)",
}


def _argumentos(args: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in args.items()) if args else "sin argumentos"


def mostrar_llamadas(llamadas: list) -> None:
    if not llamadas:
        return
    with st.expander(f"Herramientas usadas ({len(llamadas)})"):
        for ll in llamadas:
            st.markdown(f"**{ETIQUETAS.get(ll['nombre'], ll['nombre'])}** · `{_argumentos(ll['argumentos'])}`")
            try:
                contenido = json.dumps(json.loads(ll['resultado']), ensure_ascii=False, indent=2)
            except (TypeError, ValueError):
                contenido = str(ll['resultado'])
            st.code(contenido, language="json", wrap_lines=True)


def mostrar_recomendaciones(llamadas: list) -> None:
    """Tarjeta con la salida estructurada del agente Recomendador (RecomendacionChurn)."""
    for ll in llamadas:
        if ll['nombre'] != 'recomendar':
            continue
        try:
            datos = json.loads(ll['resultado'])
        except (TypeError, ValueError):
            continue
        reco = datos.get('recomendacion')
        if not reco:
            continue
        with st.container(border=True):
            router = datos.get('router') or {}
            st.markdown(f"**🟣 Recomendación del Agente Recomendador — {datos.get('cliente', '')}** "
                        f"({router.get('nombre', '')}, `{router.get('variante', '')}`)")
            st.markdown(f"**Acción:** {reco.get('accion_principal', '—')}")
            st.markdown(f"**Palanca:** {reco.get('palanca_objetivo', '—')} · **Canal:** {reco.get('canal', '—')}")
            for campo, titulo in (('resumen_riesgo', 'Riesgo'), ('justificacion', 'Justificación')):
                if reco.get(campo):
                    st.markdown(f"**{titulo}:** {reco[campo]}")
            if reco.get('limitaciones'):
                st.caption(f"Limitaciones: {reco['limitaciones']}")
            if datos.get('tokens_agentes'):
                st.caption(f"{datos['tokens_agentes']:,} tokens de los agentes".replace(",", "."))
