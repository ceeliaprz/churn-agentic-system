"""Experimento de variabilidad de selección de tools del Analista

Resultado ya cerrado y documentado (hallazgos_analista_para_memoria.md): la selección de
herramientas del Analista NO es fiable, ni con 20B. Se porta para poder
reproducirlo; ejecutarlo consume cuota de Groq (5 perfiles × N repeticiones).

Los CSV de resultados originales están en data/results/notebook/.
"""
from __future__ import annotations

import ast
import asyncio
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def jaccard(set_a, set_b) -> float:
    set_a, set_b = set(set_a), set(set_b)
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)


def cargar_experimento(ruta: Path) -> pd.DataFrame:
    """Lee un CSV del experimento. Las columnas de listas vuelven como texto: literal_eval."""
    df = pd.read_csv(ruta)
    for col in ('tools_orden', 'tools_llamadas'):
        if col in df.columns:
            df[col] = df[col].apply(lambda v: ast.literal_eval(v) if isinstance(v, str) else v)
    return df


def anadir_tokens_por_run(df: pd.DataFrame) -> pd.DataFrame:
    """La columna 'tokens' viene acumulada a lo largo de la sesión: la diferencia entre
    filas consecutivas es lo que costó realmente cada ejecución."""
    df = df.copy()
    df['tokens_run'] = df['tokens'].diff()
    df.loc[df.index[0], 'tokens_run'] = df.loc[df.index[0], 'tokens']  # la 1ª parte de cero
    return df


def estabilidad_por_perfil(df: pd.DataFrame) -> dict:
    """Jaccard medio entre pares de runs del mismo perfil (None si hay menos de 2 válidas)."""
    salida = {}
    for perfil in df['perfil'].unique():
        subset = df[(df['perfil'] == perfil) & df['tools_llamadas'].notna()]['tools_llamadas'].tolist()
        pares = list(combinations(subset, 2))
        salida[perfil] = float(np.mean([jaccard(a, b) for a, b in pares])) if pares else None
    return salida


def frecuencia_tools_por_perfil(df: pd.DataFrame) -> dict:
    """{perfil: {'n_runs', 'media_tools', 'frecuencias': {tool: veces}}}"""
    salida = {}
    for perfil in df['perfil'].unique():
        subset = df[(df['perfil'] == perfil) & df['tools_llamadas'].notna()]
        if subset.empty:
            continue
        todas = [t for lista in subset['tools_llamadas'] for t in lista]
        salida[perfil] = {
            'n_runs': len(subset),
            'media_tools': float(subset['n_tools'].mean()),
            'frecuencias': {t: todas.count(t) for t in sorted(set(todas))},
        }
    return salida


async def ejecutar_experimento(perfiles_test: dict, ruta_csv: Path, n_repeticiones: int = 5,
                               pausa_s: float = 60, ctx=None, on_evento=print) -> pd.DataFrame:
    """Reejecuta el Analista N veces por perfil y guarda tras cada run (consume cuota)."""
    from crewai import Crew, Process

    from backend.agents.analyst import analizar_cliente, crear_agente_analista
    from backend.agents.tool_registry import RegistroLlamadas
    from backend.agents.tools import crear_tools_analista
    from backend.analytics.signals import balance_de_senales
    from backend.pipeline.cases import datos_cliente
    from backend.pipeline.orchestrator import ContextoSistema
    from backend.pipeline.retries import ejecutar_con_reintentos

    ctx = ctx or ContextoSistema.por_defecto()
    resultados = []
    for perfil_nombre, fila in perfiles_test.items():
        nombre_cliente = fila['Surname']
        customer_data = datos_cliente(fila)

        for i in range(n_repeticiones):
            registro = RegistroLlamadas()
            agente = crear_agente_analista(crear_tools_analista(ctx.base, registro),
                                           ctx.base.referencias_texto)
            tarea = analizar_cliente(nombre_cliente, customer_data, agente)
            crew = Crew(agents=[agente], tasks=[tarea], process=Process.sequential)
            try:
                resultado = await ejecutar_con_reintentos(crew, on_evento=on_evento,
                                                          antes_de_cada_intento=registro.iniciar)
            except Exception as e:
                on_evento(f"❌ [{perfil_nombre} - run {i+1}] Error no recuperable: {e}")
                resultado = None

            fila_res = {'perfil': perfil_nombre, 'cliente': nombre_cliente, 'run': i + 1}
            if resultado is None:
                on_evento(f"❌ [{perfil_nombre} - run {i+1}] Fallido tras reintentos, se omite.")
                fila_res.update({'tools_llamadas': None, 'n_tools': None,
                                 'orientacion': None, 'coincide_balance': None, 'tokens': None})
            else:
                tools = registro.tools_llamadas()
                esperada = balance_de_senales(registro.llamadas)
                analisis = resultado.pydantic
                uso = getattr(resultado, 'token_usage', None)
                fila_res.update({
                    'tools_orden': list(tools),          # orden real de llamada
                    'tools_llamadas': sorted(tools),     # conjunto, para el Jaccard
                    'n_tools': len(tools),
                    'orientacion': analisis.orientacion if analisis else None,
                    'coincide_balance': (analisis.orientacion == esperada) if analisis else None,
                    'conclusion': analisis.conclusion if analisis else None,
                    'tokens': uso.total_tokens if uso else None,
                })
                on_evento(f"[{perfil_nombre} - run {i+1}] {len(tools)} tools: {sorted(tools)}")

            resultados.append(fila_res)
            Path(ruta_csv).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(resultados).to_csv(ruta_csv, index=False)
            await asyncio.sleep(pausa_s)

    return pd.DataFrame(resultados)
