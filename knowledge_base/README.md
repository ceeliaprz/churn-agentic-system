# knowledge_base/

Conocimiento de negocio que usa el sistema, separado del código (sin RAG).

| Archivo | Qué es | Cómo se genera |
|---|---|---|
| `business_rules.yaml` | Umbrales del router, cortes de lift, palancas intervenibles, canales por variante. **Fuente única de verdad**: `config/settings.py` los lee de aquí. | A mano (valores literales del notebook) |
| `reference_stats.json` | Foto de los estadísticos de `df_train` que usan las tools del Analista: tasa base, cuartiles de saldo, tasas por segmento, anclas de los docstrings. | `python scripts/build_knowledge_base.py` |
| `variables.md` | Diccionario de variables del dataset. | A mano (celda 7 del notebook de ML) |

## Por qué `reference_stats.json` es una foto y no la fuente de las tools

Las tools del Analista siguen calculando sobre `df_train` en memoria, igual que en el
notebook (necesitan el dataframe para, por ejemplo, el percentil exacto de cualquier saldo).
El JSON sirve para documentar los estadísticos de referencia y para comprobar que no han cambiado.
