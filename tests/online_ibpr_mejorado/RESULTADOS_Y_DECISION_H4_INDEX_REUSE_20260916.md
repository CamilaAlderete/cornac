# RESULTADOS Y DECISIÓN — H4 REUTILIZACIÓN DEL ÍNDICE

**Experimento:** H4 Index Reuse — OnlineIBPRMejorado  
**Timestamp:** 20260916_062030  
**Protocolo:** `h4_index_reuse_v1_3_final_audited_20260914`  
**Protocol hash:** `a7bb3104d3ff4445935237d20e93a3e78df8448a95cec012299a17b3e628800a`  
**Data SHA256:** `29da5346c5bcf37dc927771d8ffd7ec3323dc7857ed4b0f6a45278b666954d3e`  
**Seeds:** `[777, 999]`

## 1. Objetivo

Evaluar si `OnlineIBPRMejorado`, al mantener `V` exactamente fija y actualizar `U`, puede reutilizar el índice construido sobre `V_base` sin reconstruirlo, manteniendo coherencia top-k frente a búsqueda exhaustiva. Como control operacional, se compara un Full Retrain consultado con el índice obsoleto de `V_base` y con un índice reconstruido sobre `V_full`.

Condiciones:

- `ONLINE_REUSED_INDEX`: `U_online -> index(V_base)`; ground truth exhaustivo sobre `V_online`.
- `FULL_STALE_INDEX`: `U_full -> index(V_base)`; ground truth exhaustivo sobre `V_full`.
- `FULL_REBUILT_INDEX`: `U_full -> index(V_full)`; ground truth exhaustivo sobre `V_full`.

## 2. Auditoría independiente de integridad

Se auditaron los cinco artefactos de la ejecución. Hashes SHA256 de los archivos recibidos:

- log: `ea816e48b371c19c07a2928237f61269596e2238035419aea5176b1203f575da`
- queries: `9a556eb973445a317d8212c159f9f1195eb640005f3a779bea3683f97adf7aee`
- steps: `d93466bff7950e17bcb8544b0ab4a4fbb19097b4601183b52f985294fb438c0e`
- trials: `fffd906fe21e745e4b26e49aa658f7d68efe80902668bcf039048da73813fd72`
- summary: `7e615b2996f6fa37f484406288375f3527b8e87af03a5fdf355710a93fc69a7c`

Conteos observados:

- `queries`: 4.998 filas.
- `steps`: 6 filas = 2 seeds × 3 puntos.
- `trials`: 2 filas = 2 seeds.
- `summary`: 1 fila.
- Por seed/punto aparecen exactamente los mismos usuarios en las tres condiciones.
- Total esperado y observado: `2 × (212 + 315 + 306) × 3 = 4.998` consultas.
- `k_eff = 20` en todas las consultas.
- `candidate_shortfall = 0` en las 4.998 consultas.
- No se detectaron claves duplicadas por `(seed, eval_point, condition, user_idx)`.

## 3. Reconstrucción independiente de métricas

Las listas `ann_item_indices` y `exact_item_indices` se usaron para recalcular por consulta:

- Recall@k_eff
- Position Agreement@k_eff
- Exact Set Match
- Exact Ordered Match
- candidate shortfall

Resultado: **0 discrepancias** frente a las métricas persistidas.

Las agregaciones de `steps`, `trials` y `summary` también fueron reconstruidas desde niveles inferiores. Resultado: **0 discrepancias** en los campos derivados auditados.

## 4. H4a — Evidencia estructural

Todos los invariantes requeridos se cumplieron:

- `V_online == V_base` bit a bit en 6/6 puntos.
- `max |V_online - V_base| = 0`.
- `SHA256(V_online) == SHA256(V_base)` en 6/6 puntos.
- mappings Online/Base exactos en 6/6 puntos.
- mappings Full/Base exactos en 6/6 puntos.
- índice base construido exactamente 1 vez por seed.
- reconstrucciones operacionales Online: 0.
- mismo objeto ANN base reutilizado en los 3 puntos de cada seed.
- SHA del índice base idéntico antes y después de las consultas en 6/6 puntos.
- Full Retrain cambió `V` en 6/6 puntos; no hubo ningún punto con `V_full == V_base`.
- `max |V_full - V_base|` estuvo entre `0.229496` y `0.450265`.
- índice Full reconstruido exactamente 1 vez por punto = 3 por seed.
- SHA del índice Full reconstruido permaneció sin cambios durante sus consultas.
- factores Online y Full fueron de solo lectura durante retrieval.

**Decisión H4a:** respaldada por la evidencia estructural del protocolo. En esta configuración, el corpus vectorial indexado de Online no cambia, por lo que el índice construido sobre `V_base` puede reutilizarse sin reconstrucción a través de los partial updates evaluados.

## 5. H4b — Coherencia de recuperación

Resultados globales, agrupando las 1.666 consultas de cada condición:

| Condición | Recall@20 | Position Agreement | Exact Set Match | Exact Ordered Match | Shortfall rate |
|---|---:|---:|---:|---:|---:|
| ONLINE_REUSED_INDEX | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 0.000000 |
| FULL_STALE_INDEX | 0.724070 | 0.070408 | 0.000600 | 0.000000 | 0.000000 |
| FULL_REBUILT_INDEX | 1.000000 | 0.999940 | 1.000000 | 0.999400 | 0.000000 |

Conteos concretos:

- Online reused: 1.666/1.666 exact set y 1.666/1.666 exact ordered.
- Full stale: 1/1.666 exact set y 0/1.666 exact ordered.
- Full rebuilt: 1.666/1.666 exact set y 1.665/1.666 exact ordered.
- La única discrepancia de orden en Full rebuilt mantiene exactamente el mismo conjunto top-20; difieren dos posiciones en esa consulta.

Por seed/punto, Recall@20 de `FULL_STALE_INDEX` fue:

| Seed | Punto 1 | Punto 2 | Punto 3 |
|---:|---:|---:|---:|
| 777 | 0.732547 | 0.735397 | 0.750327 |
| 999 | 0.741274 | 0.710794 | 0.682026 |

**Interpretación H4b:** en este experimento, la reutilización coherente de `index(V_base)` con `U_online` reprodujo exactamente el top-20 exhaustivo en todas las consultas. Reconstruir el índice sobre `V_full` también recuperó exactamente el conjunto top-20 en todas las consultas. En cambio, consultar `U_full` contra el índice obsoleto de `V_base` produjo una pérdida marcada de coherencia de recuperación. Este control es una comparación entre representaciones incompatibles; no establece causalidad sobre cuánto cambio de `V` produce una pérdida concreta.

## 6. Latencia e indexación

- build base medio: `0.030970 s`.
- rebuild Full medio por punto: `0.030260 s`.
- Online reused ANN mediana: `0.05435 ms`.
- Online exhaustive mediana: `0.05590 ms`.
- Full stale ANN mediana: `0.05310 ms`.
- Full stale exhaustive mediana: `0.05370 ms`.
- Full rebuilt ANN mediana: `0.05195 ms`.
- Full rebuilt exhaustive mediana: `0.05290 ms`.

En MovieLens 1M, con 3.505 ítems base, la diferencia de latencia de consulta es pequeña. Por tanto, H4 no debe interpretarse como demostración de un speedup ANN sustancial o universal. El valor principal de H4 en este dataset es la propiedad estructural de evitar reconstrucciones del índice cuando `V` permanece fija.

## 7. Conclusión científica

**H4 queda cerrada con evidencia favorable a la reutilización del índice bajo el protocolo evaluado.**

La parte estructural está directamente respaldada: `OnlineIBPRMejorado` mantuvo `V` bitwise idéntica a `V_base`, reutilizó el mismo índice sin rebuilds y preservó su estado durante las consultas. La parte de recuperación mostró coherencia top-k perfecta para `ONLINE_REUSED_INDEX` en las 1.666 consultas evaluadas. El control `FULL_STALE_INDEX` mostró que un índice construido sobre una representación de ítems anterior deja de ser coherente cuando Full Retrain relearning cambia `V`; reconstruir el índice sobre `V_full` restableció el conjunto top-k exhaustivo en todas las consultas.

La conclusión queda restringida a MovieLens 1M, las seeds `[777, 999]`, FAISS IVF con `nlist=80`, `nprobe=40`, `k=20`, el universo warm-start y las configuraciones congeladas R900/O014. No se afirma que FAISS IVF sea exacto en general, que estos parámetros ANN sean óptimos, que exista un speedup universal ni que los resultados se transfieran automáticamente a escalas mayores.

## 8. Estado de la hoja de ruta

- H1-H3: cerrado.
- **H4: cerrado.**
- Siguiente etapa: ablaciones focalizadas, sin reabrir HPO.
- Después: validación congelada en MovieLens 10M.
- H5: condicional a la evidencia acumulada.
