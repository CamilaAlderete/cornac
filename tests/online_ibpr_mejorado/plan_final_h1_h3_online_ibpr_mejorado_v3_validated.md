# Plan definitivo H1-H3 — V3 validada

## 1. Motivo de la V3

La ejecución `--plan-only` de la V2 reveló una característica metodológicamente importante de MovieLens 1M bajo un corte cronológico global 60/40:

```text
Future raw rows      : 334,591
Future warm rows     : 52,467
Warm-start fraction  : 15.6809%
```

La baja retención warm-start no constituye un error del script: la mayor parte del 40% final pertenece a usuarios que no existían todavía en el 60% base.

El problema de la V2 era distinto: primero dividía el stream raw en cuatro chunks aproximadamente iguales y después aplicaba el filtro warm-start. Como consecuencia, los chunks efectivos quedaban muy desequilibrados:

```text
4,390
4,077
5,824
38,176
```

Además, la población evaluada podía contener usuarios que todavía no habían recibido ninguna interacción nueva en los chunks de actualización anteriores.

La V3 corrige exclusivamente esos dos puntos.

---

## 2. Decisiones que permanecen congeladas

No se modifica ninguna decisión de HPO ni ninguna configuración del modelo.

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

ID: `R900`.

```python
ONLINE_ADAPTATION_CONFIG = {
    "learning_rate": 0.005,
    "lamda": 1e-06,
    "batch_size": 1024,
    "n_epochs": 3,
    "loss_mode": "angular",
    "update_V": False,
    "neg_sampling": "uniform",
    "normalize": True,
    "max_steps": None,
}
```

ID: `O014`.

También permanecen congelados:

```text
Dataset principal              MovieLens 1M
Feedback positivo              rating >= 3 -> 1.0
Corte desarrollo/final         60% / 40% cronológico global
Seeds finales                  [777, 999]
Métrica primaria               NDCG@20
Ramas                          Stale / Online / Full Retrain
Número de chunks               4
Puntos prequentiales           3
Retuning                       prohibido
```

---

## 3. Cambio 1 — filtrar warm-start antes de dividir el stream

### V2

```text
40% future raw
      ↓
dividir en 4 chunks raw
      ↓
filtrar usuarios/ítems conocidos
      ↓
chunks warm muy desbalanceados
```

### V3

```text
40% future raw
      ↓
filtrar:
user ∈ base_users
item ∈ base_items
      ↓
stream warm-start válido
      ↓
dividir ese stream en 4 chunks cronológicos
aproximadamente iguales
```

Los límites internos se ajustan hacia adelante hasta terminar el grupo completo de timestamp que cruza la frontera.

Por tanto:

```text
max_timestamp(chunk_t) < min_timestamp(chunk_t+1)
```

debe cumplirse estrictamente.

Los eventos excluidos por cold-start se siguen contabilizando y reportando.

La V3 no intenta resolver cold-start ni incorporar esos eventos artificialmente.

---

## 4. Cambio 2 — población primaria de evaluación

La evaluación primaria de H1/H3 debe medir adaptación.

Para cada punto:

```text
chunk1 update -> chunk2 eval
chunk2 update -> chunk3 eval
chunk3 update -> chunk4 eval
```

se mantiene un conjunto acumulado:

```text
updated_users =
usuarios que recibieron al menos una interacción
en alguno de los chunks de actualización ya observados
```

La población primaria del chunk de evaluación es:

```text
eval_primary =
filas warm del chunk de evaluación
cuyo usuario ∈ updated_users
```

Así, H1 responde directamente:

> ¿Mejora OnlineIBPRMejorado la calidad para usuarios conocidos que efectivamente han recibido nueva información incremental?

Las tres ramas se evalúan exactamente sobre esta misma población.

---

## 5. Evaluación suplementaria ALL-WARM

No se descartan del análisis los demás usuarios warm-start.

Además de la población primaria, la V3 evalúa las tres ramas sobre:

```text
ALL-WARM =
todas las filas warm-start del chunk de evaluación
```

Esta medición es complementaria y se registra con prefijo:

```text
allwarm_*
```

Ejemplos:

```text
allwarm_stale_NDCG@20
allwarm_online_NDCG@20
allwarm_full_NDCG@20
allwarm_online_minus_stale_NDCG@20
```

Estas métricas no sustituyen la prueba primaria de H1/H3.

---

## 6. Secuencia experimental

Para cada seed final:

```text
1. Entrenar IBPR R900 sobre el 60% base.

2. Crear:
   - IBPR_STALE
   - OnlineIBPRMejorado O014

3. Chunk 1:
   - Online partial update
   - Full Retrain desde cero sobre base + chunk1
   - evaluar chunk2

4. Chunk 2:
   - Online partial update
   - Full Retrain desde cero sobre base + chunk1 + chunk2
   - evaluar chunk3

5. Chunk 3:
   - Online partial update
   - Full Retrain desde cero sobre base + chunk1 + chunk2 + chunk3
   - evaluar chunk4
```

Cada evaluación produce:

```text
PRIMARY
+
ALL-WARM supplementary
```

---

## 7. H1

Comparación primaria:

```text
OnlineIBPRMejorado - IBPR_STALE
```

Métrica principal:

```text
NDCG@20
```

También se registran:

```text
AUC
MAP
Precision@20
Recall@20
```

Los deltas se registran por seed y punto temporal.

---

## 8. H2

Se mide:

```text
partial_fit_recent()
vs
IBPR.fit() del Full Retrain
```

El tiempo de construcción de datasets se registra separadamente.

Métricas:

```text
online_update_time_s
full_retrain_time_s
step_speedup_full_over_online
step_online_full_cost_fraction
cumulative_speedup_full_over_online
cumulative_online_full_cost_fraction
```

La comparación H2 no incluye como tiempo primario la evaluación ni la construcción del dataset.

---

## 9. H3

Se registra:

```text
Online - Full Retrain
Full Retrain - Stale
```

Además:

```text
adaptation_recovery =
(Online - Stale) / (FullRetrain - Stale)
```

sólo cuando:

```text
FullRetrain - Stale > 0
```

`adaptation_recovery` es descriptiva y no constituye una prueba de equivalencia.

---

## 10. Invariantes

La V3 mantiene los siguientes invariantes:

```text
Stale U == U_base exacto
Stale V == V_base exacto

Online V == V_base exacto
max_abs_diff(Online V, V_base) == 0

uid_map fijo
iid_map fijo
n_users fijo
n_items fijo

ningún par (u,i) evaluado pudo haber sido observado antes
```

Cualquier violación aborta el experimento.

---

## 11. Fingerprint y resume

El protocolo incorpora:

```text
protocol_version = final_h1_h3_v3_20260914
data SHA256
script SHA256
IBPR wrapper/core SHA256
Online wrapper/core SHA256
configuraciones R900/O014
seeds
regla de split
regla warm-start
regla de población primaria
```

El resume acepta una seed únicamente si posee exactamente:

```text
3 step rows
1 trial row

eval_points   = {1,2,3}
update_chunks = {1,2,3}
eval_chunks   = {2,3,4}
```

No deben reutilizarse resultados de V2 con V3.

---

## 12. Auditoría previa realizada

Antes de entregar el script se verificó:

```text
Python syntax / py_compile                           OK
STEP_FIELDS sin duplicados                          OK
TRIAL_FIELDS sin duplicados                         OK
SUMMARY_FIELDS sin duplicados                       OK
summary generado == schema declarado                OK
split warm sintético aproximadamente equilibrado    OK
separación temporal estricta sintética              OK
población primaria acumulativa sintética             OK
```

El entorno de generación no dispone de Cornac, por lo que la validación final de integración debe realizarse en el repositorio del usuario mediante `--plan-only`.

---

## 13. Próximo paso autorizado

Copiar:

```text
final_h1_h3_online_ibpr_mejorado_v3.py
```

al repositorio y ejecutar únicamente:

```powershell
python tests/online_ibpr_mejorado/final_h1_h3_online_ibpr_mejorado_v3.py --plan-only
```

Antes de autorizar la corrida definitiva se debe revisar:

```text
1. mismo total de positivos;
2. mismo corte 60/40;
3. mismo warm fraction ~15.68%;
4. cuatro chunks warm aproximadamente equilibrados;
5. separación temporal estricta;
6. población PRIMARY no vacía en los 3 puntos;
7. cantidad de usuarios PRIMARY;
8. porcentaje PRIMARY / ALL-WARM;
9. R900 y O014 intactos;
10. seeds [777,999];
11. nuevo protocol hash V3.
```

No ejecutar todavía sin `--plan-only`.
