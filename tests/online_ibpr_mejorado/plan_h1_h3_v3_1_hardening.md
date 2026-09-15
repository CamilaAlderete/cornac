# H1-H3 V3.1 — endurecimiento previo a ejecución final

## Estado

Esta versión conserva exactamente el protocolo experimental de V3:

- MovieLens 1M.
- `rating >= 3 -> 1.0`.
- corte cronológico global 60/40.
- filtrado warm-start antes de dividir el stream.
- 4 chunks warm cronológicos.
- población PRIMARY basada en usuarios expuestos previamente a una actualización.
- ALL-WARM como diagnóstico complementario.
- IBPR R900 congelado.
- OnlineIBPRMejorado O014 congelado.
- seeds `[777, 999]`.
- NDCG@20 como métrica primaria.
- ramas Stale / Online / Full Retrain.

No existe retuning ni cambio de población experimental respecto de V3.

## Cambios de endurecimiento

### 1. Interpretación H1/H2

Se eliminaron las clasificaciones automáticas:

```text
SOPORTE FUERTE
SOPORTE CONSISTENTE
SIN SOPORTE DIRECCIONAL
```

El script sólo registra resultados descriptivos:

- deltas pareados;
- cantidad de puntos positivos;
- medias;
- desviaciones;
- resultados por seed;
- resultados por punto temporal.

La interpretación de H1-H3 se realiza después de la auditoría de resultados.

### 2. Fingerprint reproducible ampliado

El `protocol_hash` incorpora ahora también:

```text
Python version
platform
Cornac version
NumPy version
PyTorch version
SciPy version
SHA256 de Dataset.build
SHA256 de ranking_eval
SHA256 de wrappers/cores IBPR y Online
SHA256 del propio script
SHA256 de los datos
```

Un resume con entorno o código incompatible debe ser rechazado.

### 3. Integridad de interacciones

Se añaden asserts para garantizar:

```text
len(rows) == número de pares (u,i) únicos
```

en el dataset positivo, base y chunks warm.

Después de cada `Dataset.build` también se exige:

```text
dataset.csr_matrix.nnz == expected_rows
len(dataset.uir_tuple[0]) == expected_rows
```

Así se detecta cualquier deduplicación silenciosa que altere el experimento.

### 4. Logging completo

En una ejecución real, el `.txt` final contiene también el plan completo:

- fingerprint;
- corte 60/40;
- warm-start fraction;
- chunks;
- población PRIMARY;
- configuraciones;
- seeds;
- entorno.

### 5. H2

La medida primaria no cambia:

```text
partial_fit_recent()
vs
IBPR.fit()
```

Es decir, costo puro de actualización/entrenamiento del modelo.

Además se registra una medida suplementaria:

```text
Online:
history_dataset_build + partial_fit_recent

Full:
full_dataset_build + IBPR.fit
```

La construcción de los datasets de test/evaluación se mide separadamente y no se incluye en ninguna de esas dos ramas.

### 6. Terminología PRIMARY

`updated_users` fue renombrado conceptualmente a:

```text
users_exposed_to_update
```

porque pertenecer a la población PRIMARY significa que el usuario recibió al menos una nueva interacción de actualización, no que se haya demostrado previamente que su vector `U` cambió numéricamente.

### 7. PLAN-ONLY

La salida ya no afirma que no se entrenó ningún modelo de ningún tipo.

La formulación correcta indica que:

- no se entrenó ningún modelo experimental final de MovieLens;
- sólo se ejecutaron smoke tests sintéticos del contrato de implementación.

## Auditoría estática realizada

```text
Sintaxis Python en memoria                          OK
STEP_FIELDS                                        112 / 112 únicos
TRIAL_FIELDS                                       102 / 102 únicos
SUMMARY_FIELDS                                     228 / 228 únicos
SUMMARY sintético                                  228 / 228 exactos
Campos faltantes en SUMMARY                        0
Campos extra en SUMMARY                            0
Clasificaciones inferenciales automáticas          eliminadas
Referencias obsoletas n_updated_users               0
```

## Archivo

```text
final_h1_h3_online_ibpr_mejorado_v3_1.py
SHA256: 7e78903afc4ff1f92675903d79306518045901cb1f98eae1cba38f75130dfb3c
Líneas: 2178
```

## Próximo paso

Antes de ejecutar el experimento definitivo:

```powershell
python tests/online_ibpr_mejorado/final_h1_h3_online_ibpr_mejorado_v3_1.py --plan-only
```

El nuevo `protocol_hash` debe ser diferente al de V3 porque se amplió el fingerprint y cambió el script.

No reutilizar CSV/resultados de V3 con V3.1.
