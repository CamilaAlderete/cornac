# H1-H3 — Validación definitiva de OnlineIBPRMejorado en MovieLens 1M

## 1. Objetivo

Este documento registra el cierre experimental definitivo de las hipótesis H1, H2 y H3 de **OnlineIBPRMejorado** sobre **MovieLens 1M**, utilizando las configuraciones previamente congeladas y sin retuning posterior.

La comparación final incluye tres ramas:

```text
IBPR_STALE
OnlineIBPRMejorado
IBPR_FULL_RETRAIN
```

El objetivo general es evaluar si OnlineIBPRMejorado puede adaptar incrementalmente las preferencias de usuarios conocidos entre reentrenamientos completos de IBPR, reduciendo sustancialmente el costo computacional y manteniendo una calidad competitiva.

---

## 2. Hipótesis evaluadas

### H1 — Adaptación frente a modelo obsoleto

> OnlineIBPRMejorado mejora la calidad de recomendación respecto de un modelo IBPR que permanece sin actualizar.

Métrica principal:

```text
NDCG@20
```

Comparación:

```text
OnlineIBPRMejorado - IBPR_STALE
```

---

### H2 — Costo de actualización

> La actualización parcial de OnlineIBPRMejorado tiene un costo computacional sustancialmente inferior al reentrenamiento completo de IBPR.

Comparación principal:

```text
partial_fit_recent()
vs
IBPR.fit()
```

También se registró una medición suplementaria incluyendo preparación del dataset de entrenamiento:

```text
Online:
history Dataset.build
+
partial_fit_recent

Full Retrain:
full Dataset.build
+
IBPR.fit
```

Esta segunda medición no representa costo end-to-end completo del sistema.

---

### H3 — Calidad frente a Full Retrain

> OnlineIBPRMejorado mantiene una calidad competitiva frente al reentrenamiento completo de IBPR a un costo muy inferior.

Comparación:

```text
OnlineIBPRMejorado - IBPR_FULL_RETRAIN
```

No se realizó una prueba formal de equivalencia ni no-inferioridad.

---

## 3. Configuraciones congeladas

### IBPR base — R900

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

### OnlineIBPRMejorado — O014

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

Seeds finales:

```text
[777, 999]
```

Después de la ejecución definitiva:

```text
RETUNING PROHIBIDO
```

---

## 4. Dataset y protocolo

Dataset:

```text
MovieLens 1M
```

Conversión a feedback implícito:

```text
rating >= 3.0 -> interacción positiva 1.0
```

Total de interacciones positivas:

```text
836,478
```

### División temporal global

```text
Primer 60% cronológico  -> base
Último 40% cronológico  -> stream final
```

El corte se ajustó para no dividir registros con el mismo timestamp.

Resultado:

```text
Target base rows       : 501,886
Effective base rows    : 501,887
Effective base fraction: 60.000024%
```

Separación temporal:

```text
base max timestamp   < future min timestamp
974678223            < 974678227
```

---

## 5. Universo warm-start

OnlineIBPRMejorado actualmente sólo trabaja con usuarios e ítems ya conocidos por el modelo base.

Por ello, dentro del 40% final se conservan únicamente interacciones donde:

```text
user ∈ base_users
item ∈ base_items
```

Resultado:

```text
Future raw rows              : 334,591
Future warm-start rows       : 52,467
Warm-start fraction          : 15.6809%

Excluded unknown user only   : 281,748
Excluded unknown item only   : 132
Excluded unknown both        : 244
```

Este porcentaje bajo no constituye un error del experimento.

Representa una limitación explícita del alcance actual:

> H1-H3 evalúan adaptación warm-start y no cold-start.

---

## 6. Chunks prequentiales

El stream warm-start se dividió, después del filtrado, en cuatro chunks cronológicos aproximadamente iguales:

```text
chunk 1: 13,118
chunk 2: 13,118
chunk 3: 13,114
chunk 4: 13,117
```

Secuencia:

```text
base
  ↓
update chunk 1
  ↓
eval chunk 2
  ↓
update chunk 2
  ↓
eval chunk 3
  ↓
update chunk 3
  ↓
eval chunk 4
```

Los chunks tienen igual cantidad aproximada de eventos, no igual duración calendario.

El protocolo es therefore **event-based prequential**.

---

## 7. Poblaciones de evaluación

### PRIMARY

La población primaria incluye únicamente interacciones de usuarios que ya recibieron al menos una interacción de actualización antes de ser evaluados.

```text
update1 -> eval2
PRIMARY rows  : 5,300
PRIMARY users : 212

update2 -> eval3
PRIMARY rows  : 8,546
PRIMARY users : 315

update3 -> eval4
PRIMARY rows  : 9,143
PRIMARY users : 306
```

Las métricas PRIMARY determinan H1 y H3.

### ALL-WARM

También se evalúan todas las interacciones warm-start del chunk como diagnóstico complementario.

ALL-WARM no reemplaza la evaluación PRIMARY.

---

## 8. Integridad de la corrida

Timestamp:

```text
20260914_175307
```

Protocol version:

```text
final_h1_h3_v3_1_hardened_20260914
```

Protocol hash:

```text
05e4f07096b02ae1730278076b35c13a031179ff170a1d6eb902d633173fdf77
```

Data SHA256:

```text
29da5346c5bcf37dc927771d8ffd7ec3323dc7857ed4b0f6a45278b666954d3e
```

Entorno:

```text
Python   3.12.0
Cornac   2.3.5
NumPy    2.4.2
PyTorch  2.10.0+cpu
SciPy    1.17.1
```

Auditoría:

```text
Seeds                       2/2
Steps                       6/6
Trials                      2/2
Eval points                 {1,2,3}
Update chunks               {1,2,3}
Eval chunks                 {2,3,4}

Protocol hash único         OK
Data SHA256 único           OK

Stale U exacta al base      OK
Stale V exacta al base      OK

Online V exacta al base     OK
max |ΔV_online|             0.0

Online maps exactos         OK
Full maps exactos           OK

Full V distinta al base     esperado

Trials reconstruidos desde steps    0 discrepancias
Summary reconstruido desde steps    228/228 campos correctos
```

La corrida se considera experimentalmente válida.

---

# 9. Resultado H1

## NDCG@20 agregado

```text
IBPR_STALE             0.079471
OnlineIBPRMejorado     0.078064

Online - Stale        -0.001407 ± 0.011173
Puntos positivos       4 / 6
```

Cambio relativo aproximado:

```text
-1.77%
```

Medias por seed:

```text
seed 777   ΔH1 = -0.000215
seed 999   ΔH1 = -0.002599
```

Por tanto:

> **H1 no queda respaldada en el agregado definitivo de MovieLens 1M.**

Sin embargo, el comportamiento temporal no es uniforme.

### Evolución temporal

```text
eval point 1
Online - Stale = +0.004253

eval point 2
Online - Stale = +0.006557

eval point 3
Online - Stale = -0.015030
```

Interpretación:

```text
inicio del stream     -> adaptación positiva
zona intermedia       -> adaptación positiva
último horizonte      -> degradación fuerte del top-k
```

El resultado muestra que OnlineIBPRMejorado puede adaptarse favorablemente durante etapas iniciales, pero la ventaja no se mantiene durante todo el stream.

---

## 10. Métricas secundarias de H1

Promedio Online - Stale:

```text
AUC            +0.006116   positivos 6/6
MAP            +0.000642   positivos 4/6
NDCG@20        -0.001407   positivos 4/6
Precision@20   -0.003606   positivos 3/6
Recall@20      +0.004211   positivos 4/6
```

El comportamiento de AUC es especialmente relevante:

```text
AUC Online > Stale en 6/6 puntos
```

Esto indica que la degradación final afecta principalmente la calidad del ranking top-k y no implica necesariamente un deterioro uniforme de toda la capacidad discriminativa del modelo.

No se debe utilizar este resultado para contradecir NDCG@20, porque NDCG@20 fue la métrica primaria preregistrada para H1.

---

# 11. Resultado H2

## Tiempo puro de actualización

Promedio total por seed:

```text
Online partial update       : 0.7610 s
Full Retrain                : 1985.6992 s
```

Speedup:

```text
Full / Online = 2610.25x
```

Fracción de costo:

```text
Online / Full = 0.000383
              ≈ 0.0383%
```

Resultados por seed:

```text
seed 777   2599.46x
seed 999   2621.03x
```

Por tanto:

> **H2 queda respaldada por la evidencia experimental descriptiva.**

OnlineIBPRMejorado reduce de manera muy marcada el tiempo de actualización respecto de ejecutar un Full Retrain completo de IBPR R900.

---

## 12. H2 incluyendo preparación del dataset

Medición suplementaria:

```text
Online:
history build + partial update

Full:
full dataset build + full fit
```

Promedios:

```text
Online                     ≈ 3.30 s
Full                       ≈ 1988.32 s

Speedup                    ≈ 602.32x
Online / Full              ≈ 0.001666
                           ≈ 0.1666%
```

Incluso incluyendo la preparación necesaria para el entrenamiento, la actualización incremental continúa siendo sustancialmente más barata.

Esta medición no incluye evaluación, serving, I/O ni construcción/reconstrucción de índices ANN.

---

# 13. Resultado H3

Promedio:

```text
OnlineIBPRMejorado     0.078064
IBPR_FULL_RETRAIN      0.083283

Online - Full         -0.005218 ± 0.008821
Online > Full          2 / 6
```

Diferencia relativa aproximada:

```text
-6.27%
```

Evolución temporal:

```text
eval point 1
Online - Full = -0.000505

eval point 2
Online - Full = +0.000878

eval point 3
Online - Full = -0.016029
```

Conclusión:

> **H3 no queda respaldada como afirmación global de calidad competitiva sostenida durante todo el stream.**

Sin embargo:

- en el primer punto la diferencia es pequeña;
- en el segundo punto Online incluso supera ligeramente a Full Retrain;
- en el tercero aparece una degradación marcada.

Por ello, la formulación correcta no es que OnlineIBPRMejorado sea globalmente equivalente a Full Retrain.

La evidencia indica que puede aproximarse a su calidad durante ciertos períodos, pero no mantiene esa relación de manera sostenida.

---

## 14. Adaptation recovery

El promedio calculado fue:

```text
0.6849
```

con:

```text
5 / 6 puntos válidos
std ≈ 1.4615
```

Esta métrica presenta una variabilidad muy alta.

No debe utilizarse como afirmación principal del tipo:

```text
"Online recupera el 68.5% de la mejora del Full Retrain"
```

porque esa interpretación sería engañosa.

Debe conservarse únicamente como diagnóstico descriptivo secundario.

---

# 15. ALL-WARM

Resultado complementario:

```text
Online - Stale NDCG@20
= -0.001856 ± 0.008583

Puntos positivos = 4/6
```

El patrón temporal es similar al de PRIMARY:

```text
primer período      positivo
segundo período     positivo
tercer período      negativo
```

Por tanto:

> La degradación del último período no parece ser un artefacto causado exclusivamente por la construcción de la población PRIMARY.

---

# 16. Conclusión H1-H3

Estado definitivo:

```text
H1
OnlineIBPRMejorado > IBPR_STALE
en calidad sostenida
-----------------------------------
NO RESPALDADA globalmente

Hallazgo:
adaptación positiva temprana,
seguida de degradación en el
último horizonte.


H2
Online partial update
mucho más barato que Full Retrain
-----------------------------------
RESPALDADA

~2610x speedup puro de entrenamiento
~602x incluyendo preparación
del dataset de entrenamiento.


H3
Online mantiene calidad competitiva
frente a Full Retrain durante
todo el stream
-----------------------------------
NO RESPALDADA globalmente

Hallazgo:
calidad cercana o superior en puntos
tempranos/intermedios, pero degradación
marcada al final.
```

---

# 17. Qué NO puede concluirse

No debe afirmarse:

```text
OnlineIBPRMejorado siempre mejora IBPR.
OnlineIBPRMejorado reemplaza Full Retrain.
OnlineIBPRMejorado es estadísticamente equivalente a Full Retrain.
OnlineIBPRMejorado resuelve cold-start.
68.5% representa una recuperación estable de Full Retrain.
```

Tampoco debe realizarse retuning de O014 utilizando estos resultados.

H1-H3 constituyen el resultado final congelado para MovieLens 1M.

---

# 18. Qué sí puede concluirse

La evidencia soporta afirmar que:

1. OnlineIBPRMejorado puede producir adaptación positiva frente a un modelo stale durante ciertos períodos del stream.
2. Esa mejora no se mantiene de manera sostenida bajo el protocolo final.
3. La degradación ocurre especialmente en el último horizonte temporal.
4. La actualización parcial tiene un costo computacional extremadamente inferior al Full Retrain.
5. Durante algunos períodos, Online logra calidad próxima o incluso ligeramente superior al Full Retrain.
6. La combinación de bajo costo y degradación acumulativa motiva estudiar una política híbrida donde las actualizaciones online ocurran entre reentrenamientos completos.

El punto 6 constituye una motivación para investigación posterior y no una validación de H5.

---

# 19. Pregunta abierta principal

El resultado definitivo transforma la pregunta experimental.

Ya no es solamente:

> ¿Puede OnlineIBPRMejorado adaptarse?

La evidencia indica que sí puede hacerlo durante ciertos períodos.

La pregunta siguiente es:

> ¿Durante cuánto tiempo puede acumular actualizaciones incrementales antes de que la calidad top-k se degrade suficientemente como para justificar un nuevo Full Retrain?

Esta pregunta debe abordarse posteriormente mediante experimentos focalizados y sin modificar retrospectivamente H1-H3.

---

# 20. Próximos pasos

Hoja de ruta posterior:

```text
H1-H3 definitivo                         COMPLETADO

H4 — preservación de V / índice          SIGUIENTE

Ablaciones focalizadas                   PENDIENTE
- update_V
- loss
- n_epochs
- max_steps
- comportamiento temporal
- posible forgetting/recency

MovieLens 10M frozen-config              PENDIENTE

H5 — ciclo híbrido                       CONDICIONAL
```

H4 debe mantenerse separado de H1-H3.

Las ablaciones posteriores son explicativas y no reemplazan estos resultados.

---

# 21. Archivos oficiales de la corrida

Timestamp:

```text
20260914_175307
```

Archivos:

```text
final_h1_h3_online_ibpr_mejorado_20260914_175307.txt
final_h1_h3_online_ibpr_mejorado_steps_20260914_175307.csv
final_h1_h3_online_ibpr_mejorado_trials_20260914_175307.csv
final_h1_h3_online_ibpr_mejorado_summary_20260914_175307.csv
```

Estos archivos deben conservarse sin modificación como evidencia experimental definitiva de H1-H3 sobre MovieLens 1M.

---

# 22. Estado final

```text
H1-H3 MovieLens 1M
-------------------
EJECUTADO       ✅
AUDITADO        ✅
CONGELADO       ✅
RETUNING        PROHIBIDO

H1              no respaldada globalmente
H2              respaldada
H3              no respaldada globalmente
```
