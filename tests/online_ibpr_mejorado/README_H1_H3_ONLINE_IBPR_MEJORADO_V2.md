# H1-H3 — Validación definitiva de OnlineIBPRMejorado en MovieLens 1M

## 1. Propósito de este documento

Este README registra el cierre experimental definitivo de H1, H2 y H3 para **OnlineIBPRMejorado** sobre **MovieLens 1M**, utilizando configuraciones previamente congeladas y sin retuning posterior.

La comparación final incluye tres ramas:

```text
IBPR_STALE
OnlineIBPRMejorado
IBPR_FULL_RETRAIN
```

Este documento debe leerse como el registro oficial de:

- protocolo;
- configuraciones congeladas;
- integridad de la corrida;
- resultados;
- interpretación;
- limitaciones;
- afirmaciones permitidas;
- afirmaciones que no deben hacerse.

---

# 2. Hipótesis evaluadas

## H1 — Adaptación frente a modelo stale

Pregunta:

> ¿OnlineIBPRMejorado mejora la calidad de recomendación respecto de un IBPR que permanece sin actualizar?

Comparación principal:

```text
OnlineIBPRMejorado - IBPR_STALE
```

Métrica primaria fijada previamente en el protocolo:

```text
NDCG@20
```

---

## H2 — Costo de actualización

Pregunta:

> ¿La actualización parcial de OnlineIBPRMejorado tiene un costo computacional sustancialmente inferior al reentrenamiento completo de IBPR?

Comparación primaria:

```text
partial_fit_recent()
vs
IBPR.fit()
```

También se registra una medición suplementaria:

```text
Online:
history Dataset.build
+
partial_fit_recent()

Full Retrain:
full Dataset.build
+
IBPR.fit()
```

Esta medición suplementaria **no representa costo end-to-end completo del sistema**.

---

## H3 — Calidad frente a Full Retrain

Pregunta:

> ¿OnlineIBPRMejorado mantiene una calidad competitiva frente al Full Retrain a un costo muy inferior?

Comparación principal:

```text
OnlineIBPRMejorado - IBPR_FULL_RETRAIN
```

Importante:

- no se fijó un margen formal de no-inferioridad;
- no se realizó prueba de equivalencia;
- no se realizó prueba formal de no-inferioridad.

Por tanto, H3 debe interpretarse descriptivamente y **no** como una prueba formal de equivalencia o competitividad estadística.

---

# 3. Configuraciones congeladas

## 3.1 IBPR base — R900

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

---

## 3.2 OnlineIBPRMejorado — O014

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

Después de esta corrida:

```text
RETUNING PROHIBIDO
```

Los resultados de H1-H3 no deben utilizarse para volver a optimizar R900 u O014.

---

# 4. Dataset y conversión a feedback implícito

Dataset:

```text
MovieLens 1M
```

Conversión:

```text
rating >= 3.0 -> interacción positiva 1.0
rating < 3.0  -> no observado
```

Total de interacciones positivas:

```text
836,478
```

---

# 5. División temporal global

Se ordenaron globalmente las interacciones positivas por timestamp.

Separación conceptual:

```text
0% ---------------------------- 60% ---------------------------- 100%
       base/desarrollo                          stream final
```

Resultado efectivo:

```text
Target base rows        : 501,886
Effective base rows     : 501,887
Effective base fraction : 60.000024%
Tie rows moved to base  : 1
```

Separación temporal:

```text
base max timestamp   = 974678223
future min timestamp = 974678227
```

Por tanto:

```text
base < future
```

se cumple estrictamente.

---

# 6. Advertencia metodológica sobre el tramo final 40%

El tramo cronológicamente posterior no fue utilizado para seleccionar R900 ni O014.

Sin embargo, **no debe describirse como un “holdout completamente virgen”**.

Durante etapas previas del proyecto existieron pilotos y, antes de la ejecución final, también se inspeccionaron características estructurales del stream mediante `--plan-only`, como:

- fracción warm-start;
- tamaño de chunks;
- número de usuarios;
- cobertura PRIMARY.

Por tanto, la formulación correcta es:

> **evaluación con configuración completamente congelada y sin utilizar estos resultados para volver a ajustar hiperparámetros.**

La validez metodológica se apoya en:

- congelación previa de R900;
- congelación previa de O014;
- seeds finales reservadas;
- separación temporal explícita;
- ausencia de retuning posterior;
- evaluación multi-seed;
- posterior validación en MovieLens 10M con configuración congelada.

---

# 7. Universo warm-start

OnlineIBPRMejorado, en su alcance actual, trabaja únicamente con usuarios e ítems conocidos por el modelo base.

Dentro del 40% final se conservan únicamente interacciones que cumplen:

```text
user ∈ base_users
item ∈ base_items
```

Resultado:

```text
Future raw rows             : 334,591
Future warm rows            : 52,467
Warm-start fraction         : 15.6809%

Excluded unknown user only  : 281,748
Excluded unknown item only  : 132
Excluded unknown both       : 244
```

Este porcentaje bajo no representa un error.

Representa una limitación del alcance actual:

> H1-H3 evalúan adaptación warm-start y no cold-start.

---

# 8. Chunks prequentiales

Después de aplicar el filtro warm-start, el stream válido se divide en cuatro chunks cronológicos aproximadamente iguales:

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

Los chunks tienen aproximadamente igual cantidad de eventos, no igual duración calendario.

Por tanto, el protocolo es:

> **prequential basado en eventos**

y no basado en ventanas temporales de igual duración.

---

# 9. Poblaciones de evaluación

## 9.1 PRIMARY

PRIMARY contiene únicamente interacciones de usuarios que ya estuvieron expuestos a al menos una interacción de actualización previa.

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

PRIMARY determina H1 y H3.

---

## 9.2 ALL-WARM

Como diagnóstico complementario también se evalúan todas las interacciones warm-start del chunk de evaluación.

ALL-WARM no reemplaza PRIMARY.

---

# 10. Definición exacta de las tres ramas

## IBPR_STALE

```text
U = U_base
V = V_base
sin actualización posterior
```

Se actualiza únicamente el historial usado para excluir ítems observados durante evaluación.

---

## OnlineIBPRMejorado

Parte del mismo IBPR base R900.

En cada actualización:

```text
actualiza U
mantiene V fija
usa sólo el chunk reciente como recent_pairs
usa el historial acumulado para negative sampling válido
```

---

## IBPR_FULL_RETRAIN

Importante:

Full Retrain **no** utiliza las 334,591 interacciones raw futuras ni incorpora cold-start.

Se reentrena desde cero con:

```text
base
+
stream warm-start acumulado hasta el punto actual
```

Es decir, Full Retrain trabaja sobre el mismo universo warm-start fijo utilizado para la comparación controlada con Online.

Por tanto, H2 y H3 comparan estrategias de actualización dentro del mismo alcance warm-start.

---

# 11. Integridad de la corrida definitiva

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

---

# 12. Auditoría de integridad

La corrida final produjo:

```text
Seeds             2/2
Steps             6/6
Trials            2/2
Eval points       {1,2,3}
Update chunks     {1,2,3}
Eval chunks       {2,3,4}
```

Invariantes:

```text
Protocol hash único              OK
Data SHA256 único                OK

Stale U exacta al base           OK
Stale V exacta al base           OK

Online V exacta al base          OK
max |ΔV_online|                  0.0

Online maps exactos              OK
Full maps exactos                OK

Full V distinta al base          esperado
```

Reconstrucción independiente:

```text
Trials reconstruidos desde steps     0 discrepancias
Summary reconstruido desde steps     228/228 campos correctos
```

La corrida se considera experimentalmente válida.

---

# 13. Nota estadística sobre los 6 puntos

Los seis puntos finales son:

```text
2 seeds
×
3 puntos temporales por seed
```

No deben interpretarse como seis réplicas estadísticas independientes.

Los tres puntos de una misma seed pertenecen al mismo stream acumulativo y están temporalmente relacionados.

Por tanto:

- medias;
- desviaciones estándar;
- conteos positivos;

se interpretan **descriptivamente**.

Los valores expresados como:

```text
media ± std
```

no representan intervalos de confianza ni evidencia inferencial formal.

---

# 14. Resultado H1

## 14.1 NDCG@20 agregado

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

Interpretación:

> **La evidencia final no respalda la dirección global propuesta por H1 bajo este protocolo.**

No debe decirse que H1 fue “estadísticamente rechazada”.

---

# 15. Evolución temporal de H1

```text
eval point 1
Online - Stale = +0.004253

eval point 2
Online - Stale = +0.006557

eval point 3
Online - Stale = -0.015030
```

El patrón observado es:

```text
primer horizonte       positivo
segundo horizonte      positivo
tercer horizonte       negativo
```

Esto muestra que la ventaja observada en los primeros puntos no se mantiene en el último horizonte.

No debe afirmarse todavía que la **acumulación de updates cause** esa degradación.

Las causas posibles deberán estudiarse posteriormente mediante ablaciones.

---

# 16. Métricas secundarias de H1

Promedio Online - Stale:

```text
AUC            +0.006116   positivos 6/6
MAP            +0.000642   positivos 4/6
NDCG@20        -0.001407   positivos 4/6
Precision@20   -0.003606   positivos 3/6
Recall@20      +0.004211   positivos 4/6
```

AUC favorece Online en los 6 puntos, mientras que las métricas centradas en los primeros puestos muestran un comportamiento menos favorable, especialmente al final del stream.

Interpretación prudente:

> Las métricas responden de manera distinta a la adaptación. El efecto observado depende de la región del ranking y de la métrica empleada.

No debe inferirse todavía una causa concreta.

NDCG@20 continúa siendo la métrica primaria para H1.

---

# 17. Resultado H2

## 17.1 Tiempo puro de actualización

Promedio total por seed:

```text
Online partial update   : 0.7610 s
Full Retrain            : 1985.6992 s
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

Interpretación:

> **H2 queda fuertemente respaldada descriptivamente en esta corrida experimental.**

La actualización parcial es sustancialmente más barata que el Full Retrain.

---

# 18. H2 incluyendo preparación del dataset de entrenamiento

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

Esta medición no incluye:

- evaluación;
- serving;
- I/O;
- latencia de consulta;
- build/rebuild de índices ANN;
- otros overheads del sistema.

Por tanto, no debe llamarse “costo end-to-end total”.

---

# 19. Alcance de los speedups de H2

Los factores:

```text
2610.25x
602.32x
```

son resultados observados bajo este entorno concreto:

```text
Windows 11
Python 3.12
Cornac 2.3.5
PyTorch CPU
MovieLens 1M
R900
O014
```

No deben presentarse como constantes universales del algoritmo.

La conclusión generalizable dentro de esta evaluación es:

> OnlineIBPRMejorado tiene un costo de actualización marcadamente inferior al Full Retrain bajo las configuraciones y condiciones evaluadas.

---

# 20. Resultado H3

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

---

# 21. Interpretación correcta de H3

No se fijó previamente un margen formal que defina cuánto puede degradarse NDCG@20 para seguir considerando Online “competitivo”.

Tampoco se realizó una prueba formal de:

```text
equivalencia
no-inferioridad
```

Por tanto, no corresponde clasificar H3 simplemente como:

```text
respaldada
o
refutada
```

La interpretación correcta es:

> **H3 no puede confirmarse formalmente como calidad competitiva sostenida.**

La evidencia muestra:

- menor NDCG@20 medio para Online;
- diferencia pequeña en el primer punto;
- ligera ventaja Online en el segundo punto;
- brecha marcada en el tercer punto;
- costo computacional muy inferior para Online.

Sin un margen predefinido de no-inferioridad, estos resultados deben presentarse descriptivamente.

---

# 22. Adaptation recovery

Promedio:

```text
0.6849
```

Puntos válidos:

```text
5 / 6
```

Desviación:

```text
std ≈ 1.4615
```

La variabilidad es muy elevada.

Por tanto, no debe afirmarse:

```text
"Online recupera el 68.5% de la mejora del Full Retrain"
```

como conclusión general.

`adaptation_recovery` se conserva únicamente como diagnóstico secundario.

---

# 23. ALL-WARM

Resultado complementario:

```text
Online - Stale NDCG@20
= -0.001856 ± 0.008583

Puntos positivos = 4/6
```

El patrón temporal es consistente con PRIMARY:

```text
primer período      positivo
segundo período     positivo
tercer período      negativo
```

Por tanto:

> La caída observada en el último horizonte no parece explicarse exclusivamente por la construcción de PRIMARY.

Esto no identifica todavía la causa de dicha caída.

---

# 24. Estado final de H1-H3

```text
H1
Online > Stale de forma global
--------------------------------
LA EVIDENCIA FINAL NO RESPALDA
LA DIRECCIÓN PROPUESTA

Hallazgo:
mejoras tempranas seguidas por
una caída en el último horizonte.


H2
Online partial update
mucho más barato que Full Retrain
--------------------------------
RESPALDADA DESCRIPTIVAMENTE

~2610x en entrenamiento puro
~602x incluyendo preparación
del dataset de entrenamiento.


H3
Online mantiene calidad competitiva
frente a Full Retrain
--------------------------------
NO CONFIRMADA FORMALMENTE

No existe margen de no-inferioridad
ni prueba formal de equivalencia.

Hallazgo:
Online es próximo a Full en algunos
puntos, pero presenta una brecha
marcada en el último horizonte.
```

---

# 25. Qué NO puede concluirse

No debe afirmarse que:

```text
OnlineIBPRMejorado siempre mejora IBPR.

OnlineIBPRMejorado reemplaza Full Retrain.

OnlineIBPRMejorado es equivalente a Full Retrain.

OnlineIBPRMejorado es no-inferior a Full Retrain.

OnlineIBPRMejorado resuelve cold-start.

La acumulación de partial updates causó la caída final.

68.5% representa una recuperación estable de Full Retrain.

Los speedups observados son universales.

Los 6 puntos son réplicas estadísticas independientes.
```

Tampoco debe realizarse retuning de O014 usando estos resultados.

---

# 26. Qué SÍ puede concluirse

La evidencia permite afirmar que:

1. OnlineIBPRMejorado muestra mejoras frente a Stale en algunos puntos tempranos del stream.
2. Esa ventaja no se mantiene en el agregado global de NDCG@20.
3. En el último horizonte aparece una caída marcada en NDCG@20.
4. AUC muestra un comportamiento distinto y favorece Online en 6/6 puntos.
5. La actualización parcial tiene un costo muy inferior al Full Retrain bajo el entorno evaluado.
6. Online presenta calidad próxima a Full Retrain en algunos puntos, pero no existe evidencia formal suficiente para declarar equivalencia o no-inferioridad sostenida.
7. La preservación exacta de V durante Online se mantiene en todos los puntos de H1-H3.
8. Los resultados motivan estudiar cuándo y por qué aparece la caída tardía, sin modificar retrospectivamente H1-H3.

---

# 27. Pregunta experimental que emerge

Los resultados no permiten afirmar todavía que la causa de la caída sea la acumulación de updates.

La pregunta adecuada es:

> **¿Qué factores explican que la adaptación online muestre mejoras en los primeros horizontes y una degradación de NDCG@20 en el último horizonte?**

Posibles factores a estudiar posteriormente:

```text
número acumulado de actualizaciones
deriva temporal
cambio de distribución de usuarios
cambio de distribución de ítems
normalización repetida
n_epochs
max_steps
loss
update_V
recencia / forgetting
```

Estas hipótesis causales requieren experimentos específicos.

---

# 28. Relación con un posible sistema híbrido

Los resultados hacen razonable estudiar posteriormente una estrategia conceptual:

```text
Full IBPR
→ partial
→ partial
→ ...
→ Full IBPR
```

Sin embargo:

> H1-H3 no validan todavía H5 ni determinan una frecuencia óptima de Full Retrain.

La frecuencia de retraining no debe elegirse retrospectivamente utilizando estos tres puntos a conveniencia.

---

# 29. Próximos pasos

Hoja de ruta:

```text
H1-H3 definitivo                         COMPLETADO

H4 — preservación de V / índice          SIGUIENTE

Ablaciones focalizadas                   PENDIENTE

MovieLens 10M frozen-config              PENDIENTE

H5 — ciclo híbrido                       CONDICIONAL
```

Las ablaciones posteriores tendrán función explicativa.

No reemplazarán los resultados definitivos de H1-H3.

---

# 30. Archivos oficiales de la corrida

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

Estos archivos deben conservarse sin modificación como evidencia experimental de H1-H3 en MovieLens 1M.

---

# 31. Estado oficial

```text
H1-H3 MovieLens 1M
-------------------
EJECUTADO       ✅
AUDITADO        ✅
CONGELADO       ✅
RETUNING        PROHIBIDO

H1
evidencia final no respalda
la dirección global propuesta

H2
respaldada descriptivamente
en el entorno evaluado

H3
no confirmada formalmente
como calidad competitiva sostenida
```
