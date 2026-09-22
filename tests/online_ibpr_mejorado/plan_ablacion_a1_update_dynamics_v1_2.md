# Plan de ablación focalizada A1 — Dinámica operacional de OnlineIBPRMejorado

**Versión:** v1.2  
**Fecha:** 2026-09-21  
**Estado previo:** H1–H3 cerrados; H4 cerrado.  
**Regla:** R900 y O014 permanecen congelados. Esta etapa es diagnóstica, no HPO.

## 1. Motivación

En la evaluación final H1–H3, OnlineIBPRMejorado mostró `Online - Stale` positivo en los puntos 1 y 2, pero negativo en el punto 3. Como las poblaciones PRIMARY cambian entre puntos, esto se describe como una **caída relativa en el tercer punto**, no como prueba directa de deterioro temporal.

A1 busca descomponer qué aspectos operacionales de las actualizaciones repetidas están asociados con ese comportamiento.

## 2. Elementos congelados

```python
R900 = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}

O014 = {
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

También quedan congelados MovieLens 1M, la transformación `rating >= 3`, el mismo base temporal, los mismos cuatro chunks warm-start, seeds `[777, 999]`, poblaciones PRIMARY, métricas y referencias Full/Stale de H1–H3.

## 3. Ramas

### S — SEQUENTIAL_ORIGINAL

Política real ya evaluada en H1–H3:

```text
U0,V0
 -> update(C1, H1)
 -> evaluar C2
 -> update(C2, H2)
 -> evaluar C3
 -> update(C3, H3)
 -> evaluar C4
```

con:

```text
H1 = Base+C1
H2 = Base+C1+C2
H3 = Base+C1+C2+C3
```

Debe reproducir las métricas Online finales auditadas.

### H — RETROSPECTIVE_FIXED_HISTORY

Control retrospectivo. Para cada punto se vuelve a `U_base,V_base`, se mantienen los chunks separados en llamadas sucesivas, pero todas las llamadas usan el history final ya observado en ese punto.

Ejemplo punto 3:

```text
Base
 -> update(C1, H3)
 -> update(C2, H3)
 -> update(C3, H3)
 -> evaluar C4
```

No usa C4 ni ningún dato posterior a la evaluación, por lo que no hay leakage hacia el test. Sin embargo, **no representa una política online desplegable**, porque retrospectivamente da a C1 información de C2/C3 para negative sampling.

Comparación principal:

```text
S vs H
```

Pregunta descriptiva:

> ¿Cambia el resultado cuando las llamadas anteriores excluyen como negativos todos los positivos que ya son conocidos en el punto evaluado?

No atribuir causalidad única.

### C — RESET_CUMULATIVE

Para cada punto se vuelve a `U_base,V_base` y se ejecuta una sola llamada con todos los positivos recientes acumulados:

```text
p1: update(C1, H1)
p2: update(C1+C2, H2)
p3: update(C1+C2+C3, H3)
```

Comparación principal:

```text
H vs C
```

Ambas ramas parten de la misma base, usan los mismos positivos acumulados, el mismo history final del punto y el mismo número total de optimizer steps. Lo que cambia es el paquete operacional de varias llamadas frente a una llamada: reinicios de Adam, normalizaciones intermedias, orden/batches y progresión de seeds.

No se atribuirá el contraste a una causa individual.

### R — RESET_CURRENT

Para cada punto se vuelve a `U_base,V_base` y se usa sólo el chunk actual:

```text
p1: update(C1, H1)
p2: update(C2, H2)
p3: update(C3, H3)
```

Su interpretación principal usa la población `CURRENT-USER matched`: filas PRIMARY cuyos usuarios aparecen en el chunk actual. Esto evita penalizar a R por evaluar usuarios que esa rama no actualizó.

Pregunta:

> ¿Qué ocurre si se descarta la adaptación previa y se adapta desde U_base usando sólo el chunk más reciente?

## 4. Política de seeds

Predefinida antes de observar resultados:

```text
S: seed, seed+1, ..., seed+point-1
H: seed, seed+1, ..., seed+point-1
C: una llamada con seed+point-1
R: una llamada con seed+point-1
```

La distinta progresión de seeds forma parte de la diferencia operacional entre una llamada y múltiples llamadas.

## 5. Control de carga

Con `batch_size=1024`, `n_epochs=3`, `max_steps=None`:

```text
punto 1: S=39,  H=39,  C=39
punto 2: S=78,  H=78,  C=78
punto 3: S=117, H=117, C=117
```

S/H/C llegan a cada punto con el mismo número de positivos acumulados y el mismo número total de optimizer steps. Esto controla la carga total, pero no vuelve equivalentes sus trayectorias.

R procesa sólo el chunk corriente: 39 steps por punto con los tamaños actuales.

## 6. Guard obligatorio del punto 1

En el punto 1, las cuatro ramas son exactamente:

```text
Base -> C1 con H1 y la misma seed
```

Por lo tanto:

```text
U_S == U_H == U_C == U_R bit a bit
```

Si falla, el experimento aborta.

## 7. Métricas

Principal:

```text
NDCG@20
```

Secundarias:

```text
Recall@20
Precision@20
MAP
AUC
```

Se reportan por seed, punto y rama. Las comparaciones principales son pareadas dentro del mismo punto:

```text
S - H
H - C
S - C
```

Para R:

```text
S - R
```

principalmente sobre `CURRENT-USER matched`.

Las medias sobre los tres puntos sólo son resúmenes descriptivos; no sustituyen el análisis por punto porque las poblaciones PRIMARY cambian.

## 8. Diagnósticos de U

Por rama/punto:

```text
||U - U_base||_F
||U_update_users - U_base_update_users||_F
cosine mean / median / p05
n_user_rows_bitwise_changed
max abs diff en usuarios no representados por los positivos de la rama
```

`V` debe permanecer exactamente idéntica a `V_base` en todos los casos.

## 9. Tiempos

A1 no intenta repetir H2. Sólo registra:

```text
last_call_time_s
state_construction_time_s
```

`state_construction_time_s` es el coste requerido para construir ese estado desde la base según la rama. No se suman los valores de distintos puntos como si fueran un coste de política común.

## 10. Guards de reproducibilidad

Antes de aceptar resultados:

```text
- entorno exacto congelado;
- fingerprints R900/O014 exactos;
- protocolo final H1-H3 exacto;
- dataset SHA exacto;
- reference H1-H3 steps semantic SHA exacto;
- Stale reproduce H1-H3;
- S reproduce Online H1-H3;
- V == V_base por shape + dtype + bytes;
- uid_map/iid_map exactos;
- punto 1 U idéntica en S/H/C/R;
- ningún eval chunk entra en entrenamiento antes de evaluarse.
```

## 11. Interpretación permitida

A1 es una ablación mecanística descriptiva con dos seeds. No se utilizará para seleccionar una nueva configuración ni para reabrir O014.

No afirmar:

```text
- causalidad única;
- que el punto 3 prueba deterioro temporal general;
- que Adam, normalización, seeds o negative sampling son individualmente la causa;
- que una rama diagnóstica es automáticamente una mejor política de producción.
```

## 12. Orden posterior

```text
A1 v1.2
  ↓
A2 granularidad/frecuencia de updates, sólo si A1 la justifica
  ↓
update_V=False vs True, sólo si sigue siendo necesaria
  ↓
MovieLens 10M con configuración principal congelada
```
