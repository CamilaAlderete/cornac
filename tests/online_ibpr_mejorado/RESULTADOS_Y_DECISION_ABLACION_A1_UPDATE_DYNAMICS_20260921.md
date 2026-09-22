# RESULTADOS Y DECISIÓN — ABLACIÓN A1: DINÁMICA DE ACTUALIZACIÓN

**Experimento:** A1 v1.2 — OnlineIBPRMejorado  
**Timestamp:** 20260921_172730  
**Protocol hash:** `8090d3adb71088aac68ea46cdf63e30c5ae65ab3784e3d27ea2ed32ea5b0fef3`  
**Dataset SHA256:** `29da5346c5bcf37dc927771d8ffd7ec3323dc7857ed4b0f6a45278b666954d3e`

## 1. Objetivo

A1 estudia si el comportamiento observado en H1-H3 depende de la dinámica operacional con la que se aplican las nuevas interacciones a `OnlineIBPRMejorado`.

No es un nuevo HPO. R900 y O014 permanecen congelados.

Ramas:

- **S — SEQUENTIAL_ORIGINAL:** flujo online real y persistente.
- **H — RETROSPECTIVE_FIXED_HISTORY:** replay desde base, C1..Ci en llamadas separadas, usando el history final del punto en todas las llamadas.
- **C — RESET_CUMULATIVE:** replay desde base, todas las interacciones C1..Ci en una única llamada.
- **R — RESET_CURRENT:** replay desde base usando sólo el chunk que acaba de llegar.

## 2. Auditoría de integridad

Los archivos contienen exactamente:

```text
2 seeds × 3 puntos × 4 ramas = 24 filas de steps
8 filas de trials
1 fila de summary
```

No se detectaron duplicados en `(seed, eval_point, branch)`.

Verificaciones reconstruidas independientemente:

```text
Deltas de métricas desde steps                  PASS
Trials reconstruidos desde steps               PASS
Summary reconstruido desde steps/trials        PASS
Discrepancias aritméticas                       0
V exacta respecto de V_base                     24/24
max_abs_diff(V)                                 0
uid_map exacto                                  PASS
iid_map exacto                                  PASS
Sequential reproduce H1-H3                     PASS
Stale reproduce H1-H3                          PASS
Punto 1: U idéntica S/H/C/R                    PASS
```

Por tanto, A1 se considera una ejecución válida del protocolo congelado.

## 3. NDCG@20 — medias entre seeds

| Punto | Stale | S Sequential | H Fixed History | C Cumulative | R Current | Full Ref |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.072362 | 0.076615 | 0.076615 | 0.076615 | 0.076615 | 0.077120 |
| 2 | 0.071877 | 0.078434 | 0.080619 | 0.076262 | 0.076229 | 0.077555 |
| 3 | 0.094174 | 0.079144 | 0.080228 | 0.075088 | 0.087982 | 0.095173 |

Promedio descriptivo de los tres puntos:

| Rama | NDCG@20 | vs Stale | vs Full Ref |
|---|---:|---:|---:|
| S Sequential | 0.078064 | -0.001407 | -0.005218 |
| H Fixed History | 0.079154 | -0.000317 | -0.004129 |
| C Cumulative | 0.075988 | -0.003483 | -0.007294 |
| R Current | 0.080275 | +0.000804 | -0.003008 |

Estas medias son descriptivas; las poblaciones PRIMARY cambian por punto y no deben sustituir el análisis pareado dentro de cada punto.

## 4. Contraste S vs H — history utilizado para negative sampling

Contraste global:

```text
S - H = -0.001089 NDCG@20
```

Punto 3:

```text
S - H = -0.001084
```

Sin embargo, en el punto 3 el signo no fue uniforme por seed:

```text
seed 777: S > H
seed 999: H > S
```

Por tanto, A1 **no aporta evidencia consistente para atribuir la caída del punto 3 al history utilizado en las llamadas anteriores**.

La diferencia entre S y H es pequeña frente a otros contrastes y cambia de signo entre seeds.

## 5. Contraste H vs C — múltiples llamadas frente a una llamada acumulada

Resultado medio:

```text
H - C = +0.003165
```

Punto 3:

```text
H - C = +0.005140
```

La dirección fue consistente en los dos seeds en puntos 2 y 3: la rama segmentada H obtuvo mayor NDCG@20 que C.

En el punto 3:

```text
H = 0.080228
C = 0.075088
diferencia = +0.005140
```

Esto equivale aproximadamente a un 6.8% respecto del NDCG de C.

Interpretación permitida:

> Con la configuración O014 y el stream evaluado, procesar el conjunto acumulado de interacciones mediante llamadas separadas produjo mejor calidad que procesar ese mismo conjunto acumulado mediante una sola llamada.

No debe atribuirse la diferencia a una causa única. H y C difieren también en reinicios de Adam, normalizaciones intermedias, shuffling/batching y política de seeds.

Esta observación es especialmente relevante porque S/H/C realizaron el mismo número total de optimizer steps en cada punto:

```text
punto 1: 39
punto 2: 78
punto 3: 117
```

Por tanto, la diferencia no se explica simplemente porque una rama haya realizado más pasos.

## 6. Contraste S vs C

Resultado medio:

```text
S - C = +0.002076
```

Punto 3:

```text
S - C = +0.004056
```

S fue superior a C en ambos seeds en los puntos 2 y 3.

Por tanto, **agrupar todas las interacciones acumuladas en una única llamada no corrigió el comportamiento tardío y, en este experimento, resultó peor que la actualización secuencial original**.

Este resultado es directamente relevante para la pregunta sobre si recibir pocas o muchas interacciones por actualización puede producir comportamientos distintos: en este protocolo, la misma cantidad acumulada de evidencia y el mismo número de optimizer steps no produjeron el mismo resultado cuando cambió la granularidad operacional.

No se generaliza todavía a cualquier tamaño de batch o dataset.

## 7. RESET_CURRENT — uso exclusivo del chunk reciente

Sobre PRIMARY original, en el punto 3:

```text
S = 0.079144
R = 0.087982
R - S = +0.008837
```

Sobre `CURRENT-USER matched`, que es la comparación más interpretable:

```text
S = 0.078179
R = 0.091511
R - S = +0.013333
```

La mejora de R frente a S en CURRENT-USER matched fue consistente en ambos seeds:

```text
seed 777: R - S = +0.012064
seed 999: R - S = +0.014602
```

En el punto 3, esta diferencia representa aproximadamente un 17% respecto del NDCG de S en esa misma población.

Las métricas secundarias de ranking también se movieron a favor de R frente a S en CURRENT-USER matched en el punto 3:

```text
MAP          : +0.003833
NDCG@20      : +0.013333
Precision@20 : +0.008730
Recall@20    : +0.008105
```

AUC fue ligeramente menor:

```text
AUC: -0.001654
```

### Importante

R **no soluciona completamente** la caída.

En CURRENT-USER matched del punto 3:

```text
R NDCG@20     = 0.091511
Stale NDCG@20 = 0.099031
R - Stale     = -0.007519
```

Por tanto, el resultado correcto no es:

> "reiniciar con el chunk actual arregla el modelo".

La interpretación correcta es:

> Descartar la adaptación acumulada previa y adaptar desde el modelo base únicamente con el chunk reciente mitigó de forma importante la pérdida observada frente a Sequential en el tercer punto, pero no recuperó el nivel de Stale en esa población.

Esto proporciona evidencia descriptiva de que la acumulación/dinámica previa de actualizaciones puede contribuir al comportamiento tardío, sin establecer una causa única.

## 8. Movimiento de U

En el punto 3, las medias entre seeds fueron aproximadamente:

```text
C RESET_CUMULATIVE             ||U-Ubase||F = 8.840
S SEQUENTIAL                   ||U-Ubase||F = 7.471
H FIXED_HISTORY                ||U-Ubase||F = 7.458
R RESET_CURRENT                ||U-Ubase||F = 4.637
```

C fue la rama que más desplazó U y también la que presentó menor NDCG@20 en el punto 3.

R desplazó menos U y obtuvo el mayor NDCG entre las cuatro ramas en ese punto.

Esto es únicamente una asociación descriptiva; con este diseño no puede afirmarse que un mayor desplazamiento de U cause menor calidad.

## 9. Nota numérica sobre usuarios no presentes en el update

Aunque `update_V=False`, `normalize=True` normaliza U al finalizar una llamada.

Por eso se observan cambios numéricos mínimos en filas de usuarios no directamente presentes en los positivos recientes:

```text
max abs diff en usuarios no actualizados ≈ 1.2e-7
```

Esto explica que `n_user_rows_bitwise_changed` pueda ser mayor que el número de usuarios directamente incluidos en el update.

La magnitud es extremadamente pequeña y no viola el invariante central de H4, que se refiere a V, pero debe documentarse para no describir literalmente el procedimiento como "sólo ciertas filas de U cambian bit a bit".

## 10. Respuesta a la pregunta experimental de A1

A1 permite concluir descriptivamente que:

1. **La forma de aplicar las nuevas interacciones importa.**
2. Procesar todo lo acumulado en una única llamada no fue equivalente a procesarlo mediante actualizaciones sucesivas, aun con el mismo número total de optimizer steps.
3. La rama acumulativa de una sola llamada fue consistentemente peor que la replay segmentada en puntos 2 y 3.
4. En el punto 3, utilizar únicamente el chunk reciente desde el modelo base produjo una mejora importante frente al estado Sequential en los usuarios comparables.
5. Esto sugiere que la dinámica/acumulación de actualizaciones merece estudiarse directamente mediante granularidad/frecuencia.
6. A1 no demuestra que una cantidad grande de interacciones deteriore universalmente el modelo.
7. A1 tampoco identifica una causa única como Adam, normalización, seeds o negative sampling.

## 11. Decisión

**A1 se considera cerrada.**

No se modifica O014 y no se selecciona ninguna de las ramas A1 como nueva configuración principal.

Los resultados justifican ejecutar **A2 — granularidad/frecuencia de partial updates**, manteniendo:

```text
- mismo R900/O014;
- mismo stream total;
- mismo orden cronológico;
- misma cantidad total de interacciones;
- mismos puntos de evaluación;
- variando únicamente, hasta donde permita la implementación, cómo se divide el stream en llamadas incrementales.
```

A2 debe diseñarse antes de ejecutarse para evitar convertir esta evidencia en un nuevo HPO.

## 12. Hashes de archivos auditados

```text
log:
2e12064bec64dbb1eed557798410c6f9a23e843b17800ac3409cb411d0b825a9

steps:
e6cf3b8e1e288caf706f0f9b05b5c8fca3b3da8d881bdac872b8ed24d64aaf7b

trials:
99aaf26c286225de7cbd5f7defade7e90f983dfec556dce131ca383d195ad43a

summary:
371677b834416070c5151afdbad1261e28eb0b5d2839f7ae9235d2c1793f9d48
```
