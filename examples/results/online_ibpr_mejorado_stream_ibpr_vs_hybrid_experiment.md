
## De qué trata el experimento

El experimento compara dos sistemas de recomendación en un escenario **secuencial por tandas**:

* **Modelo base:** `IBPR`, entrenado una sola vez con un bloque histórico inicial.
* **Modelo híbrido:** ese mismo `IBPR` como base, más `OnlineIBPRMejorado`, inicializado con `warm-start` desde los factores `U` y `V` del base, y luego actualizado incrementalmente con nuevas interacciones. 

El flujo del experimento es este:

1. se cargan interacciones positivas en orden cronológico;
2. se separan en `base`, `stream` y `holdout`;
3. el `stream` se divide en 4 chunks;
4. se entrena `IBPR` sobre `base`;
5. se inicializa `OnlineIBPRMejorado` con los parámetros del base;
6. en cada chunk, primero se evalúa el estado actual de ambos modelos sobre la siguiente tanda y luego solo el híbrido se actualiza con `partial_fit_recent(...)`;
7. al final se hace una evaluación sobre un holdout futuro.  

Además, el experimento mide dos cosas distintas:

* **calidad de recomendación**, usando `ranking_eval(...)`;
* **latencia de serving**, usando un pipeline de recuperación top-k. 

En el serving:

* el base usa `IBPR_indexed_topk`;
* el híbrido usa `IBPR_indexed_plus_online_rerank`.

Y aquí está el punto importante: en la versión corregida, el híbrido recupera candidatos usando el **vector de usuario actualizado del modelo online** (`online_model.U[user_idx]`) sobre el índice construido con los ítems del base, y luego reranquea esos candidatos con el propio modelo online. Eso deja el experimento alineado con la arquitectura híbrida que querías defender. 

---

## Qué dicen los resultados

### 1. El comportamiento inicial es el esperado

En el **Stage 1**, base e híbrido salen exactamente iguales en calidad:

* AUC `0.8413`
* MAP `0.0734`
* NDCG@20 `0.0939`
* Precision@20 `0.0929`
* Recall@20 `0.0386` 

Eso tiene sentido, porque en ese punto el híbrido todavía no recibió ninguna actualización parcial; solo fue warm-started desde el base.

---

### 2. Después de las primeras tandas, el híbrido empieza a diferenciarse

En el **Stage 2**, el híbrido mejora respecto al base:

* Recall@20 sube de `0.0185` a `0.0229`
* NDCG@20 sube de `0.0771` a `0.0829`
* Precision@20 sube de `0.0769` a `0.0823`
* AUC y MAP también mejoran un poco. 

Eso muestra que la actualización online sí está capturando información reciente y la está usando para ajustar el ranking.

---

### 3. La mejora no es uniforme en todos los chunks

En el **Stage 3**, el híbrido mejora AUC, MAP, NDCG y Precision, pero cae un poco en Recall@20 (`0.0354` frente a `0.0394`).
En el **Stage 4**, el híbrido mejora MAP y NDCG, empata en Precision, pero queda ligeramente por debajo en AUC y Recall@20. 

Eso quiere decir que la adaptación online **no mejora todas las métricas en todas las tandas**, lo cual es normal en un flujo temporal: cada bloque nuevo puede favorecer ciertos ajustes más que otros.

---

### 4. El resultado más importante está en el holdout final

En la evaluación final sobre datos futuros, el híbrido supera al base en todas las métricas principales:

* AUC: `0.8220` vs `0.8216`
* MAP: `0.0979` vs `0.0974`
* NDCG@20: `0.1367` vs `0.1341`
* Precision@20: `0.1298` vs `0.1286`
* Recall@20: `0.0528` vs `0.0508` 

El resumen lo expresa como una mejora final de `ΔRecall@20 = +0.0020`. 

Eso sugiere que, aunque la mejora no es enorme, **sí hay una ganancia acumulada y consistente** al permitir que el modelo se adapte por tandas.

---

### 5. El costo de actualización online es muy bajo

Entrenar el base tomó `146.6759 s`.
En cambio, la suma de todas las actualizaciones parciales del híbrido fue solo `0.2939 s`, con promedio de `0.0735 s` por chunk. 

Ese contraste es muy fuerte y apoya bien una de tus ideas principales:

> el modelo híbrido logra adaptarse a nuevas interacciones con un costo muchísimo menor que volver a entrenar el modelo base.

---

### 6. El overhead de latencia existe, pero es pequeño

En el holdout final:

* `IBPR_indexed_topk`: `0.7319 ms`
* `IBPR_indexed_plus_online_rerank`: `0.7516 ms` 

La diferencia es de solo `0.0197 ms`, que también aparece en el resumen. 

Y al mismo tiempo, el recall del pipeline de serving sube:

* base: `0.0492`
* híbrido: `0.0524` 

Entonces el mensaje es claro:

> el rerank añade una penalización temporal muy pequeña, pero devuelve una mejora pequeña en calidad.

---

## Observaciones metodológicas importantes

### 1. `ranking_eval(...)` y el benchmark de latencia no miden exactamente lo mismo

Esto es importante aclararlo en tu trabajo.

* `ranking_eval(...)` mide la **calidad del ranking del modelo**.
* `benchmark_retrieval(...)` mide la **latencia del pipeline de serving** que implementaste. 

Entonces, cuando muestras mejoras en AUC, MAP, NDCG, Precision y Recall, estás hablando de **calidad de ranking**.
Cuando muestras `mean_ms`, `p95_ms` y `qps`, estás hablando de **tiempo de respuesta operativo**.

Ambas mediciones son válidas, pero no son exactamente la misma cosa. Conviene explicarlo así para que quede metodológicamente claro.

---

### 2. El experimento es de **warm-start**, no de cold-start

El código filtra `stream` y `holdout` para conservar solo usuarios e ítems que ya existían en la base. 

Eso significa que este benchmark no está evaluando:

* usuarios completamente nuevos;
* ítems completamente nuevos.

Está evaluando otra pregunta, más acotada y perfectamente válida:

> qué pasa cuando el sistema ya tiene una base entrenada y luego recibe nuevas preferencias de entidades conocidas.

Eso hay que dejarlo explícito como alcance del trabajo.

---

### 3. El retrieval usa `NearestNeighbors` como proxy práctico

El recuperador se construye con `NearestNeighbors(metric="cosine", algorithm="auto")` sobre los vectores de ítems. 

Eso está bien como aproximación experimental de retrieval, pero no conviene venderlo como prueba definitiva de escalabilidad masiva de indexación. Lo correcto sería decir algo como:

> se utilizó un backend práctico de recuperación por similitud coseno para comparar el comportamiento del pipeline base e híbrido dentro del entorno experimental.

Eso hace que tu afirmación sea sólida y no exagerada.

---

### 4. Lo que realmente se demuestra aquí

Tus resultados no dicen que `OnlineIBPRMejorado` por sí solo sea más rápido o más lento en cualquier escenario imaginable.
Lo que sí dicen es esto:

* si mantienes la recuperación sobre la base indexada;
* y usas el modelo online para ajustar al usuario y reranquear candidatos;

entonces obtienes una mejora pequeña en calidad con un costo de latencia muy bajo.  

---

## Conclusión integrada

Este experimento evalúa un sistema híbrido en el que `IBPR` actúa como modelo base para recuperación eficiente y `OnlineIBPRMejorado` actúa como componente de adaptación incremental sobre nuevas tandas de datos. El flujo temporal está bien planteado: primero se evalúa sobre la siguiente tanda y luego se actualiza el híbrido. Los resultados muestran que el sistema híbrido parte igual que el base y luego logra pequeñas mejoras acumuladas en calidad, especialmente en el holdout final, donde supera al base en AUC, MAP, NDCG@20, Precision@20 y Recall@20. Al mismo tiempo, el costo total de actualización online es extremadamente bajo y el overhead de latencia en serving resulta mínimo. 

La interpretación correcta sería:

> el híbrido permite incorporar adaptación online sin perder la estructura de recuperación eficiente del modelo base, obteniendo una mejora modesta pero consistente en calidad, con una penalización temporal muy pequeña.
