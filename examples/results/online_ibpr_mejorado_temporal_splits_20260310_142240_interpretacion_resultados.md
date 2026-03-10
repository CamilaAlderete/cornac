Estos resultados dicen algo bastante importante:

## 1) La mejora del `partial_fit_recent(...)` **sí se mantiene** al cambiar el corte temporal

En los tres splits, el `OnlineIBPRMejorado` con la configuración fija:

* `cosine_bpr`
* `epochs=1`
* `max_steps=5`
* `update_V=False`

mejora al baseline `IBPR` después de los partial retrainings. Esa configuración es precisamente la que deja fija el script del experimento temporal. 

Eso es valioso porque implica que la ganancia que viste antes **no parece depender de un único split afortunado**.

---

## 2) Qué pasa en `split_50_20_30`

Baseline:

* AUC **0.8129**
* MAP **0.0937**
* NDCG@20 **0.1316**
* Precision@20 **0.1257**
* Recall@20 **0.0463** 

Después del **stage 4/4**:

* AUC **0.8144**
* MAP **0.0954**
* NDCG@20 **0.1343**
* Precision@20 **0.1294**
* Recall@20 **0.0476** 

### Lectura

Acá la mejora es bastante limpia:

* AUC sube **+0.0015**
* MAP sube **+0.0017**
* NDCG@20 sube **+0.0027**
* Precision@20 sube **+0.0037**
* Recall@20 sube **+0.0013**

Y además el progreso es bastante gradual:

* stage 1 ya mejora todo respecto al baseline
* stages 2, 3 y 4 siguen sumando pequeñas mejoras, sobre todo en NDCG, Precision y Recall.  

Este split es una señal clara de que el partial retraining **sí aporta valor real**.

---

## 3) Qué pasa en `split_60_20_20`

Baseline:

* AUC **0.8219**
* MAP **0.0953**
* NDCG@20 **0.1343**
* Precision@20 **0.1251**
* Recall@20 **0.0479** 

Después del **stage 4/4**:

* AUC **0.8233**
* MAP **0.0962**
* NDCG@20 **0.1362**
* Precision@20 **0.1269**
* Recall@20 **0.0472** 

### Lectura

Acá el patrón es un poco más mixto:

* AUC mejora **+0.0014**
* MAP mejora **+0.0009**
* NDCG mejora **+0.0019**
* Precision mejora **+0.0018**
* pero Recall cae **-0.0007**

Además, la trayectoria intermedia no es totalmente monótona:

* stage 1 mejora casi todo
* stage 2 y stage 3 tienen pequeños retrocesos en Precision/Recall
* recién stage 4 consolida una mejora más clara en AUC, MAP y NDCG. 

Entonces, en este split:

* el partial retraining **sí mejora**, pero no de forma tan redonda como en `50/20/30`
* parece haber un pequeño trade-off con Recall

---

## 4) Qué pasa en `split_70_15_15`

Baseline:

* AUC **0.8304**
* MAP **0.0973**
* NDCG@20 **0.1330**
* Precision@20 **0.1273**
* Recall@20 **0.0434** 

Después del **stage 4/4**:

* AUC **0.8338**
* MAP **0.1011**
* NDCG@20 **0.1404**
* Precision@20 **0.1349**
* Recall@20 **0.0441** 

### Lectura

Este es el split donde el efecto se ve más fuerte:

* AUC sube **+0.0034**
* MAP sube **+0.0038**
* NDCG@20 sube **+0.0074**
* Precision@20 sube **+0.0076**
* Recall@20 sube **+0.0007**

Y además el progreso chunk a chunk es bastante limpio:

* stage 1 ya mejora AUC, MAP, NDCG y Precision
* stage 2 sigue creciendo
* stage 3 vuelve a crecer
* stage 4 da el mejor resultado final en todas las métricas menos cambios mínimos de recall.  

Este split sugiere que **cuando el modelo base tiene más historia acumulada (70%) y el bloque de adaptación es más compacto (15%), el partial retraining funciona especialmente bien**.

---

## 5) Qué patrón general aparece entre los tres splits

Hay tres señales consistentes:

### A) AUC, MAP y NDCG mejoran en los tres splits

Eso es probablemente lo más importante.
Aunque el tamaño de la mejora cambia, el direction es estable:

* `50/20/30`: mejora clara
* `60/20/20`: mejora moderada
* `70/15/15`: mejora fuerte  

### B) Precision también mejora en los tres splits

Eso es bueno porque implica que el top-k recomendado se vuelve algo más preciso en todos los escenarios.  

### C) Recall no es tan estable

* en `50/20/30` mejora
* en `60/20/20` cae un poco
* en `70/15/15` mejora un poco

Entonces el Recall es la métrica más sensible a cómo definiste el split temporal.  

---

## 6) Qué significa esto para tu hipótesis

La lectura más razonable es esta:

> **El partial retraining parece robusto a distintos cortes temporales, pero el tamaño del beneficio depende del split.**

O sea:

* no es un artefacto de una sola partición
* pero tampoco da exactamente la misma mejora siempre

Y eso es normal.
Cambiar el split cambia:

* cuánta historia tiene el `IBPR` base
* cuánta señal nueva recibe el online
* y qué tan exigente es el test final

---

## 7) Qué split parece más favorable

De los tres, el mejor resultado general parece ser `split_70_15_15`.

No porque tenga el baseline más bajo, sino porque el partial retraining mejora bastante:

* MAP
* NDCG
* Precision
* AUC

y lo hace de forma bastante progresiva hasta stage 4.  

Mi interpretación sería:

* con **más historia base**, `IBPR` deja un espacio latente más sólido
* y el `partial_fit_recent(...)` puede hacer un ajuste fino más eficaz
* sin necesidad de mover `V`

---

## 8) Qué dicen sobre la estabilidad con varios stages

En los tres splits, el modelo:

* no se rompe
* no colapsa
* y en general **mejora o se mantiene cerca de la mejora anterior** conforme avanzan los 4 stages

Eso es importante porque refuerza lo que ya habías visto en el experimento secuencial:
**varios partial retrainings consecutivos no parecen desestabilizar el modelo**, al menos con esta configuración rápida.

---

## 9) Qué conclusión práctica sacaría

Yo resumiría los resultados así:

> Con `cosine_bpr`, `epochs=1`, `max_steps=5` y `update_V=False`, el `OnlineIBPRMejorado` muestra mejoras consistentes frente al baseline `IBPR` en distintos cortes temporales. Las ganancias son más fuertes en AUC, MAP, NDCG y Precision que en Recall. El split `70/15/15` parece especialmente favorable, lo que sugiere que el partial retraining funciona mejor cuando parte de una base histórica más consolidada.

## 10) Recomendación

Con lo que muestran estos resultados, la estrategia híbrida se ve bastante bien parada:

* `IBPR` para el entrenamiento base
* `OnlineIBPRMejorado` para actualizaciones parciales rápidas

Y si tuvieras que elegir una lectura operativa, sería esta:

* **sí conviene seguir explorando el partial retraining**
* y **sí parece razonable preferir escenarios con buena historia base antes de depender demasiado del online**

Si querés, el siguiente paso puede ser cruzar **los resultados del experimento temporal** con **los del experimento de chunks** y decirte cuál configuración luce más robusta en conjunto.
