# Plan H4 — Reutilización del índice con OnlineIBPRMejorado

## 1. Estado de entrada

H1-H3 sobre MovieLens 1M están cerradas, auditadas y congeladas.

Configuraciones que permanecen fijas:

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

FINAL_SEEDS = [777, 999]
```

No se realizará retuning para H4.

---

# 2. Pregunta de H4

La pregunta de H4 no es simplemente si `V` permanece igual; eso ya fue verificado en H1-H3.

La pregunta fuerte es:

> Si OnlineIBPRMejorado mantiene exactamente fijos los factores de ítems `V`, ¿puede reutilizarse el mismo índice construido sobre `V_base` después de las actualizaciones incrementales, evitando reconstrucciones y manteniendo una recuperación top-k coherente con la búsqueda exhaustiva?

La contraparte es:

> Cuando IBPR se reentrena completamente y `V` cambia, ¿el índice construido sobre `V_base` deja de representar correctamente al nuevo modelo y debe reconstruirse?

---

# 3. Evidencia previa

La implementación actual de `IBPR` y `OnlineIBPRMejorado`:

```text
hereda ANNMixin
get_vector_measure() -> MEASURE_DOT
get_user_vectors()    -> U
get_item_vectors()    -> V
score(u)              -> V.dot(U[u])
```

Esto establece una separación natural:

```text
índice ANN     <- V
query vector   <- U[u]
```

Con `update_V=False`:

```text
Online actualiza U
Online no modifica V
```

Por tanto, conceptualmente el índice de ítems puede permanecer fijo mientras cambian las consultas de usuario.

---

# 4. Backend ANN fijado para H4

Se propone:

```text
FAISS
Cornac FaissANN / FAISS IndexIVFFlat
CPU
inner product
```

Justificación:

1. IBPR y OnlineIBPRMejorado exponen `MEASURE_DOT`.
2. `FaissANN` mapea `MEASURE_DOT` a `METRIC_INNER_PRODUCT`.
3. El wrapper construye el índice directamente desde los item vectors.
4. FAISS dispone actualmente de wheel para Windows x86-64 + Python 3.12.
5. No es necesario implementar un índice propio.
6. Permite medir build/rebuild y consultar los mismos item IDs.

H4 no pretende replicar exactamente uno de los índices específicos del paper de 2017. El paper estudia LSH, KD-tree e inverted index. Aquí H4 es una validación de la propiedad de **reutilización del índice de ítems** utilizando el soporte ANN actual de Cornac.

---

# 5. Parámetros ANN

No se optimizarán parámetros ANN con resultados finales.

Se utilizarán los valores por defecto del wrapper Cornac `FaissANN`:

```text
nlist  = 100
nprobe = 50
use_gpu = False
```

Para facilitar reproducibilidad del benchmark:

```text
num_threads = 1
seed = 42
```

Si la instalación local de Cornac 2.3.5 presenta una incompatibilidad estructural con estos parámetros, se documentará antes de ejecutar H4 y cualquier corrección se realizará sin observar resultados H4.

---

# 6. Dataset y stream

H4 reutilizará exactamente la estructura de datos de H1-H3:

```text
MovieLens 1M
rating >= 3 -> 1.0
corte global 60/40
warm-start filter first
4 warm chunks
```

Chunks esperados:

```text
13,118
13,118
13,114
13,117
```

Secuencia:

```text
update chunk1 -> eval chunk2
update chunk2 -> eval chunk3
update chunk3 -> eval chunk4
```

La población de usuarios para las consultas H4 será la misma población PRIMARY de cada punto.

---

# 7. Modelos por seed

Para cada seed final:

```text
1. entrenar IBPR R900 sobre base;
2. guardar U_base y V_base;
3. inicializar Online O014 desde U_base/V_base;
4. construir UNA vez el índice base sobre V_base;
5. procesar los tres partial updates;
6. entrenar Full Retrain R900 desde cero en cada punto;
7. evaluar reutilización/rebuild del índice.
```

---

# 8. Invariante principal de Online

Después de cada partial update:

```python
np.array_equal(V_online, V_base) == True
```

y:

```text
max_abs_diff(V_online, V_base) = 0
SHA256(V_online) = SHA256(V_base)
```

Además:

```text
online_operational_index_build_count   = 1
online_operational_index_rebuild_count = 0
```

El índice construido inicialmente debe seguir siendo el mismo objeto/estado operacional utilizado en los tres puntos.

---

# 9. Invariante de Full Retrain

En cada Full Retrain:

```text
V_full != V_base
```

Se registrará:

```text
max_abs_diff(V_full, V_base)
SHA256(V_full)
```

La política operacional de Full Retrain será:

```text
full_index_rebuild_count = 1 por punto
```

Total esperado:

```text
3 rebuilds por seed
```

---

# 10. Comparaciones ANN

## 10.1 Online — índice reutilizado

En cada punto:

```text
query = U_online actualizada
index = índice original construido con V_base
```

Ground truth:

```text
exhaustive score = V_online.dot(U_online[u])
```

Como:

```text
V_online == V_base
```

el índice sigue representando exactamente el conjunto de item vectors del modelo Online actual.

La diferencia entre ANN y exhaustive deberá reflejar únicamente la aproximación propia del índice, no desactualización de `V`.

---

## 10.2 Full Retrain — índice base obsoleto

Control:

```text
query = U_full actualizada
index = índice original construido sobre V_base
ground truth = V_full.dot(U_full[u])
```

Este índice está basado en una representación de ítems distinta del modelo Full actual.

Se medirá su recuperación frente al ground truth de Full Retrain.

---

## 10.3 Full Retrain — índice reconstruido

Después:

```text
rebuild index sobre V_full
query = U_full
```

Se vuelve a medir contra:

```text
V_full.dot(U_full[u])
```

La comparación:

```text
FULL_STALE_INDEX
vs
FULL_REBUILT_INDEX
```

permitirá cuantificar el efecto real de mantener un índice construido con factores de ítems obsoletos.

---

# 11. Exclusión de ítems ya observados

La búsqueda exhaustiva y ANN deben trabajar sobre el mismo conjunto elegible.

Para cada usuario:

```text
seen_items = historial acumulado post-update
```

La búsqueda exhaustiva elimina esos items.

Para ANN se recuperará:

```text
raw_k = min(n_items, TOP_K + n_seen_items)
```

y después se filtrarán los `seen_items`.

Esta regla evita que la pérdida de elementos top-k sea causada artificialmente por pedir muy pocos candidatos antes de filtrar el historial.

---

# 12. Métricas de recuperación

Para cada usuario:

## Recall@20 respecto de exhaustive

```text
|ANN_top20 ∩ exhaustive_top20| / 20
```

Mide recuperación del conjunto correcto.

## Position Agreement@20

```text
número de posiciones r donde:
ANN_top20[r] == exhaustive_top20[r]
/
20
```

Mide concordancia sensible al orden.

## Exact Set Match@20

```text
set(ANN_top20) == set(exhaustive_top20)
```

Se reportará la proporción de usuarios con match exacto del conjunto.

## Exact Ordered Match@20

```text
ANN_top20 == exhaustive_top20
```

Se reportará la proporción de usuarios con lista completa idéntica y en el mismo orden.

---

# 13. Métricas de índice

Por seed y punto:

```text
V exact equal
max abs diff V
V SHA256

base index build time

Online:
operational rebuild count
reused-index recall@20
reused-index position agreement@20
exact set match rate
exact ordered match rate

Full:
stale-index recall@20
rebuilt-index recall@20
stale-index position agreement@20
rebuilt-index position agreement@20
rebuild time
rebuild count
```

---

# 14. Métricas de latencia

Se medirán separadamente:

```text
ANN query latency
exhaustive query latency
```

Para cada usuario:

```text
mean ms
median ms
p95 ms
```

Benchmark policy:

```text
warm-up previo
5 repeticiones por usuario
mediana por usuario
```

La latencia se reportará como evidencia descriptiva del entorno actual.

No se afirmará que el speedup observado sea universal.

MovieLens 1M tiene sólo ~3.5k ítems conocidos en la base, por lo que H4 en ML1M tiene como objetivo primario demostrar **correctitud y reutilización**. La validación de escala en ML10M será más informativa para eficiencia de retrieval.

---

# 15. Qué decide H4

H4 se considerará respaldada estructuralmente si se observa simultáneamente:

```text
1. V_online == V_base bit a bit en todos los puntos;

2. online operational rebuild count == 0;

3. el índice base puede consultarse con U_online actualizada;

4. la recuperación ANN reutilizando el índice no presenta una degradación
   atribuible a una V obsoleta;

5. Full Retrain cambia V;

6. Full Retrain requiere reconstruir el índice para que el índice vuelva a
   representar los item vectors del modelo actual.
```

No se exigirá Recall@20 = 1.0 para FAISS IVF, porque se trata de búsqueda aproximada.

La precisión ANN se reportará cuantitativamente.

---

# 16. Qué H4 NO demostrará

H4 no demostrará por sí sola:

```text
que FAISS sea el mejor índice;
que los parámetros ANN sean óptimos;
que el speedup de ML1M generalice a datasets grandes;
que Online tenga mejor calidad de recomendación que Full Retrain;
que H1 o H3 cambien de conclusión.
```

---

# 17. Relación con el paper IBPR

El paper de Indexable BPR estudia recuperación eficiente con:

```text
LSH
KD-tree
inverted index
```

y plantea la indexación de los vectores de ítems como mecanismo para evitar búsqueda exhaustiva.

Nuestro H4 utiliza FAISS como backend moderno compatible con producto interno a través de Cornac.

Por tanto:

> H4 no es una replicación exacta del benchmark de índices del paper original; es una evaluación experimental de la propiedad de reutilización del índice que surge al mantener `V` fija durante la adaptación online.

---

# 18. Preflight obligatorio

Antes de generar/ejecutar el script H4 definitivo se debe comprobar en el entorno real:

```text
Python                     3.12.x
Cornac                     2.3.5
import faiss               OK
FaissANN import            OK
MEASURE_DOT support        OK
tiny IndexIVFFlat build    OK
tiny query                 OK
IBPR ANN contract          OK
Online ANN contract        OK
```

Este preflight no utiliza MovieLens final ni produce resultados H4.

---

# 19. Orden de ejecución

```text
1. ejecutar h4_ann_preflight.py;

2. si PASS:
   congelar backend y parámetros ANN;

3. generar h4_index_reuse_online_ibpr_mejorado.py;

4. ejecutar --plan-only;

5. auditar plan;

6. ejecutar H4 definitivo;

7. auditar CSV/log;

8. generar README H4;

9. continuar con ablaciones focalizadas.
```

---

# 20. Estado

```text
H1-H3      CERRADO
H4 plan    DEFINIDO
H4 preflight PENDIENTE
H4 run     PENDIENTE
```
