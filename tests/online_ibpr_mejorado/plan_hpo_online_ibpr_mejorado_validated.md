# Plan validado de HPO — OnlineIBPRMejorado

## 1. Objetivo

Seleccionar exclusivamente los parámetros de adaptación incremental de
`OnlineIBPRMejorado`, manteniendo congelado el IBPR base.

IBPR base congelado:

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

El HPO online no vuelve a seleccionar estos valores.

---

## 2. Arquitectura online que queda fija

```text
k            = 20
update_V     = False
neg_sampling = uniform
normalize    = True
max_steps    = None
```

### Justificación

- `k=20`: debe coincidir con los factores del IBPR warm-start.
- `update_V=False`: es parte central de la propuesta y permite reutilizar el índice de ítems.
- `neg_sampling=uniform`: es la estrategia soportada por el modo `recent_pairs`.
- `normalize=True`: mantiene la geometría angular consistente con IBPR. En la versión
  mejorada del core, cuando `update_V=False` se normaliza solamente `U`; `V` permanece
  bit a bit idéntica. El script comprueba `np.array_equal(V_after, V_before)`.
- `max_steps=None`: cada partial update procesa todo el chunk. El truncamiento de pasos,
  si se estudia, pertenece a una ablación de coste posterior.

---

## 3. Parámetros que se optimizan

```python
learning_rate = [0.001, 0.0025, 0.005, 0.01, 0.02]
lamda         = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
batch_size    = [128, 256, 512, 1024]
n_epochs      = [1, 2, 3]
loss_mode     = ["cosine_bpr", "angular"]
```

No se ejecuta el producto cartesiano completo. Se realiza un screening
reproducible de 20 configuraciones con cobertura de todos los niveles.

---

## 4. Configuraciones ancla protegidas

Se conservan tres referencias durante todas las etapas:

1. **Piloto de adaptación**
   ```text
   lr=0.01, lamda=0.001, batch=512, n_epochs=1, cosine_bpr
   ```

2. **Región del IBPR base con actualización cosine**
   ```text
   lr=0.0025, lamda=1e-05, batch=512, n_epochs=1, cosine_bpr
   ```

3. **Región del IBPR base con objetivo angular**
   ```text
   lr=0.0025, lamda=1e-05, batch=512, n_epochs=1, angular
   ```

La tercera ancla evita que el objetivo angular original quede representado
únicamente por una combinación aleatoria.

---

## 5. Datos de desarrollo

Se utiliza MovieLens 1M con feedback implícito:

```text
rating >= 3 -> interacción positiva 1.0
```

Sólo el primer 60% cronológico global está disponible para HPO.

```text
0-60%   desarrollo/HPO
60-100% no utilizado por este HPO
```

Por lo tanto, el stream y holdout posteriores destinados a H1-H3 permanecen
fuera de la selección de hiperparámetros.

---

## 6. Escenarios temporales por usuario

Dentro del 60% de desarrollo:

```text
S50: prefijo 50% por usuario -> base; futuro -> 4 chunks
S65: prefijo 65% por usuario -> base; futuro -> 4 chunks
S80: prefijo 80% por usuario -> base; futuro -> 4 chunks
```

Un usuario entra al stream de un escenario cuando dispone de al menos cuatro
interacciones futuras. Los usuarios restantes pueden aportar su prefijo al
modelo base.

Las interacciones de stream con ítems no presentes en el catálogo base se
excluyen porque el alcance es warm-start y `V` permanece estático.

Los escenarios se usan como condiciones de robustez; no se interpretan como
un experimento causal puro sobre el porcentaje de historia, ya que el conjunto
de usuarios elegibles puede variar ligeramente entre escenarios.

---

## 7. Secuencia prequential

Para cada escenario y seed se entrena una sola vez el IBPR base congelado.
Todas las configuraciones online parten de copias exactas de esos mismos `U`
y `V`.

La secuencia es:

```text
base
  -> llega chunk 1
  -> partial update con chunk 1
  -> evaluar chunk 2
  -> llega chunk 2
  -> partial update con chunk 2
  -> evaluar chunk 3
  -> llega chunk 3
  -> partial update con chunk 3
  -> evaluar chunk 4
```

El chunk evaluado nunca se usa para entrenar antes de su evaluación.

### Historial para negative sampling

Cuando llega un chunk, el `history_csr` del partial update contiene:

```text
historial observado previo + chunk que acaba de llegar
```

Esto coincide con el experimento maestro validado. Todos los positivos del
chunk actual quedan excluidos del muestreo negativo.

---

## 8. Se usa el wrapper real, no el core directamente

Las actualizaciones que forman parte del HPO llaman:

```python
OnlineIBPRMejorado.partial_fit_recent(...)
```

El core se invoca directamente únicamente al inicio en un *smoke test* pequeño
del contrato de implementación (por ejemplo, para comprobar que `V` permanece
exactamente fija con `update_V=False` y `normalize=True`). Ningún *trial* del
HPO entrena llamando directamente al core.

Esto garantiza que los *trials* prueben exactamente el camino público que se
usará después en H1-H3, incluyendo la política reproducible de seeds:

```text
seed, seed+1, seed+2, ...
```

para partial updates sucesivos no vacíos.

---

## 9. Baseline y evaluación

En cada punto se compara:

```text
IBPR_STALE
vs
OnlineIBPRMejorado
```

El modelo stale no modifica sus factores, pero su conjunto de interacciones
observadas sí se actualiza para excluir de futuras recomendaciones los ítems
que el usuario ya consumió.

No se ejecuta full retraining durante HPO online porque no es necesario para
seleccionar los parámetros del partial update.

Se registran:

- AUC
- MAP
- NDCG@20
- Precision@20
- Recall@20
- deltas Online - Stale
- tiempos de partial update
- igualdad exacta de `V`

---

## 10. Métrica de selección

Primaria:

```text
mean delta NDCG@20 = OnlineIBPRMejorado - IBPR_STALE
```

Dentro de una misma etapa todos los candidatos comparten exactamente los mismos
escenarios, seeds y baselines stale; por tanto, ordenar por delta NDCG es
equivalente a ordenar por NDCG online absoluto, pero el delta expresa
directamente el beneficio de adaptación.

Desempates:

1. mayor NDCG@20 absoluto;
2. menor variabilidad del delta;
3. menor tiempo de adaptación;
4. menor `n_epochs`.

Si el mejor delta final resulta no positivo, el script lo informa explícitamente;
no se fuerza una conclusión favorable a H1.

---

## 11. Etapas

### O-A
```text
20 configuraciones
x S50, S65
x seed 42
```

### O-B
```text
top 5 + anclas protegidas
x S80
x seed 42
```

Se recalcula el ranking en S50+S65+S80.

### O-C
```text
top 3 + anclas protegidas
x S50, S65, S80
x seeds [42, 123, 2024]
```

Los trials físicos ya ejecutados se reutilizan.

Seeds `[777, 999]` permanecen fuera del HPO para H1-H3 final.

---

## 12. Invariante de índice

Cada partial update exige:

```python
np.array_equal(V_after, V_before) == True
```

y al terminar cada trial:

```python
np.array_equal(V_final, V_base) == True
```

Cualquier violación aborta el HPO.

Esto convierte la preservación de `V` en una condición de corrección, no sólo
en una observación posterior.

---

## 13. Reanudación

El trial físico se identifica por:

```text
configuración real + escenario + seed
```

Los detalles por chunk sólo se persisten después de completar el trial. Si una
ejecución anterior quedó interrumpida, las filas del mismo trial se reemplazan
antes de escribirse nuevamente, evitando duplicados en el CSV de chunks.

---

## 14. Regla de parada

Stage O-C cierra el HPO online.

No se abre automáticamente otro refinamiento aunque el ganador esté en un
extremo del espacio.

Después:

```text
1. congelar configuración online;
2. H1-H3 con configuración completa congelada;
3. H4 reutilización del índice;
4. ablaciones focalizadas;
5. MovieLens 10M sin nuevo HPO.
```
