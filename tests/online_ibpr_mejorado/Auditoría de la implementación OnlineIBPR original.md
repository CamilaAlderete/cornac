## 4.1 Auditoría de la implementación original de OnlineIBPR

La revisión directa de la implementación original de `OnlineIBPR` confirmó varios problemas que justificaron el desarrollo de `OnlineIBPRMejorado`.

### Construcción incorrecta de las tripletas BPR

La implementación original convertía la matriz de interacciones a formato COO y construía:

```python
triplets[:, 0] = X.row
triplets[:, 1] = X.col
triplets[:, 2] = X.data
```

Por tanto, la estructura almacenada correspondía realmente a:

```text
(usuario, ítem, valor de interacción)
```

y no a la tripleta BPR esperada:

```text
(usuario, ítem positivo, ítem negativo)
```

Posteriormente, el algoritmo utilizaba directamente:

```python
regJ = V[triplets[:, 2], :]
```

tratando el **valor de la interacción como identificador del ítem negativo `j`**.

Esto resulta particularmente problemático en feedback implícito, donde las interacciones positivas se representan habitualmente con valor `1.0`, ya que múltiples observaciones pueden terminar utilizando el mismo índice de ítem como supuesto negativo.

### Ausencia de negative sampling válido

La implementación original no construía explícitamente un ítem negativo `j` no observado por el usuario.

No existía una comprobación equivalente a:

```text
j ∉ positivos_históricos(u)
```

ni se utilizaba el historial acumulado del usuario para impedir que un positivo conocido fuese seleccionado como negativo.

`OnlineIBPRMejorado` corrige este punto utilizando `history_csr` para construir el conjunto de positivos conocidos de cada usuario y realizar muestreo negativo válido.

### Ausencia de una operación incremental explícita

El wrapper original sólo exponía el entrenamiento mediante:

```python
fit(train_set, ...)
```

que volvía a invocar `online_ibpr(...)`. No existía una operación pública equivalente a:

```python
partial_fit_recent(recent_pairs, history_csr, ...)
```

para incorporar explícitamente únicamente nuevas interacciones.

La implementación mejorada introduce esta operación como mecanismo principal de adaptación incremental.

### Warm-start no obligatorio

El código original permitía inicializar aleatoriamente `U` y `V` cuando no se proporcionaban factores previos. Sin embargo, el optimizador actualizaba únicamente `U`.

Por tanto, era posible ejecutar el supuesto modo online con factores de ítems `V` inicializados aleatoriamente y posteriormente congelados.

`OnlineIBPRMejorado` exige factores previamente entrenados cuando `update_V=False`, garantizando que la adaptación online parta de un IBPR base válido.

### `batch_size` declarado pero no utilizado

Aunque la función original recibía un parámetro `batch_size`, el entrenamiento procesaba todas las observaciones simultáneamente en cada época y no implementaba división real en mini-batches. 
### Inconsistencia entre entrenamiento angular y scoring

El aprendizaje original utilizaba distancia angular:

```text
Scorei = arccos(cosine(Uu, Vi))
Scorej = arccos(cosine(Uu, Vj))
```

pero la normalización final de `U` y `V` estaba comentada.

Al mismo tiempo, el wrapper realizaba las recomendaciones mediante producto interno y declaraba `MEASURE_DOT` como medida de recuperación.

Sin normalización, el producto interno y la distancia angular no necesariamente producen el mismo orden de recomendación.

La implementación utilizada actualmente evita esta inconsistencia en la configuración experimental principal: el IBPR base produce factores normalizados y `OnlineIBPRMejorado`, con `normalize=True` y `update_V=False`, vuelve a normalizar los factores de usuario después de cada actualización sin modificar `V`. 
### Consecuencia para la investigación

Estos hallazgos justifican que `OnlineIBPRMejorado` no se considere simplemente un cambio de hiperparámetros sobre `OnlineIBPR`, sino una corrección y estabilización del mecanismo de actualización incremental.

Las mejoras principales son:

```text
- construcción correcta de positivos recientes (u,i);
- muestreo explícito de negativos válidos j;
- historial acumulado mediante history_csr;
- warm-start obligatorio;
- partial_fit_recent como API incremental;
- mini-batches reales;
- actualización controlable de V;
- reproducibilidad de seeds;
- preservación exacta de V cuando update_V=False;
- consistencia del ranking con la representación angular normalizada.
```