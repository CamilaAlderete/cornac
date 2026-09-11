# Justificación técnica de OnlineIBPRMejorado frente a la implementación OnlineIBPR original

## 1. Propósito de este documento

Este documento registra la justificación técnica y experimental de las modificaciones introducidas en `OnlineIBPRMejorado` respecto de la implementación `OnlineIBPR` original analizada en este proyecto.

El objetivo no es afirmar que el concepto de Online-IBPR sea incorrecto. La idea general de adaptar un modelo IBPR previamente entrenado mediante actualizaciones parciales de los factores de usuario, manteniendo fijos los factores de ítems, es razonable y especialmente atractiva en escenarios donde se desea evitar un reentrenamiento completo y preservar la representación de ítems utilizada para recuperación top-k.

La necesidad de mejora surge de varios problemas concretos detectados en la implementación original.

---

## 2. Intuición valiosa del OnlineIBPR original

La implementación original introduce una idea útil frente al IBPR offline:

```text
IBPR offline:
actualiza U y V

OnlineIBPR:
actualiza solamente U
mantiene V fija
```

Donde:

- `U`: factores latentes de usuarios.
- `V`: factores latentes de ítems.

Esta decisión está orientada, al menos conceptualmente, a:

1. reutilizar factores previamente entrenados;
2. adaptar los factores de usuario con información nueva proporcionada externamente;
3. reducir la cantidad de parámetros modificados durante la actualización;
4. mantener estable la representación de los ítems;
5. potencialmente reutilizar un índice ANN/top-k construido sobre `V`.

Por tanto, la idea central del Original es conservada en `OnlineIBPRMejorado`.

Debe aclararse que la implementación Original no impone por sí misma que los datos entregados a `fit()` sean necesariamente recientes. La recencia depende del protocolo externo que construye el dataset de actualización.

---

## 3. Problema principal: construcción incorrecta de tripletas BPR

El IBPR base utiliza tripletas:

```text
(u, i, j)
```

donde:

- `u` es el usuario;
- `i` es un ítem positivo observado por el usuario;
- `j` es un ítem no observado por ese usuario y tratado como negativo de muestreo para el objetivo BPR.

La implementación Original construye:

```python
X = train_set.matrix
X = X.tocoo()

triplets[:, 0] = X.row
triplets[:, 1] = X.col
triplets[:, 2] = X.data
```

Esto produce realmente:

```text
(u, i, rating)
```

Sin embargo, posteriormente interpreta la tercera columna como si fuera el índice del ítem `j`:

```python
regJ = V[triplets[:, 2], :]
```

Por tanto, el valor de feedback es utilizado como índice del supuesto ítem no observado.

### Consecuencia en nuestro escenario implícito

En los experimentos de este proyecto:

```text
rating >= 3  ->  1.0
```

Por ello:

```text
X.data = 1.0
```

y la implementación Original termina utilizando sistemáticamente:

```text
j = 1
```

en lugar de generar un ítem no observado específico de cada usuario.

Este constituye el defecto estructural principal identificado.

---

## 4. Ausencia de negative sampling condicionado al historial

En el contexto BPR utilizado en este trabajo, el ítem muestreado como `j` debe cumplir conceptualmente:

```text
j ∉ positivos conocidos del usuario
```

La implementación Original no construye ni consulta el historial positivo del usuario para verificar esta condición.

No existe un procedimiento equivalente a:

```text
obtener positivos conocidos del usuario
muestrear j
rechazar j si ya pertenece al conjunto positivo
```

Como consecuencia:

- el mismo `j` puede repetirse sistemáticamente;
- `j` puede pertenecer al conjunto de ítems positivos conocidos del usuario;
- no existe muestreo negativo condicionado al historial.

`OnlineIBPRMejorado` corrige esto construyendo conjuntos de positivos por usuario a partir del historial y rechazando cualquier candidato que ya pertenezca al conjunto positivo correspondiente.

---

## 5. Evidencia experimental del problema de `j`

La comparación diagnóstica Original vs Mejorado confirmó experimentalmente lo observado en el código.

Resultados principales:

```text
Original j == positivo actual:
≈ 0.19%

Original j pertenecía al conjunto positivo conocido acumulado del usuario
al momento de realizar la actualización:
≈ 54.80%
```

Este segundo porcentaje fue calculado utilizando el historial acumulado posterior a incorporar el chunk actual de actualización.

Por tanto, la interpretación precisa es:

> Aproximadamente el 54.8% de los `j` utilizados por el Original pertenecían al conjunto positivo conocido acumulado del usuario al momento de realizar la actualización, incluyendo las interacciones del chunk recién incorporado.

Esto no significa necesariamente que todos esos ítems hubieran sido observados antes del inicio del chunk. La medida utilizada incorpora el historial disponible después de agregar dicho chunk.

El hecho de que `j == i` sea poco frecuente no elimina el problema, ya que un `j` diferente de `i` puede igualmente pertenecer al conjunto de positivos conocidos del usuario.

---

## 6. Contrato de warm-start insuficientemente definido

La lógica `U-only` presupone que la representación fija de ítems `V` sea útil, compatible y semánticamente alineada con el espacio latente utilizado para actualizar los usuarios.

Sin embargo, la implementación Original permite que:

```text
U = None
V = None
```

y, en ese caso, inicializa ambos factores aleatoriamente.

Luego el optimizador se construye como:

```python
optimizer = torch.optim.Adam([U], ...)
```

Por tanto:

```text
V aleatoria
↓
V no pertenece al optimizer
↓
V permanece aleatoria
```

Esto resulta problemático si se pretende utilizar el modelo como una adaptación de un sistema previamente entrenado.

La precisión importante es la siguiente:

- para una actualización `U-only`, resulta esencial que `V` sea una representación fija útil y compatible;
- `U` podría, en principio, inicializarse y aprenderse desde otra condición inicial, aunque ese caso ya no representaría un warm-start de usuarios previamente entrenados;
- en nuestro protocolo experimental se exige que tanto `U` como `V` procedan de IBPR R900 porque el objetivo es adaptar usuarios conocidos desde un estado previamente entrenado.

`OnlineIBPRMejorado` hace explícito este contrato en su modo de actualización parcial.

---

## 7. `batch_size` no tiene efecto en la implementación Original

Aunque `batch_size` forma parte de la API, el core Original no crea mini-batches.

Cada epoch procesa todas las filas del `train_set` en una sola operación y realiza un único:

```python
optimizer.step()
```

Por tanto, si:

```text
n_epochs = 3
```

y la ejecución finaliza normalmente, se realizan exactamente:

```text
3 optimizer.step()
```

independientemente del valor de `batch_size`.

Esto implica que:

- `batch_size` es un parámetro sin efecto real;
- el comportamiento no coincide con la interfaz documentada;
- no existe una verdadera actualización mini-batch;
- los tiempos frente a `OnlineIBPRMejorado` no representan un presupuesto computacional equivalente.

`OnlineIBPRMejorado` utiliza mini-batches reales.

---

## 8. Regularización mal adaptada al esquema `U-only`

El Original utiliza:

```python
regI_unq = V[np.unique(triplets[:, 1:]), :]
```

pero las columnas de `triplets[:, 1:]` corresponden realmente a:

```text
item_id
rating
```

y no a:

```text
i
j
```

Por tanto se mezclan IDs de ítems con valores de feedback.

Además, la loss incluye regularización sobre `V`, aunque `V` no pertenece al optimizador.

Esto implica que `V` participa en el grafo computacional y puede recibir gradientes durante `backward()`, pero dichos gradientes no son aplicados por Adam porque `V` no forma parte de los parámetros del optimizador.

En consecuencia, para la actualización efectiva `U-only`, el término de regularización de `V` no produce actualización alguna sobre los factores de ítems.

`OnlineIBPRMejorado` adapta la regularización a los parámetros que realmente se optimizan.

---

## 9. Falta de una interfaz explícita de actualización parcial

El Original expone esencialmente:

```python
fit(train_set)
```

No define claramente una operación del tipo:

```python
partial_fit_recent(recent_pairs, history)
```

Por tanto, la semántica online depende de código externo que:

1. construya un nuevo `train_set`;
2. preserve correctamente los mapas de usuarios e ítems;
3. reutilice los factores previos;
4. invoque nuevamente `fit()`.

`OnlineIBPRMejorado` incorpora explícitamente:

```text
recent_pairs = interacciones nuevas
history_csr  = historial observado
```

y separa el entrenamiento completo de la actualización parcial.

---

## 10. Riesgo de integración por inconsistencia de IDs

Una actualización online debe preservar la correspondencia:

```text
raw user -> user_idx
raw item -> item_idx
```

entre el modelo base y cada actualización posterior.

La implementación Original no impone por sí misma invariantes explícitos sobre:

- `uid_map`;
- `iid_map`;
- cantidad de usuarios;
- cantidad de ítems;
- shapes de `U` y `V`.

Esto no constituye necesariamente un error del algoritmo si el código externo preserva correctamente los mapas, pero sí representa un riesgo de integración no controlado por la API Original.

Si un dataset de actualización reconstruyera los mapas con otro orden, los índices internos podrían dejar de representar las mismas entidades.

En nuestros experimentos diagnósticos este riesgo fue controlado explícitamente mediante mapas globales congelados y assertions.

---

## 11. Normalización: limitación, no error fatal en nuestro protocolo

El Original deja comentada la normalización final de `U` y `V`.

En nuestro protocolo específico:

- `V` procede de R900;
- `V` está normalizada;
- `V` permanece fija;
- cada usuario se evalúa contra la misma colección de vectores de ítems.

Por ello, la falta de normalización de `U` no modifica directamente el orden de ranking por producto interno para un mismo usuario con vector no nulo, ya que la norma de `U_u` actúa como un factor escalar común para todos sus ítems candidatos.

Por tanto, este punto se considera una limitación de consistencia y dinámica de optimización, pero no el defecto principal responsable de los resultados observados.

---

## 12. Qué conserva OnlineIBPRMejorado del Original

`OnlineIBPRMejorado` no reemplaza la idea central del Original.

Conserva:

```text
adaptación desde factores previamente entrenados
actualizar U
mantener V fija por defecto
procesar interacciones nuevas
preservar potencialmente la representación de ítems
```

La mejora consiste en hacer coherente y verificable esa actualización dentro del escenario evaluado.

---

## 13. Qué corrige OnlineIBPRMejorado

| Componente | Original | Mejorado |
|---|---|---|
| Warm-start obligatorio para actualización parcial | No | Sí |
| Actualización de U | Sí | Sí |
| V fija | Sí, siempre en esta implementación | Sí, por defecto |
| Positivo reciente como interfaz explícita | No; `i` se obtiene del `train_set` | Sí |
| Historial explícito | No | Sí |
| Tripleta `(u,i,j)` coherente con BPR | No | Sí |
| Negative sampling condicionado al historial | No | Sí |
| Rechazo de positivos conocidos | No | Sí |
| Mini-batches reales | No | Sí |
| `batch_size` efectivo | No | Sí |
| `partial_fit_recent` | No | Sí |
| Validación de shapes | No explícita | Sí |
| Validación de usuarios/ítems conocidos | No explícita | Sí |
| Reproducibilidad por seed | No explícita | Sí |
| `V` exacta con `update_V=False` | Sí de facto | Sí como contrato |
| Manejo de update vacío | No explícito | Sí |
| `max_steps` | No | Sí |

---

## 14. Evidencia experimental de la mejora

La comparación diagnóstica utilizó:

- MovieLens 1M;
- feedback implícito;
- primer 60% cronológico de desarrollo;
- tres escenarios: S50, S65 y S80;
- tres seeds diagnósticas;
- tres puntos prequentiales por trial;
- 27 puntos pareados en total.

Resultado medio en NDCG@20:

```text
IBPR Stale              0.047529
OnlineIBPR Original     0.049350
OnlineIBPRMejorado      0.069311
```

Deltas:

```text
Original - Stale        +0.001821
Mejorado - Stale        +0.021782
Mejorado - Original     +0.019961
```

El Mejorado superó al Original en:

```text
27/27 puntos NDCG@20
```

y también en los 27/27 puntos para las demás métricas registradas en la comparación diagnóstica.

Estos resultados muestran que el Original conserva cierta capacidad de adaptación, pero que la implementación corregida produce una adaptación considerablemente más fuerte y consistente dentro del protocolo diagnóstico utilizado.

---

## 15. Limitaciones de la comparación diagnóstica

Esta comparación no constituye una evaluación independiente de selección de modelo.

En particular:

- O014 fue seleccionado previamente utilizando este mismo horizonte general de desarrollo;
- el OnlineIBPR Original no recibió un HPO independiente;
- los parámetros numéricos fueron igualados donde la API lo permitía;
- los 27 puntos no deben interpretarse como 27 réplicas IID independientes;
- escenarios, seeds y puntos temporales presentan dependencia estructural;
- el experimento compara paquetes completos de implementación;
- no permite atribuir causalmente el gap a una única corrección;
- los tiempos Original vs Mejorado son descriptivos y no representan presupuestos de optimización equivalentes.

Por ello, el resultado debe interpretarse como una comparación diagnóstica consistente, no como una prueba de superioridad universal frente a la mejor configuración posible del Original.

---

## 16. Qué NO debe afirmarse

Estos resultados no permiten afirmar que:

- el concepto Online-IBPR sea incorrecto;
- OnlineIBPRMejorado sea universalmente superior a cualquier variante posible de OnlineIBPR;
- O014 sea la configuración óptima global;
- el gap observado sea causado únicamente por el negative sampling;
- los tiempos Original vs Mejorado representen una comparación computacional equivalente;
- los 27 puntos sean réplicas IID independientes;
- la reutilización efectiva de un índice ANN ya haya sido demostrada experimentalmente;
- el método resuelva cold-start de usuarios o ítems;
- H2 ya haya demostrado una reducción de costo frente a full retrain.

---

## 17. Alcance actual de OnlineIBPRMejorado

La versión actual de `OnlineIBPRMejorado` está diseñada para el escenario de:

```text
usuarios conocidos
ítems conocidos
factores U y V previamente entrenados
V fija por defecto
interacciones recientes incrementales
```

No pretende todavía resolver:

```text
cold-start de usuarios
cold-start de ítems
incorporación dinámica de nuevos usuarios mediante nuevas filas en U
incorporación dinámica de nuevos ítems mediante nuevas filas en V
gestión física del índice ANN
reconstrucción automática del índice
```

Estas restricciones deben mantenerse explícitas en la interpretación de resultados.

---

## 18. Formulación recomendada para la tesis

> La implementación OnlineIBPR analizada introduce una estrategia apropiada para adaptación incremental de IBPR al mantener fijos los factores de ítems y actualizar únicamente los factores de usuario. Esta decisión está orientada a reducir el costo de actualización al modificar una parte del modelo y, al mantener estable la representación de ítems, preservar potencialmente la estructura utilizada para recuperación top-k. Sin embargo, el análisis del código identificó limitaciones importantes en la construcción de las tripletas de entrenamiento, particularmente el uso del valor de feedback como índice del supuesto ítem no observado, la ausencia de muestreo negativo condicionado al historial del usuario, la falta de una interfaz explícita de actualización parcial y el uso inefectivo del parámetro de mini-batch. OnlineIBPRMejorado conserva la estrategia fundamental de actualización de usuarios con factores de ítems fijos, pero redefine estos componentes para proporcionar una actualización incremental coherente con la semántica de BPR, reproducible y adecuada para evaluación prequential dentro del escenario de usuarios e ítems conocidos considerado en este trabajo.

---

## 19. Conclusión

La motivación de `OnlineIBPRMejorado` no consiste en descartar el planteamiento del OnlineIBPR original, sino en completar y corregir su mecanismo de actualización.

La evidencia disponible se organiza en tres niveles:

```text
1. Auditoría estática del código
2. Contraste con el IBPR base
3. Evaluación diagnóstica Original vs Mejorado
```

Los tres niveles convergen en la misma conclusión:

> La idea de adaptar únicamente los factores de usuario manteniendo fija la representación de ítems es válida, pero la implementación Original no construye correctamente el componente BPR de la actualización online. OnlineIBPRMejorado conserva esa intuición y corrige los elementos necesarios para una actualización incremental coherente con la semántica BPR y con el protocolo experimental definido, dentro del escenario de usuarios e ítems conocidos evaluado en este trabajo.

---

## 20. Trazabilidad de evidencia

### Código analizado

```text
ibpr.py
recom_ibpr.py
online_ibpr.py
recom_online_ibpr.py
online_ibpr_mejorado.py
recom_online_ibpr_mejorado.py
```

### Script de comparación diagnóstica

```text
compare_online_ibpr_original_vs_mejorado.py
```

### Ejecución diagnóstica de referencia

```text
timestamp:
20260909_134700

protocol_version:
original_vs_mejorado_diag_v2

protocol_hash:
913c8696f00666c73fa70c7f12c04029749ea5d8fe840e66ca2f151f0ddae172
```

### Archivos de resultados asociados

```text
compare_online_ibpr_original_vs_mejorado_20260909_134700.txt
compare_online_ibpr_original_vs_mejorado_steps_20260909_134700.csv
compare_online_ibpr_original_vs_mejorado_trials_20260909_134700.csv
compare_online_ibpr_original_vs_mejorado_summary_20260909_134700.csv
```

### Resultado principal trazable

```text
Trials completos:                   9/9
Puntos pareados:                   27/27
Mejorado > Original en NDCG@20:    27/27
Original V exacta:                 True
Mejorado V exacta:                 True
```

La trazabilidad anterior permite vincular las afirmaciones experimentales de este documento con una ejecución concreta y reproducible del protocolo diagnóstico.

---

## 21. Estado dentro de la hoja de ruta

```text
Auditoría OnlineIBPR original              COMPLETADA
Justificación de OnlineIBPRMejorado        COMPLETADA
Comparación diagnóstica Original/Mejorado  COMPLETADA
Retuning posterior                         NO PERMITIDO

Pendiente:
H1-H3 definitivo
H4 reutilización/indexación
Ablaciones focalizadas
```
