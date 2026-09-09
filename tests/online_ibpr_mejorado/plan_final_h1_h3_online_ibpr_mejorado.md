# Plan experimental pre-registrado — H1-H3 de OnlineIBPRMejorado

## 1. Propósito

Este documento define y congela el protocolo del experimento definitivo H1-H3 de `OnlineIBPRMejorado`.

El objetivo es comparar, sobre exactamente el mismo stream temporal y con configuraciones previamente congeladas:

```text
IBPR_STALE
vs
OnlineIBPRMejorado
vs
IBPR_FULL_RETRAIN
```

Este plan debe considerarse la especificación metodológica previa a la ejecución.

Después de ejecutar H1-H3:

- no se modificarán `R900` ni `O014`;
- no se cambiarán los chunks porque un resultado sea favorable o desfavorable;
- no se agregarán o eliminarán seeds basándose en los resultados observados;
- no se modificará la métrica primaria;
- cualquier análisis adicional deberá identificarse explícitamente como exploratorio o como una etapa posterior.

---

## 2. Estado previo y configuraciones congeladas

### 2.1 IBPR base congelado — R900

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

Identificador experimental:

```text
R900
```

Esta configuración ya fue seleccionada y cerrada mediante HPO temporal warm-start sobre MovieLens 1M.

No se vuelve a optimizar en H1-H3.

### 2.2 OnlineIBPRMejorado congelado — O014

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

Identificador experimental:

```text
O014
```

Esta configuración ya fue seleccionada y cerrada mediante HPO prequential por usuario.

No se vuelve a optimizar en H1-H3.

### 2.3 Seeds finales reservadas

```python
FINAL_SEEDS = [777, 999]
```

Estas seeds quedaron fuera del HPO online y se utilizarán para la evaluación definitiva H1-H3.

---

## 3. Hipótesis evaluadas

### H1 — Adaptación frente a un modelo stale

Se estudia si, después de incorporar nuevas interacciones:

```text
OnlineIBPRMejorado > IBPR_STALE
```

en calidad de ranking.

Métrica primaria:

```text
NDCG@20
```

La comparación principal será el delta pareado:

```text
ΔH1 = NDCG@20_Online - NDCG@20_Stale
```

### H2 — Eficiencia frente a Full Retrain

Se estudia si:

```text
coste(partial update) << coste(full retrain)
```

sobre el mismo historial observado.

Se medirán:

```text
online_update_time_s
full_retrain_time_s
cumulative_online_time_s
cumulative_full_retrain_time_s

speedup =
cumulative_full_retrain_time_s /
cumulative_online_time_s

online_full_cost_fraction =
cumulative_online_time_s /
cumulative_full_retrain_time_s
```

### H3 — Calidad frente a Full Retrain

Se estudia:

```text
OnlineIBPRMejorado
vs
IBPR_FULL_RETRAIN
```

sin exigir que Online supere al full retrain.

Se calculará:

```text
ΔH3 = NDCG@20_Online - NDCG@20_FullRetrain
```

y, como medida descriptiva complementaria:

```text
adaptation_recovery =
(Online - Stale) /
(FullRetrain - Stale)
```

para `NDCG@20`, únicamente cuando:

```text
FullRetrain - Stale > 0
```

No se afirmará equivalencia estadística entre Online y Full Retrain salvo que en una etapa futura se diseñe una prueba formal específica.

---

## 4. Dataset y transformación implícita

Dataset:

```text
MovieLens 1M
```

Carga:

```python
movielens.load_feedback(fmt="UIRT", variant="1M")
```

Transformación:

```text
rating >= 3 -> interacción positiva 1.0
rating < 3  -> no observado
```

Las interacciones positivas se ordenarán globalmente por timestamp.

La implementación deberá conservar un orden determinista cuando existan timestamps idénticos.

Regla recomendada:

```text
(timestamp, posición_original_en_el_dataset)
```

La posición original se utiliza únicamente como desempate determinista.

No se interpretará un desempate dentro del mismo timestamp como evidencia de precedencia temporal real.

---

## 5. Separación cronológica definitiva

La separación principal queda congelada en:

```text
MovieLens 1M positivo, orden cronológico global

0% ----------------------------- 60% ----------------------------- 100%
      entrenamiento IBPR base R900          stream definitivo H1-H3
      + horizonte de desarrollo HPO         sin retuning posterior
```

Formalmente:

```python
base_end = int(len(all_positive_rows) * 0.60)

base_rows_raw   = all_positive_rows[:base_end]
future_rows_raw = all_positive_rows[base_end:]
```

El primer 60% se utiliza para entrenar el IBPR inicial definitivo.

El 40% posterior constituye el stream definitivo H1-H3.

El 40% posterior no fue utilizado por los HPO de R900 u O014, aunque partes posteriores de MovieLens 1M sí fueron observadas durante pilotos antiguos.

Por ello, la formulación correcta es:

> evaluación con configuración completamente congelada y sin retuning posterior.

No debe denominarse “holdout completamente virgen”.

---

## 6. Universo warm-start congelado

`OnlineIBPRMejorado` soporta actualmente únicamente usuarios e ítems conocidos.

Después de construir `base_rows_raw`, se obtendrán:

```python
BASE_USERS = usuarios presentes en base_rows_raw
BASE_ITEMS = ítems presentes en base_rows_raw
```

Estas dos colecciones definen el universo warm-start del experimento.

Una interacción futura será elegible cuando:

```text
u ∈ BASE_USERS
y
i ∈ BASE_ITEMS
```

Las interacciones futuras que involucren:

```text
usuario nuevo
o
ítem nuevo
```

quedan fuera del experimento H1-H3.

### Regla de equidad

El mismo filtrado se aplica a:

```text
IBPR_STALE
OnlineIBPRMejorado
IBPR_FULL_RETRAIN
```

Full Retrain no podrá obtener una ventaja incorporando usuarios o ítems cold-start que OnlineIBPRMejorado no puede representar.

Las filas excluidas deberán contabilizarse y reportarse.

Como mínimo se registrará:

```text
n_future_raw
n_future_warm
warm_start_fraction

n_excluded_unknown_user
n_excluded_unknown_item
n_excluded_unknown_both
```

---

## 7. Construcción de los chunks globales

El stream futuro se dividirá en:

```python
N_STREAM_CHUNKS = 4
```

A diferencia del HPO online, los chunks de H1-H3 son:

```text
GLOBALES
CRONOLÓGICOS
NO construidos por usuario
```

### 7.1 Orden correcto de construcción

Primero:

```text
1. separar globalmente 0%-60% / 60%-100%;
2. construir los límites temporales de los 4 chunks sobre future_rows_raw;
3. después aplicar dentro de cada chunk el filtro warm-start.
```

Los límites temporales no se redefinen después de observar cuántas filas warm-start retiene cada chunk.

### 7.2 Tamaño objetivo

Los cuatro chunks deben contener aproximadamente el mismo número de interacciones positivas crudas del tramo futuro.

Objetivo conceptual:

```text
future 40%

chunk 1 ≈ 25% del future
chunk 2 ≈ 25% del future
chunk 3 ≈ 25% del future
chunk 4 ≈ 25% del future
```

Esto equivale aproximadamente a:

```text
60%-70%
70%-80%
80%-90%
90%-100%
```

del total positivo, pero los límites efectivos se definirán sobre las filas reales.

### 7.3 Regla para timestamps empatados

No se permitirá dividir un mismo timestamp entre dos chunks.

Si el corte objetivo cae dentro de un grupo de filas con el mismo timestamp:

```text
el límite se desplaza hasta el final de ese timestamp
```

De esta forma deberá cumplirse entre chunks consecutivos:

```text
max_timestamp(chunk_t) < min_timestamp(chunk_t+1)
```

Si por las características del dataset esta desigualdad estricta no pudiera lograrse, el modo `--plan-only` deberá abortar y reportar el problema antes de entrenar cualquier modelo.

### 7.4 Filtrado warm-start por chunk

Después de fijar los límites globales:

```python
known_chunk_t = [
    row
    for row in raw_chunk_t
    if row.user in BASE_USERS
    and row.item in BASE_ITEMS
]
```

Cada chunk conocido debe contener al menos una interacción.

Si un chunk queda vacío después del filtrado:

```text
ABORTAR
```

No se redefinirá automáticamente el protocolo.

---

## 8. Secuencia global prequential

Se utilizarán tres puntos de evaluación:

```text
BASE 0%-60%

-> llega chunk 1
-> actualizar/reentrenar con chunk 1
-> evaluar chunk 2

-> llega chunk 2
-> actualizar/reentrenar con chunk 2
-> evaluar chunk 3

-> llega chunk 3
-> actualizar/reentrenar con chunk 3
-> evaluar chunk 4
```

Por tanto:

```text
n_updates_per_seed = 3
n_eval_points_per_seed = 3
```

Con dos seeds:

```text
total paired evaluation points = 6
total online partial updates   = 6
total full retrains            = 6
total initial base trainings   = 2
```

El chunk evaluado nunca puede haber sido utilizado previamente para entrenamiento.

---

## 9. Historial observado y exclusión de candidatos

En cada paso:

```text
observed_history_t =
base
+ todos los chunks conocidos ya llegados
```

Para el primer punto:

```text
observed_history_1 = base + chunk1
eval_1             = chunk2
```

Para el segundo:

```text
observed_history_2 = base + chunk1 + chunk2
eval_2             = chunk3
```

Para el tercero:

```text
observed_history_3 = base + chunk1 + chunk2 + chunk3
eval_3             = chunk4
```

Este mismo historial observado se utilizará para las tres ramas al excluir ítems ya consumidos del conjunto de candidatos.

Así:

```text
Stale mantiene factores stale,
pero su historial de consumo sí avanza.
```

Este comportamiento coincide con el baseline stale utilizado durante el HPO online.

---

## 10. Identidad de usuarios e ítems

El `Dataset` correspondiente al IBPR base define:

```text
uid_map
iid_map
```

Estos mapas quedan congelados para todo el trial de una seed.

Todos los datasets posteriores deben reutilizar esos mismos mapas.

Objetivos:

```text
- conservar los mismos índices internos de usuarios;
- conservar los mismos índices internos de ítems;
- mantener U y V comparables entre ramas;
- impedir ampliaciones accidentales del catálogo;
- garantizar una comparación warm-start estricta.
```

No se permitirá que Full Retrain reconstruya un mapa diferente.

---

## 11. Entrenamiento del IBPR inicial

Para cada seed:

```text
777
999
```

se entrenará un IBPR inicial independiente sobre `base_rows_raw`.

Antes de cada entrenamiento:

```python
np.random.seed(seed)
torch.manual_seed(seed)
```

Modelo:

```python
IBPR(
    k=20,
    max_iter=50,
    learning_rate=0.0025,
    lamda=1e-05,
    batch_size=512,
)
```

Se registrará:

```text
base_train_time_s
n_base_rows
n_base_users
n_base_items
```

Este coste inicial es común al sistema desplegado y **no forma parte del speedup principal de H2**.

---

## 12. Rama IBPR_STALE

La rama Stale parte de copias del IBPR inicial.

Durante todo el stream:

```text
U_stale = U_base
V_stale = V_base
```

No se ejecuta ningún entrenamiento adicional.

Sin embargo:

```text
el historial observado sí avanza
```

para impedir recomendar nuevamente ítems consumidos.

En cada evaluación:

```text
modelo       = factores base sin adaptar
train_set    = historial acumulado hasta el chunk recibido
test_set     = siguiente chunk no visto
```

---

## 13. Rama OnlineIBPRMejorado

La rama Online parte exactamente de:

```text
U_base
V_base
```

Para cada chunk recibido:

```python
OnlineIBPRMejorado.partial_fit_recent(
    recent_pairs=current_chunk,
    history_csr=history_after_current_chunk,
    max_steps=None,
    n_epochs=3,
)
```

con la configuración congelada O014.

### 13.1 Historial para negative sampling

Antes de cada partial update:

```text
history_csr =
historial previo
+
chunk que acaba de llegar
```

Por tanto, todos los positivos del chunk recién recibido ya forman parte del conjunto de positivos conocidos y no pueden ser muestreados como negativos.

### 13.2 Progresión de seeds online

El wrapper vigente utiliza:

```text
seed
seed + 1
seed + 2
...
```

para actualizaciones parciales sucesivas no vacías.

Para un trial con seed `s`:

```text
update chunk1 -> s
update chunk2 -> s+1
update chunk3 -> s+2
```

No se modificará esta política.

### 13.3 Invariante obligatorio de V

Antes de comenzar:

```python
V_reference = V_base.copy()
```

Después de cada partial update deberá cumplirse:

```python
np.array_equal(V_online, V_reference) == True
```

También se calculará:

```python
v_max_abs_diff =
max(abs(V_online - V_reference))
```

y deberá ser:

```text
0
```

Cualquier violación:

```text
ABORTAR TRIAL
```

H4 realizará posteriormente la evaluación formal de reutilización del índice, pero H1-H3 conservará esta condición como invariante de corrección.

---

## 14. Rama IBPR_FULL_RETRAIN

En cada punto temporal se entrenará un nuevo IBPR **desde cero** sobre todo el historial acumulado disponible.

Ejemplo:

```text
después de chunk1:
train = base + chunk1

después de chunk2:
train = base + chunk1 + chunk2

después de chunk3:
train = base + chunk1 + chunk2 + chunk3
```

Se utilizará siempre R900:

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

### 14.1 Desde cero

Full Retrain no utiliza:

```text
U_base como init_params
V_base como init_params
U_online
V_online
```

Cada retraining representa un reentrenamiento completo.

### 14.2 Seed de Full Retrain

Dentro de un trial con seed `s`, todos los full retrains utilizarán:

```python
np.random.seed(s)
torch.manual_seed(s)
```

antes del `.fit()` correspondiente.

La seed se mantiene constante dentro del trial; lo que cambia entre puntos temporales es el historial de entrenamiento.

Esto evita introducir una fuente adicional de variación por una política de seeds distinta entre retrainings.

### 14.3 Universo fijo

Aunque Full Retrain actualice tanto `U` como `V`, se mantendrán:

```text
los mismos usuarios base
los mismos ítems base
los mismos uid_map/iid_map
```

No se incorporará cold-start en esta comparación.

---

## 15. Evaluación de ranking

Métricas:

```python
[
    AUC(),
    MAP(),
    NDCG(k=20),
    Precision(k=20),
    Recall(k=20),
]
```

Métrica primaria:

```text
NDCG@20
```

Configuración de evaluación:

```text
rating_threshold = 1.0
exclude_unknowns = True
```

Para cada punto:

```text
train_set = historial observado después del chunk recibido
test_set  = siguiente chunk conocido no visto
```

Los tres modelos deberán utilizar exactamente el mismo:

```text
train_set
test_set
uid_map
iid_map
```

La evaluación no debe modificar ningún modelo.

---

## 16. Métricas pareadas

En cada combinación:

```text
seed × eval_point
```

se calculará para todas las métricas de calidad:

```text
Online - Stale
Online - Full Retrain
Full Retrain - Stale
```

Para NDCG@20 además:

```text
adaptation_recovery =
(Online - Stale) /
(FullRetrain - Stale)
```

Regla:

```text
si FullRetrain - Stale <= 0:
    adaptation_recovery = NA
```

No se forzará un valor numérico cuando el denominador no represente una ganancia positiva de Full Retrain sobre Stale.

---

## 17. Medición de tiempos para H2

Se utilizará:

```python
time.perf_counter()
```

### 17.1 Tiempo Online

El tiempo primario de adaptación online incluye únicamente:

```text
OnlineIBPRMejorado.partial_fit_recent(...)
```

No incluye:

```text
carga de MovieLens
construcción inicial de chunks
evaluación de métricas
escritura de CSV
```

### 17.2 Tiempo Full Retrain

El tiempo primario de full retraining incluye únicamente:

```text
IBPR.fit(accumulated_train_set)
```

No incluye:

```text
carga de MovieLens
construcción inicial de chunks
evaluación de métricas
escritura de CSV
```

### 17.3 Dataset/preparación

Si el script registra tiempo de preparación de datasets, deberá guardarse en una columna separada.

No se mezclará con el tiempo principal de H2.

### 17.4 Coste inicial

`base_train_time_s` se reportará, pero no se sumará a:

```text
cumulative_online_time_s
```

ni a:

```text
cumulative_full_retrain_time_s
```

porque el IBPR inicial es un coste común al punto de partida de ambas estrategias.

### 17.5 Coste acumulado

Por seed:

```text
cumulative_online_time_s =
sum(partial_update_time_s)

cumulative_full_retrain_time_s =
sum(full_retrain_time_s)

speedup =
cumulative_full_retrain_time_s /
cumulative_online_time_s

online_full_cost_fraction =
cumulative_online_time_s /
cumulative_full_retrain_time_s
```

También se calcularán estas relaciones por punto temporal.

---

## 18. Reglas predefinidas de interpretación

Estas reglas se fijan antes de observar los resultados definitivos.

### 18.1 H1

Variable primaria:

```text
ΔNDCG@20 = Online - Stale
```

Se reportará:

```text
media
desviación estándar
mediana
mínimo
máximo
número de puntos positivos
media por seed
media por punto temporal
```

Clasificación descriptiva:

```text
SOPORTE FUERTE:
- Δ medio > 0;
- ambos promedios por seed > 0;
- 6/6 puntos seed×tiempo > 0.

SOPORTE CONSISTENTE:
- Δ medio > 0;
- ambos promedios por seed > 0;
- al menos 5/6 puntos > 0.

EVIDENCIA MIXTA:
- Δ medio > 0 pero no se cumplen las condiciones anteriores.

SIN SOPORTE DIRECCIONAL:
- Δ medio <= 0.
```

Estas categorías son descriptivas y no sustituyen una prueba formal de significancia.

### 18.2 H2

H2 se considera direccionalmente favorable si:

```text
cumulative_online_time_s <
cumulative_full_retrain_time_s
```

en ambas seeds.

Para interpretar “sustancialmente inferior” se utilizará como referencia práctica predefinida:

```text
speedup >= 2x
```

equivalente a:

```text
online_full_cost_fraction <= 0.50
```

Si Online es más barato pero no alcanza ese umbral, el resultado se describirá como una reducción de coste, no como evidencia fuerte de una reducción sustancial.

### 18.3 H3

H3 no se convertirá en una prueba binaria de equivalencia.

Se informará:

```text
Online - Full Retrain
Full Retrain - Stale
adaptation_recovery
```

junto con:

```text
cost fraction
speedup
```

La expresión “calidad competitiva” sólo se utilizará acompañada de los valores concretos del gap de calidad y del coste.

No se utilizarán expresiones como:

```text
equivalente
igual
no inferior estadísticamente
```

sin una prueba formal diseñada para esas afirmaciones.

---

## 19. Validaciones obligatorias antes de entrenar

El modo `--plan-only` deberá validar como mínimo:

### Datos

```text
dataset = MovieLens 1M
rating threshold = 3.0
total positive rows
base rows 0%-60%
future rows 60%-100%
timestamp base start/end
timestamp future start/end
```

### Universo base

```text
n_base_users
n_base_items
```

### Warm-start

```text
n_future_raw
n_future_warm
warm_start_fraction
unknown users
unknown items
unknown both
```

### Chunks

Para cada chunk:

```text
raw rows
warm rows
retention
n_users
n_items
min timestamp
max timestamp
```

Además:

```text
max_ts(chunk1) < min_ts(chunk2)
max_ts(chunk2) < min_ts(chunk3)
max_ts(chunk3) < min_ts(chunk4)
```

### Configuraciones

Imprimir completas:

```text
R900
O014
FINAL_SEEDS
```

### Secuencia

Imprimir explícitamente:

```text
update chunk1 -> eval chunk2
update chunk2 -> eval chunk3
update chunk3 -> eval chunk4
```

### Ramas

```text
IBPR_STALE
OnlineIBPRMejorado
IBPR_FULL_RETRAIN
```

### Métricas

```text
AUC
MAP
NDCG@20
Precision@20
Recall@20
```

### Coste

```text
partial update time
full retrain time
cumulative time
speedup
cost fraction
```

### H3

```text
Online - Full Retrain
adaptation_recovery
```

`--plan-only` no entrenará ningún modelo.

---

## 20. Validaciones obligatorias durante la ejecución

En cada seed se comprobará:

```text
1. mismo base para las tres ramas;
2. mismos uid_map/iid_map;
3. mismo chunk de update;
4. mismo chunk de evaluación;
5. eval chunk nunca aparece en observed_history;
6. Online history incluye el chunk recién llegado;
7. V_online permanece exactamente igual a V_base;
8. Stale no modifica U ni V;
9. Full Retrain utiliza todo el historial acumulado;
10. todos los modelos producen métricas sobre el mismo test_set.
```

Ante una violación estructural:

```text
ABORTAR
```

No continuar generando resultados parciales como si fueran válidos.

---

## 21. Resultados que debe persistir el script

Nombre propuesto:

```text
final_h1_h3_online_ibpr_mejorado.py
```

Directorio:

```text
tests/online_ibpr_mejorado/results/
```

### 21.1 Log

```text
final_h1_h3_online_ibpr_mejorado_<timestamp>.txt
```

Debe contener:

```text
plan
configuraciones
datos
progreso
validaciones
ranking final
resumen H1-H3
```

### 21.2 CSV por punto temporal

```text
final_h1_h3_online_ibpr_mejorado_steps_<timestamp>.csv
```

Una fila por:

```text
seed × eval_point
```

Campos mínimos:

```text
seed
eval_point
update_chunk
eval_chunk

n_observed_rows
n_update_rows
n_eval_rows
n_eval_users
n_eval_items

online_update_time_s
full_retrain_time_s

stale_AUC
stale_MAP
stale_NDCG@20
stale_Precision@20
stale_Recall@20

online_AUC
online_MAP
online_NDCG@20
online_Precision@20
online_Recall@20

full_AUC
full_MAP
full_NDCG@20
full_Precision@20
full_Recall@20

online_minus_stale_AUC
online_minus_stale_MAP
online_minus_stale_NDCG@20
online_minus_stale_Precision@20
online_minus_stale_Recall@20

online_minus_full_AUC
online_minus_full_MAP
online_minus_full_NDCG@20
online_minus_full_Precision@20
online_minus_full_Recall@20

full_minus_stale_NDCG@20
adaptation_recovery_NDCG@20

online_v_exact_equal
online_v_max_abs_diff

full_v_exact_equal_base
full_v_max_abs_diff_base
```

### 21.3 CSV por seed

```text
final_h1_h3_online_ibpr_mejorado_trials_<timestamp>.csv
```

Una fila por seed.

Campos mínimos:

```text
seed
base_train_time_s

mean_online_minus_stale_NDCG@20
positive_online_minus_stale_points

mean_online_minus_full_NDCG@20
mean_adaptation_recovery_NDCG@20

total_online_update_time_s
total_full_retrain_time_s
speedup
online_full_cost_fraction

all_online_v_exact_equal
max_online_v_abs_diff
```

### 21.4 CSV resumen

```text
final_h1_h3_online_ibpr_mejorado_summary_<timestamp>.csv
```

Debe contener agregados multi-seed:

```text
quality means/std
paired deltas means/std
win counts
cost totals/means
speedup
cost fraction
adaptation recovery
V invariants
```

---

## 22. Reanudación y atomicidad

El script deberá aceptar:

```bash
--timestamp YYYYMMDD_HHMMSS
```

La unidad física de reanudación será:

```text
seed
```

Regla:

```text
si una seed está completamente terminada:
    reutilizar sus resultados

si una seed está incompleta:
    eliminar/reemplazar sus filas
    y volver a ejecutar la seed completa
```

No se intentará reanudar un estado Online a mitad de una seed sin reconstruirlo.

Esto evita:

```text
duplicados
estados parciales inconsistentes
secuencias de seeds incorrectas
```

Las filas de una seed deberán considerarse válidas únicamente después de completar sus tres puntos temporales.

---

## 23. CLI prevista

El script deberá soportar como mínimo:

```bash
python final_h1_h3_online_ibpr_mejorado.py --plan-only
```

y:

```bash
python final_h1_h3_online_ibpr_mejorado.py
```

Opcionalmente:

```bash
python final_h1_h3_online_ibpr_mejorado.py --timestamp 20260909_XXXXXX
```

`--plan-only`:

```text
- carga y prepara datos;
- valida protocolo;
- imprime tamaños y límites;
- NO entrena;
- NO genera métricas definitivas.
```

---

## 24. Orden de ejecución oficial

```text
1. implementar script H1-H3;
2. ejecutar tests de invariantes existentes;
3. ejecutar --plan-only;
4. revisar manualmente el plan impreso;
5. validar tamaños, timestamps y warm-start;
6. validar que las tres ramas sean idénticamente emparejadas;
7. ejecutar H1-H3;
8. verificar integridad de CSV;
9. analizar H1;
10. analizar H2;
11. analizar H3;
12. documentar resultados;
13. cerrar H1-H3 SIN RETUNING;
14. pasar a H4.
```

---

## 25. Qué está prohibido durante H1-H3

Después de iniciar la ejecución definitiva no se debe:

```text
- modificar R900;
- modificar O014;
- cambiar NDCG@20 como métrica primaria;
- cambiar los límites de chunks por rendimiento observado;
- cambiar el filtro warm-start;
- agregar cold-start sólo a Full Retrain;
- elegir seeds nuevas porque 777 o 999 sean desfavorables;
- eliminar un punto temporal porque empeore una métrica;
- modificar n_epochs online;
- modificar max_iter del full retrain;
- modificar la pérdida online;
- convertir una ablación posterior en retuning;
- utilizar el resultado de H1-H3 para ajustar el modelo.
```

Los fallos técnicos reales del código sí deben corregirse.

Si una corrección técnica puede modificar los resultados:

```text
- documentar el fallo;
- invalidar la ejecución afectada;
- corregir el script;
- volver a ejecutar todo H1-H3 bajo el mismo protocolo.
```

---

## 26. Criterio de cierre de H1-H3

La etapa se considera cerrada cuando:

```text
- ambas seeds están completas;
- existen 6 puntos pareados válidos;
- no hay violaciones de V en Online;
- no hay fuga temporal;
- Stale, Online y Full usan exactamente el mismo stream evaluable;
- los CSV son internamente consistentes;
- H1 está cuantificada;
- H2 está cuantificada;
- H3 está cuantificada;
- los resultados están documentados;
- no se realiza retuning posterior.
```

Después:

```text
H1-H3 CERRADO
-> H4 reutilización del índice
```

---

## 27. Relación con H4

H1-H3 ya registrará:

```text
V_online exact equal
max abs diff(V)
```

como invariantes.

Sin embargo, H4 seguirá siendo una etapa separada.

H4 deberá estudiar formalmente:

```text
- construcción del índice sobre V_base;
- reutilización del mismo índice después de partial updates;
- rebuild requerido por Full Retrain;
- indexed vs exhaustive agreement@20;
- indexed vs exhaustive recall@20;
- latencia mean/median/p95;
- tiempo de build/rebuild.
```

Por tanto, H1-H3 no debe expandirse innecesariamente para resolver también H4.

---

## 28. Fuentes de verdad para implementar este plan

La implementación deberá mantenerse alineada con:

```text
CONTEXTO_MAESTRO_Y_HOJA_DE_RUTA_ONLINE_IBPR_MEJORADO.md

hyperparameter_refinement_ibpr_base_resultados_y_decision.md

plan_hpo_online_ibpr_mejorado_validated.md

hyperparameter_search_online_ibpr_mejorado.py

cornac/models/ibpr/ibpr.py
cornac/models/ibpr/recom_ibpr.py

cornac/models/online_ibpr_mejorado/online_ibpr_mejorado.py
cornac/models/online_ibpr_mejorado/recom_online_ibpr_mejorado.py

tests/online_ibpr_mejorado/
test_online_ibpr_mejorado_invariants.py
```

Configuraciones de autoridad:

```text
IBPR base:
R900

OnlineIBPRMejorado:
O014
```

---

## 29. Próximo paso después de aprobar este plan

Una vez aprobado este documento:

```text
crear
tests/online_ibpr_mejorado/
final_h1_h3_online_ibpr_mejorado.py
```

Primera ejecución permitida:

```bash
python tests/online_ibpr_mejorado/final_h1_h3_online_ibpr_mejorado.py --plan-only
```

Sólo después de validar esa salida se permitirá ejecutar el experimento definitivo completo.
