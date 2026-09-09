## Resultados de la búsqueda y optimización de hiperparámetros de OnlineIBPRMejorado

Una vez congelada la configuración del modelo IBPR base, se realizó la búsqueda de hiperparámetros correspondiente a `OnlineIBPRMejorado`. El objetivo de esta etapa fue seleccionar una configuración adecuada para la adaptación incremental del modelo, sin utilizar todavía los experimentos definitivos destinados a contrastar las hipótesis H1-H4.

### Protocolo experimental

La búsqueda se realizó sobre **MovieLens 1M**, transformando las valoraciones con `rating >= 3.0` en retroalimentación implícita positiva. Para evitar utilizar todo el dataset durante la optimización, el HPO se restringió al **primer 60 % cronológico global de las interacciones positivas**, equivalente a **501.886 interacciones**, de un total de 836.478. El 40 % global posterior no fue utilizado durante esta búsqueda.

Dentro del horizonte de desarrollo se definieron tres escenarios temporales por usuario:

- **S50:** 50 % inicial de la historia del usuario como base.
- **S65:** 65 % inicial como base.
- **S80:** 80 % inicial como base.

La porción futura de cada usuario elegible fue dividida en cuatro chunks temporales. El procedimiento prequential utilizado fue:

```text
Base
 → actualizar con chunk 1
 → evaluar sobre chunk 2
 → actualizar con chunk 2
 → evaluar sobre chunk 3
 → actualizar con chunk 3
 → evaluar sobre chunk 4
```

La retención después del filtrado warm-start fue muy elevada: **99,87 % en S50, 99,91 % en S65 y 99,94 % en S80**, por lo que la exclusión de interacciones asociadas a ítems desconocidos tuvo un impacto mínimo en el volumen evaluado.

La arquitectura estructural del modelo online se mantuvo fija durante todo el HPO:

```python
k = 20
update_V = False
neg_sampling = "uniform"
normalize = True
max_steps = None
```

De esta manera, la búsqueda se concentró únicamente en:

```text
learning_rate
lamda
batch_size
n_epochs
loss_mode
```

Además, `V` permaneció congelada durante toda la adaptación incremental, requisito central para preservar posteriormente la posibilidad de reutilizar el índice de ítems.

### Estrategia de búsqueda

La optimización se organizó en tres etapas sucesivas.

**O-A — Screening inicial**

Se evaluaron 20 configuraciones diferentes sobre S50 y S65 utilizando `seed=42`. El screening incluyó tres configuraciones ancla previamente definidas y configuraciones adicionales seleccionadas mediante un muestreo reproducible con cobertura de todos los niveles del espacio de búsqueda.

Las mejores configuraciones de esta primera etapa fueron:

| Ranking | Configuración | ΔNDCG@20 |
|---|---|---:|
| 1 | O014 | +0.022932 |
| 2 | O019 | +0.022685 |
| 3 | O012 | +0.019006 |
| 4 | O008 | +0.017557 |
| 5 | O017 | +0.016627 |

Las cinco configuraciones principales utilizaron `loss_mode="angular"`, mostrando una ventaja clara de esta función de pérdida dentro del espacio explorado.

**O-B — Robustez entre escenarios temporales**

Las cinco mejores configuraciones de O-A, junto con las configuraciones ancla protegidas, fueron evaluadas adicionalmente en S80.

O014 continuó ocupando la primera posición:

```text
O014: ΔNDCG@20 = +0.021211 ± 0.002982
O019: ΔNDCG@20 = +0.021086 ± 0.002772
O012: ΔNDCG@20 = +0.017310 ± 0.003106
```

Por lo tanto, las configuraciones mejor posicionadas en S50/S65 mantuvieron su comportamiento favorable al incorporar un tercer escenario con una historia base más extensa.

**O-C — Confirmación multi-seed**

Finalmente, las tres mejores configuraciones de O-B y las configuraciones ancla protegidas fueron evaluadas utilizando los tres escenarios temporales `S50`, `S65` y `S80` y las semillas:

```text
42
123
2024
```

El ranking final fue:

| Ranking | Configuración | ΔNDCG@20 | NDCG@20 | Tiempo de actualización |
|---|---|---:|---:|---:|
| **1** | **O014** | **+0.021625 ± 0.001952** | **0.069281** | **2.2178 s** |
| 2 | O019 | +0.021511 ± 0.001809 | 0.069167 | 3.1927 s |
| 3 | O012 | +0.017374 ± 0.002396 | 0.065030 | 1.8988 s |
| 4 | O003 | +0.011923 ± 0.002731 | 0.059579 | 1.1361 s |
| 5 | O001 | +0.008986 ± 0.001698 | 0.056642 | 1.0881 s |
| 6 | O002 | +0.007715 ± 0.001756 | 0.055371 | 1.2200 s |



### Configuración seleccionada

De acuerdo con el criterio de selección previamente establecido —maximizar el incremento medio de `NDCG@20` respecto de `IBPR_STALE`— la configuración seleccionada fue **O014**:

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

La configuración obtuvo:

```text
ΔNDCG@20 medio = +0.021625 ± 0.001952
NDCG@20 medio  = 0.069281
Tiempo medio acumulado de actualización por escenario = 2.2178 s
V exactamente igual al modelo base = True
máxima diferencia absoluta en V = 0
```



El resultado muestra que, dentro del conjunto utilizado para desarrollo y selección de hiperparámetros, la adaptación incremental produjo consistentemente una mejora frente al modelo IBPR mantenido estático.

No obstante, estos resultados **no deben interpretarse todavía como evidencia definitiva para aceptar H1**, debido a que las mismas observaciones fueron utilizadas para seleccionar la configuración online. Su función es exclusivamente identificar y congelar los hiperparámetros que posteriormente serán utilizados sin modificaciones en la evaluación definitiva.

### Decisión experimental

Con estos resultados se considera **cerrada la etapa de optimización de hiperparámetros de OnlineIBPRMejorado**.

La configuración O014 queda congelada y no se realizarán búsquedas adicionales alrededor de `batch_size=1024`, `n_epochs=3` ni de otros parámetros, aunque algunos valores seleccionados se encuentren en los extremos del espacio explorado. Extender la búsqueda después de observar estos resultados introduciría un riesgo adicional de sobreajuste al conjunto de desarrollo.

La siguiente etapa experimental utilizará, por tanto, dos configuraciones completamente congeladas:

```text
IBPR base:
k=20
max_iter=50
learning_rate=0.0025
lamda=1e-05
batch_size=512

OnlineIBPRMejorado:
learning_rate=0.005
lamda=1e-06
batch_size=1024
n_epochs=3
loss_mode=angular
update_V=False
neg_sampling=uniform
normalize=True
max_steps=None
```

A partir de este punto, los resultados de los experimentos H1-H4 **no serán utilizados para modificar estos hiperparámetros**. La siguiente etapa corresponde a la evaluación definitiva de **H1-H3**, comparando `IBPR_STALE`, `OnlineIBPRMejorado` e `IBPR_FULL_RETRAIN`.