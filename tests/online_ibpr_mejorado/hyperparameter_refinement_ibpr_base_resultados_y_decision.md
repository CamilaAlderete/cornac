# Cierre del HPO del IBPR base — análisis final y decisión metodológica

## 1. Alcance de este documento

Este documento resume y cierra la selección de hiperparámetros del modelo **IBPR base** después de:

1. una búsqueda inicial amplia;
2. un refinamiento local alrededor de la mejor configuración encontrada;
3. una confirmación final multi-seed sobre tres folds temporales warm-start.

La ejecución de refinamiento analizada es:

- timestamp: `20260902_092821`;
- dataset: MovieLens 1M;
- feedback: implícito positivo;
- transformación: `rating >= 3 -> 1.0`;
- métrica primaria de selección: `NDCG@20`;
- horizonte de HPO: primer 60% cronológico global;
- validación: temporal por usuario;
- seeds finales: `42`, `123`, `2024`;
- folds: 3 por seed;
- total de evaluaciones por finalista en R3: 9.

El 40% global posterior no se utilizó para seleccionar hiperparámetros.

---

## 2. Configuración que entró al refinamiento

La primera búsqueda HPO había seleccionado como baseline:

```python
R000 = {
    "k": 20,
    "max_iter": 20,
    "learning_rate": 0.005,
    "lamda": 0.0001,
    "batch_size": 512,
}
```

El refinamiento se realizó porque varios valores de R000 habían quedado en los bordes del espacio inicial de búsqueda.

---

## 3. Qué resolvió Stage R1

### 3.1 Dimensión latente `k`

Se probaron:

```text
k = 5, 10, 20, 30
```

Resultados principales:

- `k=5` produjo una caída clara de NDCG@20;
- `k=10` también quedó por debajo de `k=20`;
- `k=30` no superó al baseline;
- `k=20` continuó siendo la mejor región.

**Conclusión:** no existe evidencia en este refinamiento de que reducir `k` por debajo de 20 o aumentarlo a 30 mejore el modelo. El valor `k=20` queda suficientemente respaldado.

---

### 3.2 Regularización `lamda`

Se exploraron:

```text
1e-05
5e-05
1e-04  <- baseline
5e-04
```

Los valores pequeños mostraron una ligera ventaja. El mejor probe individual fue:

```text
lamda = 1e-05
```

Sin embargo, el efecto aislado fue pequeño:

- R004 (`lamda=1e-05`, resto baseline) superó al baseline en R2 por aproximadamente `+0.000385` NDCG@20;
- R005 (`lamda=5e-05`) lo superó por aproximadamente `+0.000295`.

**Conclusión:** una regularización más baja parece favorable, pero no es el principal origen de la mejora total.

---

### 3.3 `batch_size`

Se probaron:

```text
256, 512, 1024, 2048
```

El valor `512` conservó el mejor equilibrio de calidad.

- `1024` redujo tiempo de entrenamiento, pero perdió NDCG@20;
- `256` quedó por debajo;
- `2048` degradó más claramente el ranking.

**Conclusión:** `batch_size=512` queda resuelto y no requiere una expansión adicional.

---

### 3.4 Interacción `learning_rate × max_iter`

Se estudió explícitamente la relación entre tasa de aprendizaje y número de iteraciones.

La región más prometedora fue:

```text
learning_rate = 0.0025
max_iter      = 50
```

Esto fue importante porque tasas menores necesitan más iteraciones para converger.

Por ejemplo, con `learning_rate=0.0025`:

- 10 iteraciones fueron claramente insuficientes;
- 20 iteraciones mejoraron, pero todavía quedaron por debajo;
- 50 iteraciones produjeron el mejor resultado.

En cambio, `learning_rate=0.001` convergió demasiado lentamente, y `learning_rate=0.01` no resultó competitivo.

**Conclusión:** la principal mejora respecto de R000 proviene de disminuir la tasa de aprendizaje a `0.0025` y aumentar el presupuesto a `50` iteraciones.

---

## 4. Stage R2: combinación de los mejores movimientos

Los mejores cambios individuales se combinaron en R900:

```python
R900 = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

Ranking de tres folds con seed 42:

| Configuración | NDCG@20 |
|---|---:|
| **R900** | **0.081294** |
| R015 (`lr=0.0025`, `lamda=1e-04`, `iter=50`) | 0.081110 |
| R004 (`lamda=1e-05`, resto baseline) | 0.080940 |
| R005 (`lamda=5e-05`, resto baseline) | 0.080850 |
| R000 baseline | 0.080555 |
| R003 (`k=30`) | 0.079935 |

R900 fue la mejor configuración de esta etapa.

---

## 5. Stage R3: confirmación final multi-seed

Los tres finalistas fueron:

- R900;
- R015;
- R000.

Cada uno fue evaluado con:

```text
3 seeds × 3 folds = 9 ejecuciones
```

Resultado agregado:

| Configuración | NDCG@20 medio | Desv. estándar |
|---|---:|---:|
| **R900** | **0.080810** | 0.003925 |
| R015 | 0.080517 | 0.003921 |
| R000 | 0.079341 | 0.003909 |

La mejora absoluta de R900 sobre R000 fue:

```text
+0.001469 NDCG@20
```

La mejora relativa fue aproximadamente:

```text
+1.85%
```

---

## 6. Consistencia entre seeds

El comportamiento no depende únicamente del seed 42.

| Seed | R000 | R015 | R900 | R900 - R000 |
|---:|---:|---:|---:|---:|
| 42 | 0.080555 | 0.081110 | 0.081294 | +0.000740 |
| 123 | 0.078547 | 0.081036 | 0.081431 | +0.002883 |
| 2024 | 0.078920 | 0.079405 | 0.079704 | +0.000784 |

R900 supera el promedio de R000 en los **3 de 3 seeds**.

Considerando individualmente las nueve combinaciones `seed × fold`:

```text
R900 gana frente a R000 en 7 de 9
R900 pierde frente a R000 en 2 de 9
```

Esto aporta evidencia descriptiva de que la mejora no proviene de una única partición o inicialización aleatoria.

No se interpreta esta observación como una prueba formal de significancia estadística, ya que folds y seeds forman parte del mismo protocolo de desarrollo.

---

## 7. Efecto sobre las métricas secundarias

La selección se hizo exclusivamente por `NDCG@20`, tal como se fijó antes de ejecutar el HPO.

Aun así, R900 también presenta mejores medias que R000 en las métricas secundarias:

| Métrica | R000 | R900 | Diferencia |
|---|---:|---:|---:|
| AUC | 0.867549 | 0.868268 | +0.000719 |
| MAP | 0.060934 | 0.062022 | +0.001088 |
| NDCG@20 | 0.079341 | 0.080810 | +0.001469 |
| Precision@20 | 0.059256 | 0.059815 | +0.000560 |
| Recall@20 | 0.077689 | 0.079536 | +0.001847 |

La dirección conjunta de estas métricas es favorable y no muestra un intercambio evidente donde el aumento de NDCG@20 se obtenga sacrificando el resto del ranking.

---

## 8. Coste computacional

El incremento de calidad tiene un coste.

Tiempo medio de entrenamiento:

```text
R000: 93.15 s
R900: 226.79 s
```

R900 requiere aproximadamente:

```text
2.43x
```

el tiempo de entrenamiento de R000.

Esto representa aproximadamente un aumento del:

```text
143.5%
```

en coste de entrenamiento.

Este coste debe conservarse explícitamente en los experimentos posteriores, especialmente al comparar:

- IBPR stale;
- OnlineIBPRMejorado;
- full retraining.

No se debe ocultar ni normalizar este incremento.

---

## 9. Qué significa realmente la advertencia de frontera

El script informó:

```text
lamda    = LOW(1e-05)
max_iter = HIGH(50)
```

Esto significa únicamente que el ganador cayó en dos extremos del **espacio local evaluado**.

No significa que el experimento sea inválido ni que sea obligatorio continuar buscando.

### `lamda`

La evidencia indica que bajar de `1e-04` a `1e-05` aporta una mejora pequeña.

La diferencia entre R900 y R015, que difieren esencialmente en `lamda`, fue:

```text
+0.000293 NDCG@20
```

Por tanto, continuar disminuyendo `lamda` probablemente buscaría mejoras cada vez más pequeñas dentro del mismo conjunto de desarrollo.

### `max_iter`

El valor 50 fue mejor que 20 para `learning_rate=0.0025`, por lo que es posible que otro presupuesto produzca un resultado diferente.

Sin embargo, demostrar el número de iteraciones matemáticamente óptimo no es un requisito del trabajo.

La finalidad del HPO es obtener una configuración sólida y reproducible para evaluar posteriormente la contribución del método online, no maximizar indefinidamente MovieLens 1M.

---

## 10. Riesgo de continuar el HPO

Ya se utilizaron los mismos datos de desarrollo para:

1. una búsqueda inicial;
2. observar sus resultados;
3. diseñar un refinamiento dirigido;
4. observar los resultados del refinamiento;
5. realizar una confirmación multi-seed.

Una nueva expansión diseñada después de observar que `lamda=1e-05` y `max_iter=50` ganaron sería una decisión adicional condicionada por el mismo conjunto de desarrollo.

Seguir este ciclo:

```text
buscar -> observar borde -> expandir -> observar nuevo borde -> expandir...
```

puede terminar adaptando la selección de hiperparámetros específicamente a MovieLens 1M.

Por esa razón, detenerse ahora constituye una regla metodológica más conservadora que seguir buscando pequeñas ganancias.

---

## 11. Decisión final

**Se cierra el HPO del IBPR base en este punto.**

No se realizará una nueva búsqueda de frontera para `lamda` ni `max_iter` sobre MovieLens 1M.

La configuración de desarrollo seleccionada y congelada es:

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
    "verbose": True,
}
```

Resultado de selección:

```text
NDCG@20 = 0.080810 ± 0.003925
```

Esta configuración no debe describirse como un **óptimo universal**.

La formulación apropiada es:

> Configuración seleccionada mediante búsqueda temporal warm-start y refinamiento local sobre MovieLens 1M, utilizando NDCG@20 como criterio primario y confirmación multi-seed.

---

## 12. Regla de congelación

A partir de este punto:

- `k = 20` queda congelado;
- `max_iter = 50` queda congelado;
- `learning_rate = 0.0025` queda congelado;
- `lamda = 1e-05` queda congelado;
- `batch_size = 512` queda congelado.

Estos valores **no deben volver a ajustarse utilizando los resultados de H1-H4**.

Esto es especialmente importante para evitar que la configuración del modelo base se adapte posteriormente a las hipótesis que se quieren contrastar.

---

## 13. Qué queda por optimizar

El cierre del HPO del IBPR base no significa que todo el sistema esté ajustado.

El siguiente objetivo es seleccionar exclusivamente los parámetros de adaptación propios de `OnlineIBPRMejorado`, manteniendo congelada la configuración anterior del IBPR base.

Después:

1. congelar la configuración completa;
2. repetir H1-H3 con configuración congelada;
3. validar H4 (reutilización del índice);
4. ejecutar las ablaciones estrictamente necesarias;
5. utilizar MovieLens 10M como evaluación de escala/generalización con configuración congelada, sin volver a hacer HPO.

---

## 14. Estado del proyecto después de esta decisión

```text
1. Implementación OnlineIBPRMejorado             COMPLETADO
2. Tests invariantes                             COMPLETADO
3. Piloto Stale / Online / Full Retrain          COMPLETADO
4. Multi-seed piloto                             COMPLETADO
5. Figuras reproducibles                         COMPLETADO
6. HPO del IBPR base
   6.1 búsqueda inicial                          COMPLETADO
   6.2 refinamiento local                        COMPLETADO
   6.3 confirmación multi-seed                   COMPLETADO
   6.4 congelación del IBPR base                 COMPLETADO

7. HPO de adaptación OnlineIBPRMejorado          SIGUIENTE
8. H1-H3 con configuración congelada             PENDIENTE
9. H4 reutilización de índice                    PENDIENTE
10. Ablaciones focalizadas                       PENDIENTE
11. MovieLens 10M                                PENDIENTE
12. Cierre experimental y redacción final        PENDIENTE
```

---

## 15. Conclusión

El refinamiento cumplió su propósito: determinó que el ganador inicial podía mejorarse principalmente mediante una tasa de aprendizaje menor acompañada de un mayor presupuesto de iteraciones.

R900:

- obtuvo el mayor NDCG@20 agregado;
- superó al baseline en los tres seeds;
- ganó 7 de las 9 comparaciones `seed × fold`;
- mejoró también MAP, Precision@20 y Recall@20;
- mantuvo `k=20` y `batch_size=512`, que quedaron respaldados por los probes;
- tiene un coste de entrenamiento superior que debe ser reportado.

La mejora no es enorme, pero es consistente y suficiente para seleccionar una configuración de desarrollo.

La decisión metodológica es, por tanto, **detener el ajuste del IBPR base y avanzar a la optimización del componente online**.
