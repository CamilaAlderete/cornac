# Decisión metodológica — HPO del IBPR base

## 1. Objetivo

Seleccionar una configuración robusta para el modelo IBPR base antes de optimizar los hiperparámetros propios de `OnlineIBPRMejorado`.

El modelo trabaja con feedback implícito:

- `rating >= 3` se transforma en una interacción positiva con valor `1.0`.
- Los ratings inferiores a 3 no se consideran negativos explícitos.
- La métrica primaria de selección es `NDCG@20`.

---

## 2. Protocolo de validación utilizado

Se utilizó únicamente el primer 60% cronológico global de MovieLens 1M como horizonte de desarrollo para HPO.

Dentro de ese 60%, la validación se realizó temporalmente por usuario:

- Fold 1: primer 50% del historial del usuario para entrenamiento y siguiente segmento hasta 65% para validación.
- Fold 2: primer 65% para entrenamiento y siguiente segmento hasta 80% para validación.
- Fold 3: primer 80% para entrenamiento y restante hasta 100% para validación.

Este protocolo se eligió porque el alcance del trabajo es warm-start. De este modo, cada usuario evaluado posee historial previo y se mantiene la dirección temporal dentro de su propio historial.

El 40% global restante no se utilizó para seleccionar hiperparámetros.

---

## 3. Espacio de búsqueda inicial

Se evaluaron combinaciones de:

- `k`
- `learning_rate`
- `lamda`
- `batch_size`
- `max_iter`

Se incluyeron explícitamente:

1. Una configuración de referencia basada en el paper/región de parámetros de IBPR y los defaults de Cornac.
2. La configuración utilizada en los experimentos piloto.
3. Configuraciones adicionales seleccionadas de forma reproducible dentro del espacio de búsqueda.

La búsqueda se realizó por etapas para evitar un grid search exhaustivo:

- Stage A: screening inicial.
- Stage B: confirmación en tres folds temporales.
- Stage C: evaluación de `max_iter`.
- Stage D: confirmación multi-seed.

---

## 4. Resultado de la primera búsqueda

La mejor configuración encontrada fue:

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 20,
    "learning_rate": 0.005,
    "lamda": 0.0001,
    "batch_size": 512,
    "verbose": True,
}
```

En la confirmación final multi-seed obtuvo:

- `NDCG@20 = 0.079341 ± 0.003909`

También se observó que aumentar `max_iter` de 20 a 50 no mejoró el resultado medio y aumentó claramente el coste de entrenamiento.

---

## 5. Por qué todavía no se congela la configuración

Aunque la configuración anterior fue la mejor de la búsqueda inicial, varios de sus hiperparámetros quedaron exactamente en los límites del espacio explorado:

- `k = 20` fue el mínimo evaluado.
- `learning_rate = 0.005` fue el mínimo evaluado.
- `lamda = 0.0001` fue el mínimo evaluado.
- `batch_size = 512` fue el máximo evaluado.
- `max_iter = 20` fue el mínimo evaluado.

Esto sugiere que la primera búsqueda identificó una región prometedora, pero no permite concluir todavía que el óptimo se encuentre dentro de los límites evaluados.

Por este motivo, congelar inmediatamente estos valores podría introducir una decisión prematura.

---

## 6. Decisión

Antes de congelar definitivamente el IBPR base, se realizará una única búsqueda de refinamiento local alrededor de la configuración ganadora.

La finalidad del refinamiento no es repetir el HPO completo, sino comprobar si el rendimiento sigue mejorando al explorar valores adyacentes a los límites encontrados.

Se mantendrán sin cambios:

- MovieLens 1M como dataset de desarrollo.
- El primer 60% global como horizonte HPO.
- La validación temporal por usuario.
- `NDCG@20` como métrica primaria.
- El escenario de feedback implícito.
- El alcance warm-start.

Después del refinamiento:

1. se seleccionará la configuración final del IBPR base;
2. se congelarán sus hiperparámetros;
3. se procederá al HPO de los parámetros específicos de `OnlineIBPRMejorado`;
4. posteriormente se rerunearán los experimentos H1–H3 con la configuración congelada.

---

## 7. Estado actual

- Implementación del algoritmo: completada.
- Tests invariantes: completados.
- Experimentos piloto y multi-seed: completados.
- Primera búsqueda de hiperparámetros del IBPR base: completada.
- Refinamiento local del HPO del IBPR base: pendiente.
