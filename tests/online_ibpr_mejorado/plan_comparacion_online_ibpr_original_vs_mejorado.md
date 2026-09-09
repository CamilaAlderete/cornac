# Plan diagnóstico — OnlineIBPR original vs OnlineIBPRMejorado

## 1. Objetivo

Este experimento evalúa el efecto práctico de las correcciones introducidas en `OnlineIBPRMejorado` respecto de la implementación `OnlineIBPR` original.

La comparación será:

```text
IBPR_STALE
vs
OnlineIBPR ORIGINAL
vs
OnlineIBPRMejorado O014
```

El propósito es **diagnóstico y de validación técnica**.

No constituye el experimento definitivo H1-H3 y no debe utilizarse para volver a ajustar `R900` ni `O014`.

---

## 2. Qué pregunta responde

Pregunta:

> ¿Qué efecto práctico tiene la implementación incremental corregida y estabilizada de OnlineIBPRMejorado frente al comportamiento de la implementación OnlineIBPR original, partiendo del mismo IBPR base y recibiendo exactamente las mismas nuevas interacciones?

Este experimento complementa la auditoría estática del código original.

No pretende demostrar que `OnlineIBPRMejorado` es un óptimo universal ni que la implementación original sea un baseline competitivo optimizado.

---

## 3. Advertencia metodológica

`O014` fue seleccionado previamente utilizando el primer 60% cronológico global de MovieLens 1M y los escenarios S50/S65/S80.

Por tanto, esta comparación:

```text
NO es una evaluación independiente de selección de modelo.
```

Aunque se utilizarán seeds diagnósticas distintas de las seeds del HPO, el conjunto de desarrollo es el mismo.

La interpretación permitida es:

> comparación diagnóstica del comportamiento de ambas implementaciones dentro del horizonte de desarrollo ya utilizado.

No se utilizarán los resultados para retuning.

---

## 4. Datos

Dataset:

```text
MovieLens 1M
```

Transformación:

```text
rating >= 3 -> positivo 1.0
rating < 3  -> no observado
```

Horizonte utilizado:

```text
0%-60% global cronológico -> experimento diagnóstico
60%-100%                  -> NO TOCAR
```

El 60%-100% continúa reservado para H1-H3.

---

## 5. Protocolo temporal

Se reutiliza exactamente la familia de escenarios validada durante el HPO online:

```text
S50
S65
S80
```

Dentro del primer 60% global:

```text
S50: prefijo 50% por usuario -> base; futuro -> 4 chunks
S65: prefijo 65% por usuario -> base; futuro -> 4 chunks
S80: prefijo 80% por usuario -> base; futuro -> 4 chunks
```

La secuencia prequential es:

```text
base
-> llega chunk1
-> actualizar con chunk1
-> evaluar chunk2

-> llega chunk2
-> actualizar con chunk2
-> evaluar chunk3

-> llega chunk3
-> actualizar con chunk3
-> evaluar chunk4
```

El chunk evaluado nunca se utiliza previamente para entrenamiento.

Las interacciones futuras con usuarios o ítems desconocidos para el base se excluyen exactamente como en el HPO online.

---

## 6. Seeds diagnósticas

Para evitar reutilizar:

```text
HPO online: [42, 123, 2024]
H1-H3 final: [777, 999]
```

se congelan exclusivamente para esta comparación:

```python
DIAGNOSTIC_SEEDS = [31415, 27182, 16180]
```

Estas seeds no se modificarán después de observar resultados.

---

## 7. IBPR base común

Cada combinación:

```text
scenario × seed
```

entrena una única vez un IBPR base con R900:

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

Luego se copian exactamente:

```text
U_base
V_base
```

para:

```text
Stale
Original
Mejorado
```

---

## 8. Rama Stale

```text
U = U_base
V = V_base
```

No modifica factores.

El historial observado sí avanza para excluir ítems ya consumidos durante ranking.

Sirve como referencia para distinguir:

```text
adaptación útil
vs
movimiento que no mejora un modelo obsoleto
```

---

## 9. Rama OnlineIBPR original

La implementación original se ejecutará **sin modificar su código**.

Para cada chunk recién llegado se construye un `train_set` que contiene únicamente ese chunk, reutilizando los `uid_map` e `iid_map` del IBPR base.

Se llama:

```python
original_model.fit(update_chunk_train_set)
```

El modelo mantiene warm-start mediante sus `U` y `V` actuales.

### Parámetros compartidos con O014

Para evitar comparar configuraciones arbitrariamente diferentes se usan, donde la API original lo permite:

```python
ORIGINAL_DIAGNOSTIC_CONFIG = {
    "k": 20,
    "learning_rate": 0.005,
    "lamda": 1e-06,
    "batch_size": 1024,
    "max_iter": 3,
}
```

Estos valores corresponden a los parámetros numéricos compartidos con O014.

### Limitaciones que se conservan deliberadamente

No se corrige:

```text
- construcción (u,i,value) tratada como (u,i,j);
- ausencia de negative sampling válido;
- ausencia de partial_fit_recent;
- batch_size sin mini-batching real;
- ausencia de normalización final;
- scoring por producto interno;
- política original de actualización de U.
```

Corregir cualquiera de estos puntos convertiría al baseline en otra implementación y dejaría de ser `OnlineIBPR` original.

---

## 10. Rama OnlineIBPRMejorado

Se ejecuta O014 congelado:

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

Cada llegada usa:

```python
partial_fit_recent(
    recent_pairs=chunk_actual,
    history_csr=historial_previo + chunk_actual,
    max_steps=None,
    n_epochs=3,
)
```

---

## 11. Equidad de la comparación

Para cada escenario, seed y punto temporal:

```text
Stale
Original
Mejorado
```

deben compartir:

```text
mismo IBPR base
mismos U_base,V_base iniciales
mismo update chunk
mismo eval chunk
mismo historial observado para evaluación
mismo uid_map/iid_map
mismo conjunto de candidatos evaluables
mismas métricas
```

La única diferencia debe provenir de cómo Original y Mejorado procesan el chunk recibido.

---

## 12. Métricas

Calidad:

```text
AUC
MAP
NDCG@20
Precision@20
Recall@20
```

Métrica principal descriptiva:

```text
NDCG@20
```

Deltas:

```text
Original - Stale
Mejorado - Stale
Mejorado - Original
```

Tiempo:

```text
original_update_time_s
improved_update_time_s
```

Invariantes:

```text
V_original == V_base
V_mejorado == V_base
```

Diagnóstico geométrico adicional:

```text
mean ||U_original||
mean ||U_mejorado||
```

El diagnóstico de norma ayuda a observar el efecto de que Original no normaliza `U` después de actualizar, mientras O014 sí lo hace.

---

## 13. Interpretación permitida

Si:

```text
Mejorado > Original
```

de forma consistente, puede afirmarse que:

> dentro del protocolo diagnóstico de desarrollo, la implementación corregida produce mejor adaptación que la implementación original ejecutada literalmente bajo parámetros numéricos comparables.

No puede afirmarse:

```text
- superioridad universal;
- significancia causal aislada de cada corrección;
- evaluación independiente;
- que cada mejora individual sea responsable del delta completo.
```

El experimento compara **paquetes de implementación**, no una ablación factorial de correcciones individuales.

---

## 14. Salidas

Script propuesto:

```text
compare_online_ibpr_original_vs_mejorado.py
```

Resultados:

```text
results/
compare_online_ibpr_original_vs_mejorado_<timestamp>.txt

results/
compare_online_ibpr_original_vs_mejorado_steps_<timestamp>.csv

results/
compare_online_ibpr_original_vs_mejorado_trials_<timestamp>.csv

results/
compare_online_ibpr_original_vs_mejorado_summary_<timestamp>.csv
```

---

## 15. `--plan-only`

Antes de entrenar debe ejecutarse:

```bash
python tests/online_ibpr_mejorado/compare_online_ibpr_original_vs_mejorado.py --plan-only
```

Debe mostrar:

```text
MovieLens 1M
primer 60% únicamente
60%-100% no utilizado
S50/S65/S80
tamaños base/stream
retención warm-start
seeds diagnósticas
R900
config Original diagnóstica
O014
3 puntos de evaluación por trial
27 puntos pareados totales
```

`--plan-only` no debe entrenar modelos.

---

## 16. Reanudación

Unidad de trial:

```text
scenario × seed
```

Si un trial está completo:

```text
REUSE
```

Si está incompleto:

```text
eliminar sus filas de steps
repetir el trial completo
```

No se reanudan estados intermedios de los modelos.

---

## 17. Regla de cierre

La comparación queda cerrada cuando existen:

```text
3 escenarios
× 3 seeds
× 3 puntos temporales
=
27 puntos pareados válidos
```

y:

```text
- no existe fuga temporal;
- todas las ramas usan los mismos eval sets;
- los CSV son consistentes;
- se reportan deltas Original-Stale, Mejorado-Stale y Mejorado-Original;
- no se realiza retuning.
```

Después:

```text
documentar comparativa diagnóstica
-> volver al plan H1-H3
-> ejecutar --plan-only de H1-H3
```
