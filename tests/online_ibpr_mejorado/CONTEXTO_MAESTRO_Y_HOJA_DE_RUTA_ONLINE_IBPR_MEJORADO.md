# Contexto maestro y hoja de ruta — OnlineIBPRMejorado

## 1. Propósito de este documento

Este documento define el **estado actual, objetivo científico, hipótesis, alcance, decisiones congeladas y hoja de ruta experimental** del proyecto `OnlineIBPRMejorado`.

Su función es servir como **fuente de contexto principal** para continuar la investigación en futuras sesiones o chats sin depender de scripts exploratorios antiguos ni de conversaciones previas.

Este documento no sustituye:
- los papers de BPR e Indexable BPR;
- la implementación vigente;
- los tests de invariantes;
- los documentos de cierre de HPO;
- los scripts experimentales.

Su función es conectar todas esas piezas y dejar claro **qué se intenta demostrar y en qué orden**.

---

## 2. Idea central de la investigación

El objetivo de la investigación **no es demostrar que OnlineIBPRMejorado reemplaza a IBPR**.

La hipótesis de trabajo más sólida es que:

> **IBPR y OnlineIBPRMejorado cumplen funciones complementarias dentro de un mismo ciclo de recomendación.**

IBPR cumple el papel de modelo base u offline:
- aprende la representación global de usuarios e ítems;
- actualiza tanto `U` como `V`;
- produce factores compatibles con recuperación top-k mediante indexación;
- puede ejecutarse periódicamente mediante reentrenamientos completos.

OnlineIBPRMejorado cumple el papel de adaptación incremental:
- parte de un IBPR ya entrenado;
- recibe nuevas interacciones;
- modifica principalmente los factores de usuario `U`;
- puede mantener los factores de ítems `V` completamente estáticos;
- reduce la obsolescencia del modelo entre dos reentrenamientos completos;
- evita reconstruir el índice de ítems cuando `V` permanece fijo.

Por tanto, la tesis no debe formularse inicialmente como:

> “OnlineIBPRMejorado es mejor que IBPR”.

La formulación más apropiada es:

> **OnlineIBPRMejorado puede complementar a IBPR proporcionando adaptación incremental de preferencias recientes entre ciclos de reentrenamiento completo, con un coste computacional reducido y preservando la estructura de ítems necesaria para recuperación indexada.**

Esta formulación deberá mantenerse abierta a los resultados experimentales. La investigación debe demostrar o refutar sus componentes; no asumirlos de antemano.

---

## 3. Fundamento teórico

## 3.1 BPR

Bayesian Personalized Ranking (BPR) aborda recomendación con feedback implícito mediante preferencias por pares.

A partir de una interacción positiva `(u, i)`, un ítem no observado `j` puede utilizarse para expresar:

```text
u prefiere i sobre j
```

mediante una tripleta:

```text
(u, i, j)
```

El objetivo no consiste en predecir ratings absolutos, sino en aprender un ranking personalizado.

---

## 3.2 Indexable BPR

Indexable Bayesian Personalized Ranking (IBPR) conserva la lógica ordinal de BPR, pero reemplaza el kernel basado en producto interno por una formulación angular.

El objetivo de ese cambio es aprender representaciones más compatibles con estructuras geométricas de indexación utilizadas para recuperación top-k.

El trabajo original de IBPR plantea dos objetivos simultáneos:

1. calidad de recomendación;
2. eficiencia de recuperación top-k.

La representación final puede normalizarse, permitiendo utilizar los vectores con mecanismos basados en distancia o similitud angular/coseno.

---

## 4. Problema técnico identificado

Cornac incluye una implementación denominada `OnlineIBPR`, conceptualmente orientada a actualización online.

La revisión directa de su código original confirmó que el mecanismo de construcción de datos de entrenamiento no generaba correctamente las tripletas BPR `(u, i, j)`.

### 4.1 Construcción incorrecta de las tripletas BPR

La implementación original convertía la matriz de interacciones a formato COO y construía una matriz de tres columnas:

```python
triplets[:, 0] = X.row
triplets[:, 1] = X.col
triplets[:, 2] = X.data
```

Por tanto, esa estructura representaba realmente:

```text
(usuario, ítem, valor de interacción)
```

y no:

```text
(usuario, ítem positivo, ítem negativo)
```

Posteriormente se utilizaba:

```python
regJ = V[triplets[:, 2], :]
```

tratando directamente el valor almacenado de la interacción como índice del supuesto ítem negativo `j`.

En feedback implícito, donde los positivos se representan típicamente con valor `1.0`, este comportamiento puede producir repetidamente un supuesto `j=1` en lugar de muestrear un ítem no observado válido para cada usuario.

### 4.2 Ausencia de negative sampling controlado

La implementación original no construía explícitamente un negativo `j` que cumpliera:

```text
j ∉ positivos_conocidos(u)
```

No existía un historial acumulado utilizado para excluir:

- positivos históricos;
- positivos recién recibidos;
- ítems que ya no deben considerarse negativos para el usuario.

Esto impedía garantizar la semántica ordinal esperada de la tripleta BPR.

### 4.3 Ausencia de una operación incremental explícita

El wrapper original exponía `fit(train_set, ...)`, que volvía a invocar el entrenamiento sobre un `train_set`.

No existía una API pública equivalente a:

```python
partial_fit_recent(recent_pairs, history_csr, ...)
```

que expresara de manera explícita:

```text
recent_pairs = nuevas observaciones positivas
history_csr  = historial acumulado utilizado para negative sampling
```

### 4.4 Warm-start no obligatorio

La implementación original permitía inicializar `U` y `V` aleatoriamente cuando no se proporcionaban factores previos.

Sin embargo, el optimizador actualizaba únicamente `U`.

Por tanto, era posible ejecutar el supuesto modo online con `V` recién inicializada aleatoriamente y posteriormente congelada, sin exigir un IBPR base previamente entrenado.

### 4.5 `batch_size` declarado pero no utilizado para mini-batches

Aunque la función original recibía `batch_size`, el loop de entrenamiento operaba sobre todas las filas de `triplets` simultáneamente en cada época.

Por tanto, el parámetro estaba expuesto por la API pero no implementaba una partición real en mini-batches.

### 4.6 Inconsistencia entre entrenamiento angular y scoring original

La función de entrenamiento original calculaba la preferencia mediante distancia angular:

```text
Scorei = arccos(cosine(Uu, Vi))
Scorej = arccos(cosine(Uu, Vj))
```

pero la normalización final de `U` y `V` estaba comentada.

Al mismo tiempo, el wrapper utilizaba producto interno para `score()` y declaraba `MEASURE_DOT` como medida de recuperación.

Sin normalización, producto interno y distancia angular no necesariamente inducen el mismo ranking porque las magnitudes de los vectores intervienen en el producto interno.

La implementación vigente utilizada en esta investigación evita esta inconsistencia en la configuración principal:

```text
IBPR base:
    U y V normalizadas al finalizar el entrenamiento.

OnlineIBPRMejorado:
    normalize=True
    update_V=False

Después de cada partial update:
    U se normaliza;
    V permanece exactamente igual a V_base, que ya está normalizada.
```

Así, en la configuración principal:

```text
dot(U,V)
=
cosine(U,V)
```

y maximizar producto interno sobre vectores unitarios es equivalente a maximizar coseno y, por monotonía de `arccos`, a minimizar distancia angular.

### 4.7 Consecuencia para la investigación

Estos hallazgos justifican que `OnlineIBPRMejorado` no sea tratado como un simple cambio de hiperparámetros sobre `OnlineIBPR`.

La propuesta introduce y estabiliza explícitamente:

```text
- positivos recientes (u,i) correctamente representados;
- muestreo explícito de negativos válidos j;
- historial acumulado mediante history_csr;
- warm-start obligatorio;
- partial_fit_recent como API incremental;
- mini-batches reales;
- actualización opcional de V;
- preservación exacta de V cuando update_V=False;
- sucesivas actualizaciones reproducibles mediante seeds;
- consistencia entre representación angular normalizada y scoring.
```

El alcance continúa siendo warm-start para usuarios e ítems conocidos.

---

## 5. Solución implementada: OnlineIBPRMejorado

`OnlineIBPRMejorado` fue desarrollado como una extensión incremental de IBPR.

El mecanismo principal es:

```python
partial_fit_recent(recent_pairs, history_csr, ...)
```

donde:

```text
recent_pairs = nuevas interacciones positivas (u, i)
history_csr  = historial observado acumulado
```

Para cada positivo reciente se genera un negativo válido `j` que no pertenezca al conjunto conocido de positivos del usuario.

La implementación actual soporta:

- warm-start mediante `U` y `V` previamente aprendidos;
- actualización parcial;
- múltiples actualizaciones sucesivas;
- muestreo negativo reproducible;
- actualización opcional de `V`;
- preservación exacta de `V` cuando `update_V=False`;
- normalización de `U` sin modificar `V` cuando los ítems están congelados;
- dos variantes de función de pérdida:
  - `angular`;
  - `cosine_bpr`.

El alcance actual es **warm-start para usuarios e ítems conocidos**.

Cold-start queda fuera del objetivo principal.

---

## 6. Por qué mantener V fijo es una decisión central

En IBPR:

```text
U = factores de usuario
V = factores de ítems
```

La estructura de recuperación top-k se construye sobre los vectores de ítems.

Si una actualización online cambia únicamente:

```text
U
```

mientras mantiene:

```text
V
```

idéntica, entonces:

```text
índice(V) antes del update == índice(V) después del update
```

El usuario puede cambiar su posición o dirección en el espacio latente sin modificar el conjunto indexado de ítems.

Conceptualmente:

```text
                   ┌──────────────────────┐
                   │    IBPR offline      │
                   │ aprende U_base,V_base│
                   └──────────┬───────────┘
                              │
                         construir índice
                              │
                              ▼
                       INDEX(V_base)
                              │
        ┌─────────────────────┼──────────────────────┐
        │                     │                      │
        ▼                     ▼                      ▼
 nueva interacción      nueva interacción      nueva interacción
        │                     │                      │
        ▼                     ▼                      ▼
  actualizar U           actualizar U           actualizar U
  V permanece fija       V permanece fija       V permanece fija
        │                     │                      │
        └─────────────────────┴──────────────────────┘
                              │
                              ▼
                    reutilizar INDEX(V_base)
```

Esta propiedad conecta directamente la adaptación online con la motivación original de IBPR: **recomendación top-k eficiente mediante indexación**.

---

## 7. Pregunta principal de investigación

La pregunta principal queda formulada como:

> **¿Puede un modelo Indexable BPR previamente entrenado adaptarse incrementalmente a nuevas interacciones de usuarios conocidos, manteniendo estáticos los factores de ítems, con un coste computacional sustancialmente inferior al reentrenamiento completo y conservando una calidad de recomendación competitiva?**

Una pregunta de nivel sistema deriva de la anterior:

> **¿Puede esta adaptación incremental actuar como complemento entre reentrenamientos periódicos de IBPR, permitiendo un ciclo híbrido offline-online que reduzca la obsolescencia del recomendador sin reconstruir continuamente el índice de ítems?**

La segunda pregunta debe considerarse una consecuencia experimental a validar, no una conclusión predeterminada.

---

## 8. Hipótesis experimentales principales

## H1 — Adaptación frente a un modelo obsoleto

Después de recibir nuevas interacciones:

```text
OnlineIBPRMejorado > IBPR_STALE
```

en calidad de ranking, al menos de forma consistente en la métrica primaria `NDCG@20`.

Interpretación:

- `IBPR_STALE` representa un modelo que continúa funcionando después de su entrenamiento inicial pero no adapta sus factores;
- `OnlineIBPRMejorado` representa el mismo punto de partida con adaptación incremental.

H1 mide si realmente existe un beneficio de adaptación.

---

## H2 — Eficiencia frente al reentrenamiento completo

El coste de:

```text
partial update
```

debe ser sustancialmente menor que:

```text
full retraining sobre todo el historial acumulado
```

Se medirán principalmente:

- tiempo de actualización;
- tiempo acumulado;
- relación de coste;
- speedup.

H2 es una hipótesis central incluso si la mejora absoluta de ranking es moderada.

---

## H3 — Calidad frente al reentrenamiento completo

Se comparará:

```text
OnlineIBPRMejorado
vs
IBPR_FULL_RETRAIN
```

El objetivo no es exigir que OnlineIBPRMejorado supere siempre al reentrenamiento completo.

La hipótesis razonable es:

> OnlineIBPRMejorado puede mantener una calidad competitiva o recuperar una parte relevante de la adaptación que ofrecería el reentrenamiento completo, utilizando una fracción mucho menor del coste.

No se utilizará la expresión “equivalencia” salvo que se diseñe y supere una prueba estadística formal de equivalencia.

---

## H4 — Reutilización del índice

Con:

```python
update_V = False
```

debe cumplirse:

```python
V_online_after == V_base
```

de forma exacta.

Por tanto:

```text
rebuild del índice para OnlineIBPRMejorado = 0
```

mientras que un full retraining puede modificar `V` y requerir reconstrucción.

H4 deberá medir como mínimo:

- igualdad exacta de `V`;
- máxima diferencia absoluta de `V`;
- tiempo de construcción del índice base;
- coste de rebuild online;
- coste de rebuild full retrain;
- concordancia entre recuperación indexada y búsqueda exhaustiva;
- latencia de consulta top-k.

---

## 9. Hipótesis de sistema híbrido

Si H1-H4 reciben soporte suficiente, el siguiente nivel de la tesis puede formalizarse como una hipótesis adicional:

## H5 — Ciclo híbrido offline-online

Una estrategia que combine:

```text
reentrenamientos periódicos de IBPR
+
actualizaciones OnlineIBPRMejorado entre reentrenamientos
```

puede ofrecer un mejor compromiso entre:

- frescura de preferencias;
- calidad de ranking;
- coste computacional acumulado;
- frecuencia de reconstrucción del índice.

Arquitectura conceptual:

```text
                        OFFLINE
                           │
                           ▼
                    FULL IBPR RETRAIN
                     actualiza U y V
                           │
                           ├── reconstruye índice(V)
                           │
                           ▼
                    modelo base vigente
                           │
           ┌───────────────┴───────────────┐
           │                               │
           │        ONLINE / STREAM        │
           │                               │
           ▼                               │
    nueva interacción                     │
           │                               │
           ▼                               │
 OnlineIBPRMejorado                        │
 actualiza U solamente                    │
           │                               │
           ▼                               │
 mismo índice(V)                           │
           │                               │
           └──── repetir hasta próximo ────┘
                    full retrain
```

### Importante sobre el término “híbrido”

En este proyecto, **híbrido** debe significar inicialmente:

> combinación temporal de entrenamiento offline completo + adaptación online incremental.

No debe confundirse con:

- recommender híbrido colaborativo + contenido;
- fusión de scores de dos modelos;
- reranking de una lista de IBPR por otro algoritmo.

Esas variantes serían líneas diferentes y requerirían experimentos adicionales.

---

## 10. Posibles conclusiones de tesis según los resultados

La investigación no debe comprometerse con una única conclusión antes de ejecutar los experimentos.

Existen varios resultados científicamente válidos.

## Escenario A — Evidencia fuerte de complementariedad

Si:

- Online > Stale de forma consistente;
- Online ≈ Full Retrain en calidad práctica;
- Online << Full Retrain en coste;
- `V` permanece fija;
- el índice puede reutilizarse;

la conclusión principal sería:

> **OnlineIBPRMejorado es un complemento eficiente de IBPR para adaptación temporal entre reentrenamientos completos.**

Este escenario sustenta directamente la arquitectura híbrida offline-online.

---

## Escenario B — Mejora de calidad pequeña pero eficiencia extrema

Si la mejora sobre Stale es pequeña, pero:

- consistente;
- muy barata;
- mantiene calidad competitiva;
- evita rebuild del índice;

la contribución sigue siendo válida.

La tesis pasaría a enfatizar:

> **adaptación incremental de muy bajo coste y preservación de indexabilidad**, más que grandes ganancias absolutas de ranking.

---

## Escenario C — Online no mejora suficientemente la calidad

Si Online no supera de forma consistente a Stale:

- H1 quedaría sin soporte;
- no se deberá ocultar ese resultado.

Aun así podrían estudiarse:

- estabilidad;
- coste;
- sensibilidad;
- preservación del índice;
- condiciones bajo las cuales la adaptación resulta útil.

En ese caso no se afirmará que el sistema híbrido mejora el recomendador sin evidencia adicional.

---

## Escenario D — Actualizar V mejora mucho la calidad

`update_V=True` no forma parte de la configuración principal.

Si una ablación posterior mostrara una ganancia importante al actualizar `V`, el resultado se interpretaría como un trade-off:

```text
más adaptación
vs
pérdida de reutilización del índice
```

Esto no invalida la investigación; delimita las condiciones bajo las cuales conviene mantener el índice estático.

---

## 11. Estado validado de la implementación

La implementación `OnlineIBPRMejorado` dispone de tests de invariantes para comprobar, entre otros puntos:

- partial update requiere warm-start;
- con `update_V=False`, `U` cambia y `V` permanece igual;
- con `update_V=True`, `V` cambia;
- el negative sampling rechaza positivos históricos y recientes;
- shapes inválidos son rechazados;
- update vacío es identidad;
- misma seed produce el mismo resultado;
- `max_steps=0` es inválido;
- un update vacío del wrapper no consume el siguiente seed.

Estos tests deben considerarse la base de corrección funcional antes de interpretar resultados de calidad.

---

## 12. IBPR base: configuración congelada

El HPO del IBPR base está cerrado.

Configuración seleccionada:

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

Resultado de selección del HPO base:

```text
NDCG@20 = 0.080810 ± 0.003925
```

La configuración:

- fue seleccionada sobre MovieLens 1M;
- utilizó el primer 60% cronológico global como horizonte de desarrollo;
- utilizó validación temporal warm-start;
- tuvo confirmación multi-seed;
- queda congelada.

No debe llamarse “óptimo universal”.

No debe modificarse posteriormente utilizando H1-H4.

---

## 13. HPO de OnlineIBPRMejorado: cerrado y configuración congelada

El HPO de `OnlineIBPRMejorado` está **COMPLETADO**.

La búsqueda se ejecutó utilizando el IBPR base ya congelado como punto de partida:

```python
IBPR_CONFIG = {
    "k": 20,
    "max_iter": 50,
    "learning_rate": 0.0025,
    "lamda": 1e-05,
    "batch_size": 512,
}
```

Durante el HPO online no se volvieron a optimizar los parámetros del IBPR base. Cada trial partió de factores `U` y `V` aprendidos por ese IBPR congelado y buscó únicamente los parámetros propios de adaptación incremental.

La configuración seleccionada en la etapa final multi-seed O-C fue:

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

Resultado final de selección:

```text
mean ΔNDCG@20 = +0.021625 ± 0.001952
mean NDCG@20  = 0.069281
mean total partial-update time por escenario = 2.2178 s
V exact equal = True
max abs diff(V) = 0
```

La configuración O019 quedó muy próxima en calidad:

```text
O019 mean ΔNDCG@20 = +0.021511 ± 0.001809
```

pero O014 ocupó el primer lugar según la regla de selección predefinida y además presentó menor tiempo medio de adaptación.

La etapa O-C cerró formalmente el HPO online. No se realizará una nueva expansión automática alrededor de `batch_size=1024`, `n_epochs=3` ni de otros límites observados.

Los resultados de este HPO son **resultados de desarrollo y selección de hiperparámetros**. No constituyen por sí mismos evidencia definitiva para H1.

A partir de este punto, tanto el IBPR base como `OnlineIBPRMejorado` quedan completamente congelados para H1-H4.

---

## 14. Datos utilizados durante HPO online

MovieLens 1M se transforma a feedback implícito:

```text
rating >= 3 -> positivo 1.0
rating < 3  -> no observado
```

El HPO online utiliza sólo:

```text
primer 60% cronológico global
```

Dentro de ese horizonte se construyen escenarios temporales por usuario:

```text
S50
S65
S80
```

con cuatro chunks futuros.

La evaluación es prequential:

```text
llega chunk 1
-> actualizar con chunk 1
-> evaluar chunk 2

llega chunk 2
-> actualizar con chunk 2
-> evaluar chunk 3

llega chunk 3
-> actualizar con chunk 3
-> evaluar chunk 4
```

El chunk evaluado no se utiliza antes para entrenamiento.

---

## 15. Advertencia metodológica sobre MovieLens 1M

El 40% global cronológicamente posterior **no fue utilizado durante las etapas de selección de hiperparámetros del IBPR base ni de OnlineIBPRMejorado**.

Sin embargo, durante etapas piloto anteriores del proyecto ya se observaron resultados sobre particiones posteriores de MovieLens 1M.

Por tanto, ese segmento no debe describirse como un “holdout completamente virgen”, “nunca observado” o equivalente.

La formulación correcta para los experimentos definitivos será:

> **evaluación con configuración completamente congelada y sin utilizar esos resultados para volver a ajustar hiperparámetros.**

La defensa metodológica debe apoyarse en:

- congelación previa de R900;
- congelación previa de O014;
- ausencia de retuning posterior;
- evaluación multi-seed;
- separación cronológica explícita entre desarrollo y evaluación;
- posterior validación de escala/generalización en MovieLens 10M con configuración congelada.

La reutilización del mismo primer 60% para los HPO del modelo base y del componente online debe interpretarse como un único horizonte de **desarrollo y selección**, no como dos conjuntos de validación independientes.

El IBPR base se optimizó primero dentro de ese horizonte y quedó congelado. Posteriormente, partiendo de esa configuración fija, se optimizaron únicamente los parámetros de adaptación de `OnlineIBPRMejorado`.

Los resultados de ambas búsquedas pertenecen a la fase de desarrollo y no constituyen por sí mismos evidencia definitiva de H1-H4.

---

## 16. Comparación experimental definitiva — etapa actual

Después del cierre del HPO online, la configuración completa queda congelada.

### 16.1 Separación cronológica definitiva

MovieLens 1M se transforma primero a feedback implícito positivo:

```text
rating >= 3 -> positivo 1.0
rating < 3  -> no observado
```

Luego todas las interacciones positivas se ordenan globalmente por timestamp.

La separación experimental definitiva queda fijada conceptualmente como:

```text
MovieLens 1M positivo, orden cronológico global

0% ----------------------------- 60% ----------------------------- 100%
      entrenamiento IBPR base R900          stream definitivo H1-H3
      + horizonte de desarrollo HPO         sin retuning posterior
```

El tramo `0%-60%` se utiliza para construir el IBPR inicial definitivo con la configuración R900.

El tramo `60%-100%` constituye el stream temporal utilizado para la comparación definitiva H1-H3.

Este tramo posterior no debe llamarse “holdout completamente virgen” debido a observaciones realizadas durante pilotos anteriores; debe describirse como evaluación con configuración completamente congelada y sin retuning posterior.

### 16.2 Ramas experimentales

La comparación principal será:

```text
                 mismo IBPR inicial R900
                        │
          ┌─────────────┼──────────────┐
          │             │              │
          ▼             ▼              ▼
      IBPR_STALE   ONLINE_IBPR    FULL_RETRAIN
          │             │              │
   no cambia U,V   adapta U         reentrena
                    V fija           U y V
```

Se utilizarán exactamente las mismas nuevas interacciones, reglas warm-start y puntos de evaluación para los tres métodos.

El protocolo definitivo deberá ser **global y prequential**, a diferencia de los folds/escenarios por usuario utilizados durante HPO.

### 16.3 Configuraciones congeladas

```text
IBPR base = R900
OnlineIBPRMejorado = O014
```

No se permitirá modificar estas configuraciones utilizando resultados de H1-H4.

### 16.4 Seeds reservadas

Las seeds reservadas fuera del HPO online para esta etapa continúan siendo:

```text
[777, 999]
```

Esta decisión se mantiene como parte del protocolo congelado mientras no exista una modificación metodológica explícita previa a la ejecución.

### 16.5 Interpretación de H3

Además del delta directo:

```text
Online - Full Retrain
```

se podrá registrar como métrica descriptiva complementaria la fracción de adaptación recuperada frente a Stale:

```text
adaptation_recovery =
(Online - Stale) / (FullRetrain - Stale)
```

aplicada a `NDCG@20` cuando el denominador sea positivo y suficientemente definido.

Esta razón no constituye una prueba de equivalencia. Su objetivo es expresar qué proporción de la mejora observada mediante Full Retrain es recuperada por la adaptación online.

La expresión “calidad competitiva” deberá sustentarse en:

- diferencia absoluta y relativa de NDCG@20;
- deltas pareados por seed y punto temporal;
- métricas secundarias;
- coste computacional asociado.

No se afirmará equivalencia estadística salvo que se diseñe específicamente una prueba formal de equivalencia.

---

## 17. Métricas definitivas

## Calidad

Principal:

```text
NDCG@20
```

Secundarias:

```text
Recall@20
Precision@20
MAP
AUC
```

---

## Adaptación

Se deben estudiar deltas pareados:

```text
Online - Stale
Online - Full Retrain
```

preferentemente por seed y por punto temporal.

Como medida descriptiva complementaria para H3 podrá calcularse:

```text
adaptation_recovery =
(Online - Stale) / (FullRetrain - Stale)
```

cuando `FullRetrain - Stale > 0`.

---

## Eficiencia

```text
tiempo de partial update
tiempo de full retraining
coste acumulado
speedup
fracción de coste online/full
```

---

## Indexabilidad

```text
V exact equal
max abs diff de V
rebuild count
index build/rebuild time
indexed vs exhaustive agreement@20
indexed vs exhaustive recall@20
query mean ms
query median ms
query p95 ms
```

---

## 18. Orden experimental oficial

```text
1. Implementación OnlineIBPRMejorado                 COMPLETADO
2. Tests de invariantes                               COMPLETADO
3. Experimentos piloto                                COMPLETADO
4. Análisis multi-seed piloto                         COMPLETADO
5. Figuras preliminares reproducibles                 COMPLETADO
6. HPO del IBPR base                                  COMPLETADO
   6.1 búsqueda inicial                               COMPLETADO
   6.2 refinamiento local                             COMPLETADO
   6.3 confirmación multi-seed                        COMPLETADO
   6.4 congelación del IBPR base                      COMPLETADO

7. HPO de OnlineIBPRMejorado                          COMPLETADO
   7.1 plan validado                                  COMPLETADO
   7.2 screening O-A                                  COMPLETADO
   7.3 robustez temporal O-B                          COMPLETADO
   7.4 confirmación multi-seed O-C                    COMPLETADO

8. Congelar configuración online                      COMPLETADO
   8.1 configuración seleccionada O014                COMPLETADO
   8.2 prohibición de retuning con H1-H4              ACTIVA

9. Experimento definitivo H1-H3                       ETAPA ACTUAL
   9.1 definir/validar protocolo global prequential   PENDIENTE
   9.2 IBPR Stale                                     PENDIENTE
   9.3 OnlineIBPRMejorado                             PENDIENTE
   9.4 IBPR Full Retrain                              PENDIENTE
   9.5 calidad                                        PENDIENTE
   9.6 coste                                          PENDIENTE
   9.7 análisis pareado multi-seed                    PENDIENTE

10. H4 — reutilización del índice                     PENDIENTE
    10.1 validar V exacta
    10.2 construir índice una vez
    10.3 reutilizarlo después de partial updates
    10.4 comparar recuperación indexada/exhaustiva
    10.5 medir latencias
    10.6 comparar rebuild requerido por full retrain

11. Ablaciones focalizadas                            PENDIENTE
    Sólo las necesarias para explicar el método.

12. Validación MovieLens 10M                          PENDIENTE
    Configuración congelada.
    Sin nuevo HPO.

13. Evaluación de ciclo híbrido offline-online (H5)   CONDICIONAL
    Ejecutar si H1-H4 justifican la hipótesis.
    Comparar políticas de actualización.

14. Cierre experimental                               PENDIENTE

15. Redacción final de tesis                          PENDIENTE
```

---

## 19. Experimento opcional H5 para demostrar directamente el sistema híbrido

Si se desea que la conclusión final diga de forma explícita:

> “OnlineIBPRMejorado puede utilizarse como componente de una estrategia híbrida con IBPR”

conviene realizar un último experimento de sistema después de H1-H4.

## Políticas a comparar

### Política A — Stale

```text
IBPR inicial
sin actualización
```

### Política B — Full retrain frecuente

```text
IBPR inicial
full retrain después de cada bloque
```

Sirve como referencia de máxima actualización con alto coste.

### Política C — Online entre retrainings

```text
IBPR inicial
partial
partial
partial
...
```

### Política D — Híbrida

Ejemplo conceptual:

```text
full IBPR
-> partial
-> partial
-> partial
-> full IBPR
-> partial
-> partial
-> partial
-> full IBPR
...
```

El intervalo exacto no debe decidirse usando el resultado final a conveniencia.

Puede predefinirse o evaluarse como una pequeña ablación de frecuencia.

## Métricas del sistema híbrido

```text
NDCG@20 medio a lo largo del stream
NDCG@20 por punto temporal
coste acumulado
número de full retrains
número de index rebuilds
tiempo acumulado de rebuild
latencia top-k
```

Una política híbrida sería especialmente interesante si ocupa una región favorable del trade-off:

```text
calidad cercana a full retrain frecuente
+
coste mucho menor
+
pocos rebuilds
```

Este experimento convertiría la “complementariedad” en una afirmación evaluada directamente a nivel de sistema.

---

## 20. Ablaciones permitidas

Las ablaciones deben responder preguntas concretas.

Candidatas:

```text
update_V=False vs True
angular vs cosine_bpr
n_epochs
max_steps, sólo si se quiere estudiar calidad/coste
frecuencia de full retrain en H5
```

No se debe transformar la fase de ablaciones en otro HPO encubierto.

---

## 21. MovieLens 10M

MovieLens 10M se utilizará después de congelar todas las decisiones principales.

Objetivo:

- estudiar escala;
- comprobar que la propuesta no depende exclusivamente de MovieLens 1M;
- medir costes mayores;
- observar comportamiento de recuperación/indexación con mayor volumen.

No se volverá a optimizar el modelo sobre 10M salvo que explícitamente se cambie el diseño metodológico de la tesis.

La configuración proveniente de desarrollo debe trasladarse congelada.

---

## 22. Qué no forma parte del objetivo actual

No es objetivo principal demostrar:

- cold-start;
- recomendación basada en contenido;
- fusión de modelos heterogéneos;
- universalidad sobre cualquier dataset;
- un óptimo universal de hiperparámetros;
- que OnlineIBPRMejorado reemplaza completamente al entrenamiento offline;
- que todo full retraining es innecesario;
- equivalencia estadística con full retraining sin una prueba específica;
- que `update_V=False` sea siempre mejor que `update_V=True`.

---

## 23. Regla para interpretar los resultados

La interpretación deberá seguir este orden:

```text
1. ¿Online realmente adapta mejor que Stale?
2. ¿Cuánto cuesta esa adaptación?
3. ¿Qué calidad conserva frente a Full Retrain?
4. ¿V realmente permanece fija?
5. ¿El índice puede reutilizarse?
6. ¿La combinación offline + online tiene sentido como política de sistema?
7. ¿Los resultados se sostienen al aumentar escala?
```

No se debe seleccionar primero una conclusión y buscar después métricas que la sostengan.

---

## 24. Contribución esperada

La contribución técnica potencial de la investigación puede resumirse así:

> Se propone y evalúa una extensión incremental de Indexable BPR que permite incorporar nuevas interacciones mediante warm-start y actualización parcial de factores de usuario, manteniendo estáticos los factores de ítems cuando se desea preservar la estructura de indexación. La propuesta se estudia en términos de adaptación temporal, calidad de ranking, coste computacional y reutilización del índice, y se analiza su posible integración como componente online entre reentrenamientos periódicos del modelo IBPR base.

Esta formulación es deliberadamente prudente:

- describe lo implementado;
- especifica lo que se evaluará;
- no presupone que todas las hipótesis serán confirmadas.

---

## 25. Tesis principal recomendada en este momento

Mientras no existan los resultados definitivos, la tesis de trabajo recomendada es:

> **OnlineIBPRMejorado no se plantea como sustituto de IBPR, sino como una capa de adaptación incremental complementaria. Un IBPR offline proporciona y periódicamente renueva la representación global y el índice de ítems, mientras que OnlineIBPRMejorado adapta los factores de usuario a interacciones recientes entre esos reentrenamientos. Si la evidencia experimental confirma una mejora frente al modelo stale, un coste muy inferior al full retraining y la reutilización del índice, esta combinación constituye una estrategia híbrida offline-online para recomendación top-k adaptable y eficiente.**

---

## 26. Fuentes de verdad del proyecto

Para continuar el proyecto, deben considerarse vigentes los siguientes tipos de archivo.

## Fundamentos

```text
Bayesian Personalized Ranking .pdf
Indexable Bayesian personalized ranking for efficient top-k recom.pdf
```

## IBPR base

```text
cornac/models/ibpr/ibpr.py
cornac/models/ibpr/recom_ibpr.py
```

## Implementación OnlineIBPR original auditada

```text
cornac/models/online_ibpr/online_ibpr.py
cornac/models/online_ibpr/recom_online_ibpr.py
```

Estos archivos deben conservarse como evidencia de la implementación original auditada y de los problemas técnicos documentados en la sección 4.

## OnlineIBPRMejorado

```text
cornac/models/online_ibpr_mejorado/online_ibpr_mejorado.py
cornac/models/online_ibpr_mejorado/recom_online_ibpr_mejorado.py
```

## Corrección funcional

```text
tests/online_ibpr_mejorado/test_online_ibpr_mejorado_invariants.py
```

## Decisión congelada del IBPR base

```text
tests/online_ibpr_mejorado/
hyperparameter_refinement_ibpr_base_resultados_y_decision.md
```

Configuración vigente: `R900`.

## HPO online cerrado

Plan y script vigentes:

```text
tests/online_ibpr_mejorado/
plan_hpo_online_ibpr_mejorado_validated.md

tests/online_ibpr_mejorado/
hyperparameter_search_online_ibpr_mejorado.py
```

Resultados de cierre que deben conservarse:

```text
tests/online_ibpr_mejorado/results/
hyperparameter_search_online_ibpr_mejorado_20260908_214938.txt

tests/online_ibpr_mejorado/results/
hyperparameter_search_online_ibpr_mejorado_trials_20260908_214938.csv

tests/online_ibpr_mejorado/results/
hyperparameter_search_online_ibpr_mejorado_chunks_20260908_214938.csv

tests/online_ibpr_mejorado/results/
hyperparameter_search_online_ibpr_mejorado_summary_20260908_214938.csv

tests/online_ibpr_mejorado/results/
hyperparameter_search_online_ibpr_mejorado_best_20260908_214938.csv
```

Configuración vigente: `O014`.

## Este documento

Nombre recomendado:

```text
tests/online_ibpr_mejorado/
CONTEXTO_MAESTRO_Y_HOJA_DE_RUTA_ONLINE_IBPR_MEJORADO.md
```

---

## 27. Regla de actualización de este documento

Este documento debe modificarse únicamente cuando se cierre una etapa importante.

Ejemplos:

```text
HPO Online finalizado
-> agregar configuración online congelada y resultado.

H1-H3 finalizados
-> agregar conclusiones aceptadas/rechazadas.

H4 finalizado
-> agregar evidencia de index reuse.

H5 realizado
-> actualizar conclusión sobre sistema híbrido.

MovieLens 10M finalizado
-> agregar conclusión de escala/generalización.
```

No debe utilizarse para almacenar todos los resultados intermedios.

Los resultados detallados deben permanecer en sus CSV, scripts y documentos específicos.

---

## 28. Próximo paso exacto

El proyecto se encuentra actualmente en:

```text
EXPERIMENTO DEFINITIVO H1-H3
```

Los hiperparámetros ya no deben modificarse.

Configuraciones congeladas:

```text
IBPR base = R900
OnlineIBPRMejorado = O014
```

El siguiente trabajo consiste en diseñar e implementar el experimento definitivo global/prequential que compare:

```text
IBPR_STALE
vs
OnlineIBPRMejorado
vs
IBPR_FULL_RETRAIN
```

Los tres métodos deben:

```text
- partir del mismo IBPR inicial;
- recibir exactamente las mismas nuevas interacciones;
- evaluarse en exactamente los mismos puntos temporales;
- utilizar las mismas reglas de filtrado warm-start;
- registrar las mismas métricas;
- ejecutarse con configuración completamente congelada.
```

Seeds reservadas para esta etapa:

```text
[777, 999]
```

Antes de ejecutar el experimento completo se debe disponer de un modo de planificación/validación equivalente a `--plan-only` que permita revisar, como mínimo:

```text
- frontera cronológica global 0%-60% / 60%-100%;
- tamaño y límites del tramo base;
- tamaño y cronología del stream futuro;
- chunks o puntos de evaluación;
- usuarios e ítems warm-start retenidos;
- configuración congelada de R900;
- configuración congelada de O014;
- seeds;
- ramas Stale / Online / Full Retrain;
- métricas de calidad;
- métricas de coste;
- deltas pareados que serán calculados;
- regla descriptiva de adaptation_recovery para H3.
```

Una vez validado el plan:

```text
ejecutar H1-H3
-> analizar Online - Stale
-> analizar Online - Full Retrain
-> analizar coste partial vs full retrain
-> cerrar H1-H3 sin retuning
-> pasar a H4
```

Ese es el punto exacto desde el cual debe continuar la investigación.
