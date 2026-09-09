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

# 2. Idea central de la investigación

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

# 3. Fundamento teórico

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

# 4. Problema técnico identificado

Cornac incluye una implementación denominada `OnlineIBPR`, conceptualmente orientada a actualización online.

La investigación detectó que esa implementación no construía correctamente las tripletas `(u, i, j)` requeridas por el aprendizaje ordinal.

En particular, la información almacenada como valor de interacción podía terminar utilizándose como si fuera el identificador del ítem negativo `j`.

Además, la implementación original no proporcionaba un mecanismo explícito y controlado para:

- recibir un lote reciente de positivos `(u, i)`;
- mantener un historial acumulado;
- impedir que positivos históricos o recientes sean seleccionados como negativos;
- realizar warm-start desde un IBPR previamente entrenado;
- controlar si los factores de ítems `V` deben permanecer fijos;
- realizar sucesivas actualizaciones parciales reproducibles.

---

# 5. Solución implementada: OnlineIBPRMejorado

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

# 6. Por qué mantener V fijo es una decisión central

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

# 7. Pregunta principal de investigación

La pregunta principal queda formulada como:

> **¿Puede un modelo Indexable BPR previamente entrenado adaptarse incrementalmente a nuevas interacciones de usuarios conocidos, manteniendo estáticos los factores de ítems, con un coste computacional sustancialmente inferior al reentrenamiento completo y conservando una calidad de recomendación competitiva?**

Una pregunta de nivel sistema deriva de la anterior:

> **¿Puede esta adaptación incremental actuar como complemento entre reentrenamientos periódicos de IBPR, permitiendo un ciclo híbrido offline-online que reduzca la obsolescencia del recomendador sin reconstruir continuamente el índice de ítems?**

La segunda pregunta debe considerarse una consecuencia experimental a validar, no una conclusión predeterminada.

---

# 8. Hipótesis experimentales principales

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

# 9. Hipótesis de sistema híbrido

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

# 10. Posibles conclusiones de tesis según los resultados

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

# 11. Estado validado de la implementación

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

# 12. IBPR base: configuración congelada

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

# 13. Etapa actual: HPO de OnlineIBPRMejorado

El siguiente paso inmediato es seleccionar **únicamente los parámetros propios de adaptación online**.

El IBPR base permanece congelado.

## Parámetros fijos del modo online principal

```text
k            = 20
update_V     = False
neg_sampling = uniform
normalize    = True
max_steps    = None
```

## Parámetros candidatos

```python
learning_rate = [0.001, 0.0025, 0.005, 0.01, 0.02]
lamda         = [0.0, 1e-6, 1e-5, 1e-4, 1e-3]
batch_size    = [128, 256, 512, 1024]
n_epochs      = [1, 2, 3]
loss_mode     = ["cosine_bpr", "angular"]
```

## Métrica primaria

```text
mean ΔNDCG@20 =
OnlineIBPRMejorado - IBPR_STALE
```

La selección no utiliza Full Retrain como objetivo del HPO.

---

# 14. Datos utilizados durante HPO online

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

# 15. Advertencia metodológica sobre MovieLens 1M

El 40% global posterior está excluido del **HPO actual**.

Sin embargo, durante etapas piloto anteriores del proyecto ya se observaron resultados sobre particiones posteriores de MovieLens 1M.

Por tanto, ese segmento no debe describirse como un “holdout completamente virgen” o “nunca observado”.

La formulación más correcta para los experimentos posteriores será:

> **evaluación con configuración completamente congelada y sin utilizar esos resultados para volver a ajustar hiperparámetros.**

La defensa metodológica debe apoyarse en:

- congelación previa de la configuración;
- ausencia de retuning posterior;
- evaluación multi-seed;
- posterior validación de escala/generalización en MovieLens 10M.

---

# 16. Comparación experimental definitiva

Después del HPO online, se congela la configuración completa.

La comparación principal será:

```text
                 mismo IBPR inicial
                        │
          ┌─────────────┼──────────────┐
          │             │              │
          ▼             ▼              ▼
      IBPR_STALE   ONLINE_IBPR    FULL_RETRAIN
          │             │              │
   no cambia U,V   adapta U         reentrena
                    V fija           U y V
```

Se utilizarán exactamente las mismas nuevas interacciones y puntos de evaluación para los tres métodos.

---

# 17. Métricas definitivas

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

# 18. Orden experimental oficial

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

7. HPO de OnlineIBPRMejorado                          ETAPA ACTUAL

8. Congelar configuración online                      PENDIENTE

9. Experimento definitivo H1-H3                       PENDIENTE
   9.1 IBPR Stale
   9.2 OnlineIBPRMejorado
   9.3 IBPR Full Retrain
   9.4 calidad
   9.5 coste
   9.6 análisis pareado multi-seed

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

# 19. Experimento opcional H5 para demostrar directamente el sistema híbrido

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

# 20. Ablaciones permitidas

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

# 21. MovieLens 10M

MovieLens 10M se utilizará después de congelar todas las decisiones principales.

Objetivo:

- estudiar escala;
- comprobar que la propuesta no depende exclusivamente de MovieLens 1M;
- medir costes mayores;
- observar comportamiento de recuperación/indexación con mayor volumen.

No se volverá a optimizar el modelo sobre 10M salvo que explícitamente se cambie el diseño metodológico de la tesis.

La configuración proveniente de desarrollo debe trasladarse congelada.

---

# 22. Qué no forma parte del objetivo actual

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

# 23. Regla para interpretar los resultados

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

# 24. Contribución esperada

La contribución técnica potencial de la investigación puede resumirse así:

> Se propone y evalúa una extensión incremental de Indexable BPR que permite incorporar nuevas interacciones mediante warm-start y actualización parcial de factores de usuario, manteniendo estáticos los factores de ítems cuando se desea preservar la estructura de indexación. La propuesta se estudia en términos de adaptación temporal, calidad de ranking, coste computacional y reutilización del índice, y se analiza su posible integración como componente online entre reentrenamientos periódicos del modelo IBPR base.

Esta formulación es deliberadamente prudente:

- describe lo implementado;
- especifica lo que se evaluará;
- no presupone que todas las hipótesis serán confirmadas.

---

# 25. Tesis principal recomendada en este momento

Mientras no existan los resultados definitivos, la tesis de trabajo recomendada es:

> **OnlineIBPRMejorado no se plantea como sustituto de IBPR, sino como una capa de adaptación incremental complementaria. Un IBPR offline proporciona y periódicamente renueva la representación global y el índice de ítems, mientras que OnlineIBPRMejorado adapta los factores de usuario a interacciones recientes entre esos reentrenamientos. Si la evidencia experimental confirma una mejora frente al modelo stale, un coste muy inferior al full retraining y la reutilización del índice, esta combinación constituye una estrategia híbrida offline-online para recomendación top-k adaptable y eficiente.**

---

# 26. Fuentes de verdad del proyecto

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

## HPO online actual

```text
tests/online_ibpr_mejorado/
plan_hpo_online_ibpr_mejorado_validated.md

tests/online_ibpr_mejorado/
hyperparameter_search_online_ibpr_mejorado.py
```

## Este documento

Nombre recomendado:

```text
tests/online_ibpr_mejorado/
CONTEXTO_MAESTRO_Y_HOJA_DE_RUTA_ONLINE_IBPR_MEJORADO.md
```

---

# 27. Regla de actualización de este documento

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

# 28. Próximo paso exacto

El proyecto se encuentra actualmente en:

```text
HPO de OnlineIBPRMejorado
```

Antes de ejecutar el HPO completo:

```bash
python tests/online_ibpr_mejorado/hyperparameter_search_online_ibpr_mejorado.py --plan-only
```

Después de validar el plan generado:

```text
ejecutar HPO
-> seleccionar configuración online
-> congelarla
-> no volver a utilizar H1-H4 para retuning
```

Ese es el punto exacto desde el cual debe continuar la investigación.
