# CONTEXTO MAESTRO V2

# OnlineIBPRMejorado --- Estrategia Científica y Ruta Experimental Definitiva

## 1. Propósito del documento

Este documento define la dirección científica, metodológica y
experimental del proyecto OnlineIBPRMejorado.

Su objetivo es establecer un marco único para mantener el foco en la
contribución principal, evitar la expansión no controlada de
experimentos y garantizar reproducibilidad.

Los experimentos anteriores se consideran exploraciones preliminares
hasta que sean validados mediante el nuevo protocolo.

## 2. Problema científico

Los sistemas de recomendación basados en modelos latentes suelen
requerir reentrenamientos completos cuando llegan nuevas interacciones.

OnlineIBPRMejorado propone una adaptación incremental sobre un modelo
IBPR previamente entrenado.

## 3. Objetivo principal

Demostrar que OnlineIBPRMejorado permite adaptar un modelo IBPR ante
nuevas interacciones de usuarios conocidos, reduciendo el coste respecto
al reentrenamiento completo, manteniendo calidad competitiva y
conservando propiedades necesarias para recuperación eficiente.

## 4. Preguntas de investigación

### RQ1 --- Adaptación incremental

¿OnlineIBPRMejorado permite recuperar información nueva respecto a
mantener un modelo IBPR sin actualización?

Comparación:

IBPR_STALE vs OnlineIBPRMejorado

### RQ2 --- Eficiencia computacional

¿La actualización incremental requiere menor coste que un
reentrenamiento completo?

Comparación:

Actualización Online vs Full Retrain

### RQ3 --- Conservación de indexabilidad

¿La actualización mantiene fija la representación de ítems permitiendo
reutilizar el mecanismo de recuperación?

Condición:

V_base == V_online

## 5. Política experimental

Cada experimento debe responder una pregunta científica necesaria.

No se agregan experimentos solamente porque sean técnicamente posibles.

## 6. Política de rehacer experimentos

Los experimentos pueden rehacerse si mejora:

-   rigor metodológico;
-   reproducibilidad;
-   claridad;
-   ausencia de ambigüedades.

No se rehacen para buscar mejores números o eliminar resultados
desfavorables.

## 7. Camino crítico

1.  Validación de implementación.
2.  Selección reproducible de modelos.
3.  Evaluación principal.
4.  Validación de indexabilidad.
5.  Conclusión científica.

## 8. Evidencia principal

### Validación de implementación

Debe demostrar:

-   actualización correcta;
-   conservación de factores de ítems;
-   reproducibilidad;
-   consistencia de identificadores.

### Evaluación principal

Comparación:

-   IBPR_STALE
-   OnlineIBPRMejorado
-   IBPR_FULL_RETRAIN

### Reutilización del índice

Debe demostrar:

-   V permanece constante;
-   el índice puede reutilizarse;
-   no requiere reconstrucción.

## 9. Selección de modelos

Las configuraciones actuales no se consideran definitivas.

Se obtendrán:

-   IBPR_FINAL
-   OnlineIBPRMejorado_FINAL

mediante un proceso reproducible.

Una vez congeladas no se modifican durante la evaluación final.

## 10. Métricas

Calidad:

-   NDCG@20
-   Recall@20
-   Precision@20
-   MAP
-   AUC

Eficiencia:

-   tiempo de entrenamiento inicial;
-   tiempo de actualización incremental;
-   tiempo de reentrenamiento completo;
-   speedup.

## 11. Estructura esperada

    online_ibpr_mejorado/

    docs/
     ├── CONTEXTO_MAESTRO_V2.md
     ├── METODOLOGIA.md
     └── PROTOCOLO_EXPERIMENTAL.md

    validation/

    experiments/
     ├── selection/
     ├── final/
     └── index/

    support/

    archive/

## 12. Criterio de cierre

La investigación estará completa cuando pueda responder:

1.  ¿OnlineIBPRMejorado adapta mejor que mantener IBPR estático?
2.  ¿La adaptación requiere menos coste que reentrenar completamente?
3.  ¿La actualización mantiene la propiedad indexable?
