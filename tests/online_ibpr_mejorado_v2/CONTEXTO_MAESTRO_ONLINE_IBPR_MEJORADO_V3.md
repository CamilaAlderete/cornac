# CONTEXTO MAESTRO V3

# OnlineIBPRMejorado --- Estrategia Científica y Reconstrucción Experimental

## 1. Propósito del documento

Este documento define la dirección científica, metodológica y
experimental del proyecto OnlineIBPRMejorado.

El objetivo es establecer un marco único para demostrar la contribución
científica del modelo propuesto mediante una nueva ejecución
experimental reproducible.

Los códigos fuente, implementaciones y experimentos desarrollados
previamente se consideran material técnico de partida.

Los resultados históricos no se consideran evidencia científica
definitiva.

La evidencia oficial será generada nuevamente mediante un protocolo
experimental revisado.

------------------------------------------------------------------------

## 2. Principio científico

El objetivo del proyecto no es demostrar únicamente que
OnlineIBPRMejorado obtiene mejores métricas.

El objetivo es demostrar que las modificaciones introducidas sobre IBPR
permiten construir un componente incremental capaz de adaptar un modelo
previamente entrenado, reduciendo el coste de actualización y
conservando propiedades necesarias para recuperación eficiente.

------------------------------------------------------------------------

## 3. Problema científico

Los sistemas de recomendación basados en modelos latentes normalmente
requieren reentrenamientos completos cuando aparecen nuevas
interacciones.

Esto implica mayor coste computacional, mayor tiempo de actualización y
posible reconstrucción de estructuras auxiliares.

OnlineIBPRMejorado propone una adaptación incremental sobre un modelo
IBPR previamente entrenado.

------------------------------------------------------------------------

## 4. Evolución del modelo

### 4.1 IBPR original

IBPR representa el modelo base.

Debe analizarse:

-   entrenamiento completo;
-   actualización de factores;
-   función objetivo;
-   sampling;
-   generación de ranking.

Limitación identificada:

Las nuevas interacciones pueden requerir un proceso completo de
actualización del modelo.

------------------------------------------------------------------------

### 4.2 OnlineIBPR original

OnlineIBPR representa una primera aproximación hacia actualización
incremental.

Debe analizarse qué problema resuelve y qué limitaciones conserva.

------------------------------------------------------------------------

### 4.3 OnlineIBPRMejorado

Cada modificación debe documentarse mediante:

Problema identificado:

↓

Modificación realizada:

↓

Hipótesis esperada:

↓

Experimento de validación:

------------------------------------------------------------------------

## 5. Preguntas de investigación

### RQ1 --- Adaptación incremental

¿OnlineIBPRMejorado permite recuperar información nueva respecto a
mantener un modelo IBPR sin actualización?

Comparación:

IBPR_STALE vs OnlineIBPRMejorado

------------------------------------------------------------------------

### RQ2 --- Eficiencia computacional

¿La actualización incremental requiere menor coste que un
reentrenamiento completo?

Comparación:

Online Update vs IBPR_FULL_RETRAIN

------------------------------------------------------------------------

### RQ3 --- Conservación de indexabilidad

¿La actualización mantiene fija la representación de ítems permitiendo
reutilizar el mecanismo de recuperación?

Condición:

V_base == V_online

------------------------------------------------------------------------

## 6. Validación de modificaciones

Cada modificación debe relacionarse con una hipótesis y una evidencia
experimental.

Ejemplo:

  Modificación              Hipótesis              Evidencia
  ------------------------- ---------------------- --------------------------
  Actualización parcial     menor coste            comparación temporal
  Conservación de V         reutilización índice   igualdad V
  Cambios de optimización   estabilidad            comparación experimental
  Cambios de sampling       adaptación             métricas ranking

------------------------------------------------------------------------

## 7. Selección de modelos

Las configuraciones existentes serán consideradas candidatas.

Se realizará una selección reproducible de:

-   IBPR_FINAL
-   OnlineIBPRMejorado_FINAL

Una vez seleccionadas, las configuraciones quedan congeladas durante la
evaluación final.

------------------------------------------------------------------------

## 8. Criterio de finalización

La investigación estará completa cuando pueda responder:

1.  ¿OnlineIBPRMejorado adapta mejor que mantener IBPR estático?
2.  ¿La adaptación requiere menor coste que reentrenar completamente?
3.  ¿La actualización mantiene la propiedad indexable?
4.  ¿Las modificaciones introducidas están justificadas mediante
    evidencia experimental?
