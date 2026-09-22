# Hoja de ruta de consolidación V2 --- OnlineIBPRMejorado

## 1. Principio rector

A partir de ahora, el objetivo ya no es generar más experimentos.

El objetivo es:

> Conservar, reconstruir o fusionar únicamente los experimentos
> necesarios para demostrar, reproducir y defender la contribución
> central de OnlineIBPRMejorado.

Un experimento existente puede clasificarse como:

-   KEEP
-   REFACTOR
-   MERGE
-   RECREATE
-   SUPPORT
-   ARCHIVE
-   DELETE

Rehacer un test está permitido únicamente si mejora claridad,
reproducibilidad o rigor científico.

No está permitido rehacerlo para buscar resultados más convenientes.

------------------------------------------------------------------------

# 2. Relación con el Contexto Maestro V2

Esta hoja de ruta no define la investigación. El Contexto Maestro V2
define:

-   el problema científico;
-   la contribución;
-   las preguntas de investigación;
-   el alcance.

Esta hoja define cómo consolidar el proyecto técnico y experimental
existente.

------------------------------------------------------------------------

# 3. Tesis central

OnlineIBPRMejorado es una extensión incremental de IBPR que permite
incorporar nuevas interacciones de usuarios conocidos mediante
actualización parcial, reduciendo el coste computacional respecto al
reentrenamiento completo y preservando los factores de ítems cuando se
desea mantener reutilizable el índice top-k.

La propuesta se orienta como componente online complementario entre
ciclos periódicos de entrenamiento offline.

------------------------------------------------------------------------

# 4. Preguntas que deben quedar demostradas

## Q1 --- ¿La implementación funciona correctamente?

Debe demostrarse mediante invariantes:

-   warm-start;
-   actualización de usuarios;
-   V fija con update_V=False;
-   negative sampling válido;
-   update vacío = identidad;
-   reproducibilidad;
-   progresión de seeds;
-   mappings coherentes;
-   ausencia de leakage.

------------------------------------------------------------------------

## Q2 --- ¿Qué calidad/coste ofrece frente a IBPR?

Comparación principal:

    IBPR_STALE

    vs

    OnlineIBPRMejorado

    vs

    IBPR_FULL_RETRAIN

Evaluar:

Calidad:

-   NDCG@20
-   Recall@20
-   Precision@20
-   MAP
-   AUC

Eficiencia:

-   tiempo de actualización incremental;
-   tiempo de entrenamiento completo;
-   coste acumulado;
-   speedup.

No se requiere que OnlineIBPRMejorado supere siempre a Full Retrain.

El objetivo es demostrar un equilibrio calidad/coste.

------------------------------------------------------------------------

## Q3 --- ¿Puede reutilizarse el índice?

Demostrar:

    update_V=False

            ↓

    V_online == V_base

            ↓

    mismo índice válido

            ↓

    sin reconstrucción del índice durante actualizaciones online

------------------------------------------------------------------------

# 5. Qué queda fuera del objetivo principal

No se busca demostrar:

-   Online siempre mejor que IBPR;
-   equivalencia con Full Retrain;
-   eliminación total de Full Retrain;
-   hiperparámetro óptimo universal;
-   política híbrida universal;
-   cold-start;
-   generalización universal entre datasets.

Estos puntos pertenecen a limitaciones o trabajo futuro.

------------------------------------------------------------------------

# 6. Decisiones congeladas

Se mantienen como decisiones experimentales actuales, salvo detección de
errores reales:

-   configuración IBPR seleccionada previamente;
-   configuración OnlineIBPRMejorado seleccionada previamente;
-   seeds definidas;
-   warm-start;
-   update_V=False;
-   protocolo temporal.

Estas decisiones no se modifican retrospectivamente para mejorar
resultados.

------------------------------------------------------------------------

# 7. Estado del proyecto

Los experimentos anteriores se consideran:

-   evidencia potencial;
-   diagnósticos;
-   exploraciones previas.

Antes de incorporarlos como evidencia oficial serán auditados.

El objetivo actual es:

> poda y consolidación.

------------------------------------------------------------------------

# 8. Camino oficial

    FASE 1
    Inventario de código

            ↓

    FASE 2
    Agrupar por pregunta científica

            ↓

    FASE 3
    Clasificar:
    KEEP / REFACTOR / MERGE /
    RECREATE / SUPPORT / ARCHIVE / DELETE

            ↓

    FASE 4
    Diseñar pipeline canónico

            ↓

    FASE 5
    Refactorizar o recrear lo necesario

            ↓

    FASE 6
    Ejecutar pipeline limpio

            ↓

    FASE 7
    Generar resultados finales

            ↓

    FASE 8
    Auditar evidencia

            ↓

    FASE 9
    Cierre experimental

------------------------------------------------------------------------

# 9. Pipeline objetivo

La ruta oficial esperada:

    01_validate_implementation

            ↓

    02_select_models

            ↓

    03_compare_models_final

            ↓

    04_validate_index_reuse

            ↓

    05_generate_report

Debe existir una única ruta oficial reproducible.

------------------------------------------------------------------------

# 10. Clasificación de archivos

## KEEP

Necesario y correcto.

## REFACTOR

La prueba es necesaria pero requiere limpieza.

## MERGE

Varias pruebas responden la misma pregunta.

## RECREATE

La evidencia es necesaria pero conviene reconstruirla.

## SUPPORT

Evidencia secundaria.

## ARCHIVE

Histórico, diagnósticos, versiones antiguas.

## DELETE

No aporta evidencia ni trazabilidad.

------------------------------------------------------------------------

# 11. Experimentos secundarios

## OnlineIBPR original

Se decidirá durante la auditoría.

Puede quedar como:

-   análisis comparativo;
-   soporte;
-   comparación adicional.

No forma parte obligatoria del núcleo salvo que aporte una conclusión
necesaria.

------------------------------------------------------------------------

## Ablaciones

Las ablaciones quedan como soporte.

No forman parte del pipeline principal.

------------------------------------------------------------------------

## ML10M u otros datasets

Sólo se ejecutarán si existe una necesidad real de demostrar escala o
generalización.

No se realizará HPO adicional.

------------------------------------------------------------------------

# 12. Criterio de cierre

El proyecto experimental estará cerrado cuando:

-   exista un único pipeline oficial;
-   las afirmaciones centrales tengan evidencia;
-   los resultados finales sean reproducibles;
-   las limitaciones estén documentadas;
-   todo experimento adicional quede clasificado como trabajo futuro.

Después:

    NO MÁS EXPERIMENTOS

            ↓

    TESIS

salvo corrección de errores reales.
