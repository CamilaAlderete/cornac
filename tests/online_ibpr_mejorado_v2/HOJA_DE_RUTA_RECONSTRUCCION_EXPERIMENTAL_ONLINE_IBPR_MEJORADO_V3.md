# Hoja de ruta de reconstrucción experimental V3

# OnlineIBPRMejorado

## 1. Principio rector

El objetivo no es conservar resultados experimentales anteriores.

El objetivo es reconstruir una demostración científica reproducible
utilizando el código existente como base técnica.

Los resultados históricos serán descartados como evidencia oficial.

Los experimentos existentes serán utilizados para comprender decisiones
previas, detectar problemas y reconstruir el protocolo definitivo.

------------------------------------------------------------------------

# 2. Camino científico

    Análisis de contribución

    ↓

    Auditoría del código

    ↓

    Auditoría de experimentos existentes

    ↓

    Rediseño de pruebas necesarias

    ↓

    Nueva ejecución experimental

    ↓

    Análisis científico

    ↓

    Conclusión

------------------------------------------------------------------------

# 3. FASE 0 --- Definición de contribución

Identificar la evolución:

    IBPR original

    ↓

    OnlineIBPR original

    ↓

    OnlineIBPRMejorado

Para cada modificación:

-   problema;
-   solución propuesta;
-   hipótesis;
-   test necesario.

------------------------------------------------------------------------

# 4. FASE 1 --- Auditoría del código fuente

Analizar:

-   IBPR;
-   OnlineIBPR;
-   OnlineIBPRMejorado.

Verificar:

-   actualización de parámetros;
-   función objetivo;
-   sampling;
-   normalización;
-   manejo de factores;
-   compatibilidad con recuperación.

------------------------------------------------------------------------

# 5. FASE 2 --- Auditoría de tests existentes

Cada test será evaluado:

    Test

    ↓

    Pregunta científica

    ↓

    Validez

    ↓

    Acción

Acciones:

-   KEEP
-   REFACTOR
-   RECREATE
-   MERGE
-   SUPPORT
-   ARCHIVE
-   DELETE

------------------------------------------------------------------------

# 6. FASE 3 --- Construcción del protocolo experimental

Baselines:

## IBPR_STALE

Modelo sin actualización.

## OnlineIBPRMejorado

Actualización incremental.

## IBPR_FULL_RETRAIN

Reentrenamiento completo.

------------------------------------------------------------------------

# 7. FASE 4 --- Selección reproducible de modelos

Ejecutar selección necesaria para obtener:

    IBPR_FINAL

    OnlineIBPRMejorado_FINAL

Luego congelar configuraciones.

------------------------------------------------------------------------

# 8. FASE 5 --- Ejecución experimental

Validación de implementación:

-   actualización correcta;
-   V constante;
-   reproducibilidad;
-   consistencia.

Evaluación principal:

    IBPR_STALE

    vs

    OnlineIBPRMejorado

    vs

    IBPR_FULL_RETRAIN

Validación de índice:

    V_base == V_online

    ↓

    índice reutilizable

------------------------------------------------------------------------

# 9. FASE 6 --- Análisis de resultados

Responder:

-   ¿mejora adaptación?
-   ¿reduce coste?
-   ¿qué trade-off existe?
-   ¿qué modificación aporta realmente?

------------------------------------------------------------------------

# 10. FASE 7 --- Documentación final

    docs/

    PROBLEMA.md

    CONTRIBUCION.md

    ANALISIS_ALGORITMO.md

    PROTOCOLO_EXPERIMENTAL.md

    RESULTADOS.md

    LIMITACIONES.md

------------------------------------------------------------------------

# 11. Criterio de cierre

El proyecto estará cerrado cuando:

-   exista una ejecución experimental reproducible;
-   las afirmaciones tengan evidencia;
-   las modificaciones estén justificadas;
-   las limitaciones estén documentadas;
-   no existan experimentos adicionales necesarios.
