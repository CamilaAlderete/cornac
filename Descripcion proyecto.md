# Definición del problema e hipótesis experimental

## 1. Contexto

Indexable Bayesian Personalized Ranking (IBPR) es un algoritmo de recomendación basado en preferencias ordinales que aprende representaciones latentes normalizadas de usuarios e ítems. Estas representaciones buscan preservar la calidad de recomendación y, al mismo tiempo, ser compatibles con estructuras de indexación utilizadas para recuperar eficientemente recomendaciones top-k.

La implementación base de IBPR disponible en Cornac realiza el entrenamiento sobre tripletas `(u, i, j)`, donde `u` representa un usuario, `i` un ítem positivo y `j` un ítem negativo.

## 2. Problema identificado

Cornac contiene adicionalmente una variante denominada OnlineIBPR destinada conceptualmente a escenarios de actualización online.

Sin embargo, el análisis de su implementación muestra que la matriz de interacciones se transforma construyendo registros cuyas tres columnas corresponden al usuario, ítem y valor de feedback. Posteriormente, el valor contenido en la tercera columna es utilizado como índice del ítem negativo `j`.

Por esta razón, la implementación no construye correctamente las tripletas `(u, i, j)` requeridas por el criterio de ranking utilizado por IBPR.

Además, la implementación no proporciona un mecanismo explícito para incorporar progresivamente lotes de nuevas interacciones manteniendo el historial necesario para seleccionar negativos válidos.

## 3. Solución propuesta

Se desarrolla OnlineIBPRMejorado como una extensión orientada a la adaptación incremental de un modelo IBPR previamente entrenado.

La solución introduce un mecanismo `partial_fit_recent()` que recibe nuevas interacciones positivas `(u, i)` y utiliza el historial acumulado del usuario para seleccionar un ítem negativo `j` válido.

El modelo permite realizar warm-start utilizando los factores latentes obtenidos previamente mediante IBPR y posteriormente incorporar nuevas interacciones sin volver a entrenar necesariamente el modelo completo.

También se permite mantener los factores de ítems `V` sin modificar durante las actualizaciones online, actualizando únicamente los factores de usuario `U`.

## 4. Motivación de mantener V fijo

En IBPR, los vectores de ítems pueden ser utilizados para construir estructuras de indexación destinadas a recuperación top-k.

Si una actualización online modifica únicamente los factores de usuario mientras mantiene constantes los factores de ítems, la estructura construida sobre los ítems puede continuar siendo utilizada.

De esta manera, una nueva interacción puede modificar la representación del usuario y, por consiguiente, su consulta sobre el espacio de ítems, sin requerir necesariamente reconstruir el índice completo.

Esta propiedad constituye una de las principales motivaciones para estudiar la actualización online únicamente de los factores de usuario.

## 5. Pregunta principal de investigación

¿Puede un modelo Indexable BPR previamente entrenado adaptarse incrementalmente a nuevas interacciones de los usuarios, manteniendo estáticos los factores de los ítems, con un coste computacional significativamente menor que el reentrenamiento completo y conservando una calidad de recomendación competitiva?

## 6. Hipótesis principal

La actualización incremental de los factores de usuario de un modelo IBPR mediante nuevas interacciones permite obtener una mejor adaptación a las preferencias recientes que un modelo IBPR no actualizado, con un coste de actualización considerablemente inferior al requerido por un reentrenamiento completo.

Asimismo, al mantener constantes los factores de ítems, es posible reutilizar la estructura de indexación existente.

## 7. Hipótesis experimentales

### H1 — Adaptación

OnlineIBPRMejorado debería obtener una calidad de recomendación superior a un IBPR congelado cuando existen nuevas interacciones posteriores al entrenamiento inicial.

### H2 — Eficiencia

El coste de realizar una actualización parcial mediante OnlineIBPRMejorado debería ser sustancialmente inferior al coste de reentrenar IBPR utilizando todo el historial acumulado.

### H3 — Calidad frente al reentrenamiento

OnlineIBPRMejorado debería recuperar una parte significativa de la mejora de calidad obtenible mediante un reentrenamiento completo de IBPR.

### H4 — Reutilización del índice

Cuando se actualizan únicamente los factores de usuario y los factores de ítems permanecen constantes, el índice construido sobre los vectores de ítems debería poder continuar utilizándose sin reconstrucción.

## 8. Alcance inicial

La evaluación inicial estará restringida a usuarios e ítems previamente conocidos por el modelo base.

Por tanto, el objetivo principal no será resolver problemas de cold-start de usuarios o ítems.

El estudio se concentrará en la adaptación temporal de preferencias para entidades conocidas.

## 9. Variables principales a evaluar

La calidad de recomendación se evaluará principalmente mediante NDCG@20 y complementariamente mediante Recall@20, Precision@20, MAP y AUC.

La eficiencia se evaluará mediante tiempo de entrenamiento inicial, tiempo de actualización incremental, tiempo de reentrenamiento completo y latencia de recuperación top-k.

Cuando se evalúe el uso de índices se considerará adicionalmente el coste de construcción o reconstrucción de la estructura de indexación.

## 10. Comparación experimental principal

La evaluación principal comparará tres escenarios que parten de un mismo modelo IBPR inicial:

1. **IBPR Stale:** el modelo permanece sin modificaciones después de su entrenamiento inicial.
2. **OnlineIBPRMejorado:** el modelo incorpora progresivamente nuevas interacciones mediante actualización parcial.
3. **IBPR Full Retrain:** el modelo IBPR es reentrenado utilizando el historial completo disponible.

Esta comparación permitirá determinar simultáneamente el beneficio de adaptación, la diferencia respecto de un reentrenamiento completo y el coste computacional asociado a cada estrategia.
