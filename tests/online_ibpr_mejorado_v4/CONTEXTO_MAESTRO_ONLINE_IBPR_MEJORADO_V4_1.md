# CONTEXTO MAESTRO V4.1

# OnlineIBPRMejorado --- Marco científico y estrategia de demostración

## 1. Propósito

Este documento fija la dirección científica, metodológica y experimental del proyecto OnlineIBPRMejorado antes de reconstruir la evidencia final. La versión V4.1 es un **contrato científico de trabajo**: la FASE A solo se considerará cerrada cuando este marco sea aprobado explícitamente; hasta entonces no constituye un freeze experimental.

El objetivo no es conservar resultados históricos ni demostrar únicamente una mejora de métricas. Los códigos, pruebas y resultados existentes se consideran material técnico y diagnóstico de partida. La evidencia oficial se generará nuevamente mediante un protocolo congelado, reproducible y separado de las decisiones de desarrollo.

---

## 2. Contribución científica propuesta

OnlineIBPRMejorado se estudia como un mecanismo de adaptación incremental warm-start sobre un modelo IBPR previamente entrenado.

La contribución a demostrar es que, ante nuevas interacciones de usuarios e ítems ya conocidos, el componente incremental puede:

1. incorporar información reciente útil en las representaciones de usuario;
2. reducir el coste de actualización respecto de reentrenar completamente IBPR sobre el historial acumulado;
3. cuantificar explícitamente el trade-off de calidad respecto del reentrenamiento completo; y
4. mantener fija la representación de ítems para permitir reutilizar el índice de recuperación construido sobre esos ítems.

La contribución no incluye, en esta versión, cold-start de usuarios, incorporación de nuevos ítems ni escalabilidad industrial a catálogos masivos.

---

## 3. Alcance algorítmico

### 3.1 IBPR

IBPR es el modelo base y la referencia para entrenamiento completo. Aprende factores de usuario `U` y de ítem `V`, utiliza una función de ranking angular y expone `V` como representación de ítems para recuperación ANN.

### 3.2 OnlineIBPR original

OnlineIBPR se conserva como antecedente técnico y objeto de diagnóstico, no como baseline principal de la demostración.

La implementación auditada construye la tercera columna de sus tripletas a partir de `X.data` y después la utiliza como índice de ítem `j`. Dado que esta conducta no corresponde a un muestreo negativo BPR convencional, sus resultados no serán utilizados como evidencia principal de superioridad de OnlineIBPRMejorado.

La comparación con OnlineIBPR original podrá mantenerse únicamente como análisis de soporte para documentar las limitaciones de la implementación heredada.

### 3.3 OnlineIBPRMejorado

OnlineIBPRMejorado parte de factores `U` y `V` previamente entrenados mediante IBPR y realiza actualización parcial a partir de interacciones recientes `(u, i)`.

En el modo experimental principal:

- los usuarios e ítems deben pertenecer al universo warm-start;
- las nuevas interacciones se reciben mediante `recent_pairs`;
- el historial observado hasta el instante de actualización se utiliza para evitar positivos históricos y recientes durante el muestreo negativo;
- `U` puede adaptarse;
- `V` permanece congelada;
- el índice de recuperación se considera construido sobre `V`.

---

## 4. Contrato de implementación y propiedades a auditar

Antes de ejecutar experimentos de calidad o eficiencia, la implementación candidata debe satisfacer pruebas automáticas de contrato. La auditoría precede a cualquier modificación: primero se observa el comportamiento real del código actual; después cada hallazgo se clasifica como bug crítico, deuda de ingeniería o propiedad no necesaria para la contribución.

### 4.1 Invariantes obligatorios

1. **Warm-start**: el modo parcial requiere factores `U` y `V` previamente entrenados.
2. **V inmutable**: con `update_V=False`, `V_online` debe ser bit a bit idéntica a `V_base`.
3. **Update vacío**: una llamada sin interacciones recientes debe ser identidad y no consumir la siguiente semilla de actualización.
4. **Negative sampling válido**: un negativo nunca puede pertenecer al conjunto de positivos históricos o recientes del usuario.
5. **Reproducibilidad**: misma entrada, misma configuración y misma semilla deben producir el mismo resultado dentro del entorno congelado.
6. **Rangos y shapes**: índices y dimensiones incompatibles deben producir un error explícito.
7. **Universo fijo**: esta versión solo soporta usuarios e ítems ya conocidos por el modelo base.
8. **Casos límite explícitos**: `max_steps=0`, estrategias de sampling no implementadas y usuarios sin negativos válidos deben producir errores controlados.

Los experimentos finales no se ejecutarán si alguno de estos invariantes falla.

### 4.2 Propiedad de localidad de `U` a auditar

Se verificará si las filas de `U` correspondientes a usuarios no presentes en `recent_pairs` permanecen bit a bit idénticas después de una actualización parcial, incluyendo el caso `normalize=True`.

Esta propiedad es deseable porque facilita interpretar la operación como actualización localizada, pero **no se declara todavía requisito científico obligatorio**. Si la implementación actual la viola, se decidirá antes del freeze si el comportamiento es: (a) un bug que compromete la contribución; (b) una consecuencia numérica sin efecto sobre las hipótesis principales; o (c) una oportunidad de ingeniería fuera de alcance.

---

## 5. Preguntas de investigación e hipótesis

### RQ1 / H1 --- Adaptación incremental

**Pregunta:** ¿OnlineIBPRMejorado incorpora información reciente útil respecto de mantener el modelo IBPR sin actualización?

Comparación principal:

`IBPR_STALE` vs `OnlineIBPRMejorado`

Población principal:

usuarios warm-start que hayan recibido al menos una actualización incremental antes del punto de evaluación.

Métrica primaria:

`NDCG@20`.

Métricas secundarias:

AUC, MAP, Precision@20 y Recall@20.

La afirmación se limitará a la población efectivamente adaptada y al protocolo temporal evaluado.

### RQ2 / H2 --- Eficiencia de actualización

**Pregunta:** ¿La actualización incremental requiere menor coste computacional que reentrenar IBPR desde cero sobre el historial acumulado?

Comparación principal:

`partial_fit_recent` vs `IBPR_FULL_RETRAIN.fit`.

Se reportarán dos mediciones:

1. tiempo de actualización/entrenamiento del modelo;
2. coste inclusivo de construcción de las estructuras necesarias para cada estado experimental.

El protocolo de timing final debe fijar antes de la ejecución: mismo hardware y entorno para ambas ramas, límites de threads explícitos, mismo estado experimental de entrada, reloj monotónico de pared, exclusión del tiempo de carga inicial del dataset de la medida principal y registro separado del coste de construcción. Los cocientes de coste se calcularán de forma pareada dentro del mismo seed y punto temporal:

`speedup = T_full / T_online`.

Los tiempos se interpretarán como medidas de la **implementación real evaluada**, incluyendo sus overheads actuales; no como una complejidad asintótica ni como un benchmark industrial.

### RQ3 / H3 --- Trade-off de calidad

**Pregunta:** ¿Qué calidad conserva o pierde OnlineIBPRMejorado respecto de un reentrenamiento completo?

Comparación primaria:

`Delta_OF = OnlineIBPRMejorado - IBPR_FULL_RETRAIN`.

La diferencia directa `Delta_OF` en `NDCG@20` será el estimando principal de H3; las demás métricas serán secundarias.

Como diagnóstico complementario se podrá reportar:

`Recovery = (Online - Stale) / (Full - Stale)`.

`Recovery` no decidirá H3, no se utilizará como prueba de equivalencia y deberá presentarse junto con su denominador. Si `Full - Stale` es no positivo o demasiado cercano a cero para una interpretación estable según una regla fijada **antes** de la campaña final, el ratio se declarará no interpretable para ese punto en vez de forzarlo.

Esta RQ cuantifica el trade-off; no constituye una prueba formal de equivalencia o no-inferioridad.

### RQ4 / H4 --- Conservación de indexabilidad y reutilización del índice

**Pregunta estructural:** ¿La actualización incremental conserva exactamente la representación de ítems?

Condición obligatoria:

`V_base == V_online` bit a bit.

**Pregunta operacional:** ¿El mismo índice ANN construido sobre `V_base` puede reutilizarse después de actualizar `U` sin reconstrucción?

La evaluación distinguirá:

- `online_reused`: `U_online` consultando el índice base;
- `full_stale`: `U_full` consultando un índice construido sobre `V_base`;
- `full_rebuilt`: `U_full` consultando un índice reconstruido sobre `V_full`.

H4 demostrará reutilización funcional en el tamaño de catálogo evaluado; no se interpretará automáticamente como evidencia de escalabilidad industrial.

---

## 6. Baselines oficiales

La demostración principal utilizará exactamente tres estados:

### IBPR_STALE

Modelo IBPR entrenado en el bloque base y mantenido sin actualización.

### OnlineIBPRMejorado

Mismo estado base, adaptado incrementalmente con interacciones recientes y `V` congelada.

### IBPR_FULL_RETRAIN

Nuevo IBPR entrenado desde cero sobre todo el historial observado hasta el punto de evaluación.

OnlineIBPR original queda fuera del conjunto de baselines oficiales y se reserva para diagnóstico histórico/técnico.

---

## 7. Separación entre desarrollo y evaluación final

La selección de modelos y la evidencia final deben permanecer separadas.

### Desarrollo

Se utilizará únicamente un horizonte cronológico de desarrollo para:

- seleccionar `IBPR_FINAL`;
- seleccionar `OnlineIBPRMejorado_FINAL`;
- comparar variantes de `loss_mode` y demás hiperparámetros permitidos;
- depurar implementación e invariantes.

La selección de OnlineIBPRMejorado debe reproducir una dinámica globalmente cronológica y prequential dentro del conjunto de desarrollo, de forma compatible con la evaluación final.

### Evaluación final

El bloque cronológicamente posterior reservado para evaluación final no debe utilizarse para:

- elegir hiperparámetros;
- cambiar la función objetivo;
- decidir número de épocas;
- seleccionar semillas favorables;
- introducir nuevas variantes del algoritmo.

Una vez congeladas `IBPR_FINAL` y `OnlineIBPRMejorado_FINAL`, la evaluación H1-H4 se ejecutará sin retuning.

---

## 8. Política de reproducibilidad

Cada campaña final debe registrar como mínimo:

- versión de protocolo;
- hash del dataset procesado;
- configuración completa de modelos;
- semillas;
- versiones de Python, Cornac, NumPy, SciPy, PyTorch y FAISS cuando corresponda;
- fingerprints/hash del código relevante;
- reglas exactas de split temporal;
- población evaluada;
- archivos de resultados por seed y por punto temporal.

La reanudación de una ejecución solo será válida cuando el fingerprint del protocolo coincida exactamente.

---

## 9. Política de interpretación preespecificada

La demostración se mantendrá enfocada y evitará convertir fluctuaciones aisladas en conclusiones. La unidad principal de réplica será la **seed**; los puntos temporales dentro de una seed se tratarán como medidas repetidas. Para cada seed se agregará primero el estimando a través de los puntos temporales, y luego se resumirá la distribución entre seeds.

Antes de la campaña final se fijará el número de seeds. El objetivo operativo es cinco seeds si el presupuesto computacional lo permite; cualquier número alternativo deberá justificarse y congelarse antes de inspeccionar resultados finales.

### 9.1 H1 --- Adaptación

Estimando primario por punto: `Online_NDCG@20 - Stale_NDCG@20`.

La conclusión se basará en los deltas agregados por seed, reportando valores individuales, media, mediana y dispersión. Con cinco seeds, se considerará evidencia **consistente** de adaptación cuando el delta agregado sea positivo en al menos cuatro seeds y su tendencia central sea positiva; evidencia consistente en sentido contrario si ocurre simétricamente; cualquier otro patrón se describirá como mixto/inconcluso. Esta regla es descriptiva y no sustituye una prueba formal de significancia.

### 9.2 H2 --- Eficiencia

Estimando primario: `speedup = T_full / T_online`, calculado de forma pareada. Se considerará evidencia consistente de menor coste cuando `speedup > 1` en al menos cuatro de cinco seeds y la mediana entre seeds sea mayor que 1. También se reportarán tiempos absolutos y la medición inclusiva de construcción.

### 9.3 H3 --- Trade-off

H3 no tendrá un umbral arbitrario de “éxito”. Se reportará `Delta_OF = Online - Full` por seed y por punto, más las métricas secundarias. El objetivo es cuantificar el coste de calidad asociado a evitar el reentrenamiento, no declarar equivalencia.

### 9.4 H4 --- Indexabilidad

La condición estructural es estricta: `V_base == V_online` bit a bit en todos los puntos y seeds. La reutilización operacional requiere, además, que el índice base no sea reconstruido para la rama online y permanezca sin modificación atribuible a las consultas. Las métricas ANN caracterizan la calidad del backend configurado, pero no sustituyen esta condición estructural.

### 9.5 Límites de las afirmaciones

No se afirmará que OnlineIBPRMejorado:

- resuelve cold-start;
- incorpora nuevos ítems sin actualizar el índice;
- domina siempre a Full Retrain en calidad;
- demuestra escalabilidad industrial por usar ANN;
- supera científicamente a OnlineIBPR original mediante una comparación directa afectada por sus defectos de implementación.

Sí se intentará responder, con evidencia reproducible:

1. si adapta mejor que un IBPR estático en la población evaluada;
2. cuánto cuesta frente a reentrenar;
3. qué trade-off de calidad presenta frente al reentrenamiento completo;
4. si mantiene `V` y permite reutilizar físicamente el índice.

---

## 10. Criterio de finalización

La investigación estará metodológicamente cerrada cuando:

1. la implementación final satisfaga todos los invariantes obligatorios y la propiedad de localidad de `U` haya sido auditada y clasificada;
2. exista una selección reproducible y separada de `IBPR_FINAL` y `OnlineIBPRMejorado_FINAL`;
3. H1-H3 se ejecuten sobre datos finales no utilizados durante selección;
4. H4 demuestre o refute la conservación y reutilización del índice;
5. todas las afirmaciones de la tesis puedan vincularse a una evidencia concreta;
6. las limitaciones y el alcance warm-start estén documentados;
7. los análisis diagnósticos posteriores no hayan sido utilizados para retocar retrospectivamente la configuración final.
