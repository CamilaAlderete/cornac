# Hoja de ruta de reconstrucción experimental V4.1

# OnlineIBPRMejorado

## 1. Principio rector

El objetivo es producir una demostración científica reproducible y enfocada de la contribución de OnlineIBPRMejorado.

Los resultados y scripts existentes se utilizarán como antecedentes técnicos, diagnósticos y material de auditoría. La evidencia oficial se generará nuevamente después de congelar la implementación, los modelos y el protocolo definitivo.

No se añadirán experimentos por acumulación. Cada prueba debe responder directamente a una pregunta científica o a un invariante necesario para confiar en la evidencia.

---

## 2. Camino general

    Definir contribución y alcance

    ↓

    Estabilizar implementación e invariantes

    ↓

    Reconstruir selección reproducible

    ↓

    Congelar código + modelos + protocolo

    ↓

    Ejecutar H1-H3

    ↓

    Ejecutar H4

    ↓

    Diagnósticos solo si son necesarios

    ↓

    Analizar y documentar

---

# FASE A --- Congelación científica

## Objetivo

Fijar exactamente qué se pretende demostrar antes de volver a ejecutar experimentos.

## Entregables

- `CONTEXTO_MAESTRO_ONLINE_IBPR_MEJORADO_V4_1.md`;
- `HOJA_DE_RUTA_RECONSTRUCCION_EXPERIMENTAL_ONLINE_IBPR_MEJORADO_V4_1.md`;
- criterios de interpretación preespecificados para H1-H4;
- definición de RQ1-H1, RQ2-H2, RQ3-H3 y RQ4-H4;
- definición del alcance warm-start;
- definición de baselines oficiales;
- lista explícita de afirmaciones que quedan fuera de alcance.

## Estado de la fase

La FASE A queda **cerrada solo después de aprobación explícita de esta V4.1**. Estos documentos fijan el contrato científico; todavía no congelan hiperparámetros, seeds finales ni resultados.

## Decisiones científicas fijadas

Baselines principales:

- `IBPR_STALE`;
- `OnlineIBPRMejorado`;
- `IBPR_FULL_RETRAIN`.

OnlineIBPR original:

- antecedente técnico;
- comparación diagnóstica opcional;
- no baseline principal.

---

# FASE B --- Auditoría y estabilización de OnlineIBPRMejorado

## Objetivo

Confrontar la implementación actual con el contrato científico antes de decidir cualquier modificación.

## B1. Auditoría sin modificar código

Primero ampliar/ejecutar los tests sobre la implementación actual. Deben verificarse:

- warm-start obligatorio;
- `V` bit a bit exacta con `update_V=False`;
- update vacío = identidad;
- update vacío no consume seed;
- sampling negativo excluye positivos históricos y recientes;
- misma seed reproduce el mismo resultado;
- shapes/rangos inválidos son rechazados;
- usuario sin negativos válidos produce error explícito;
- `max_steps=0` es rechazado;
- modo parcial no acepta estrategias de sampling no implementadas;
- comportamiento de `normalize=True`;
- cambio o no cambio de filas de `U` para usuarios no afectados;
- progresión reproducible de seeds entre llamadas sucesivas.

Archivo principal:

`validation/test_online_ibpr_mejorado_invariants.py`

## B2. Clasificación de hallazgos

Cada hallazgo se clasificará antes de editar el algoritmo:

1. **bug crítico**: viola un invariante obligatorio o invalida una hipótesis principal;
2. **deuda de ingeniería**: aumenta coste o complejidad, pero puede medirse honestamente sin invalidar la contribución;
3. **propiedad deseable/no necesaria**: mejora limpieza o localidad, pero no es requerida para H1-H4.

La inmutabilidad bit a bit de filas de `U` no afectadas pertenece inicialmente a la categoría **propiedad a auditar**, no a un requisito obligatorio.

## B3. Correcciones mínimas

Solo se modificarán bugs críticos o cambios mínimos cuya necesidad científica quede documentada. No se optimizará Adam, la materialización completa de matrices ni otros overheads únicamente para mejorar H2 antes de medirlos.

Después de cualquier corrección se ejecutará de nuevo toda la suite de invariantes.

No avanzar a selección mientras los invariantes obligatorios no pasen.

---

# FASE C --- Reconstrucción de selección reproducible

## C1. IBPR_FINAL

Utilizar exclusivamente el horizonte de desarrollo.

Mantener el HPO/refinamiento de IBPR como base, revisando únicamente los aspectos necesarios para reproducibilidad y fingerprint del protocolo.

Resultado:

`IBPR_FINAL`

## C2. OnlineIBPRMejorado_FINAL

Reconstruir la selección online dentro del horizonte de desarrollo mediante un protocolo globalmente cronológico y prequential.

Flujo:

    base_dev

    ↓ update C1

    eval C2

    ↓ update C2

    eval C3

    ↓ update C3

    eval C4

Condiciones:

- universo warm-start fijado desde la base correspondiente;
- historia para negative sampling contiene solo información observada hasta cada punto;
- `update_V=False` permanece como restricción arquitectónica;
- `angular` y `cosine_bpr` pueden competir durante desarrollo;
- selección primaria orientada a ganancia de adaptación respecto de Stale;
- coste puede utilizarse como criterio secundario, no como sustituto de la métrica primaria.

Resultado:

`OnlineIBPRMejorado_FINAL`

---

# FASE D --- Freeze experimental

## Objetivo

Separar de forma irreversible desarrollo y evaluación final.

Congelar:

- código de IBPR;
- código de OnlineIBPRMejorado;
- configuración `IBPR_FINAL`;
- configuración `OnlineIBPRMejorado_FINAL`;
- dataset y regla de procesamiento;
- split temporal;
- número de chunks;
- métricas;
- semillas finales;
- versiones de dependencias;
- fingerprints de fuentes.

Después del freeze:

- no retuning;
- no cambio de `loss_mode`;
- no cambio de número de épocas;
- no selección de nuevos hiperparámetros a partir de resultados finales.

Si aparece un defecto real de implementación que obliga a modificar código, la campaña final se invalida y se vuelve a FASE B con una nueva versión de protocolo.

---

# FASE E --- Evaluación final H1-H3

## Diseño común

En cada punto temporal comparar sobre la misma población:

    IBPR_STALE

    vs

    OnlineIBPRMejorado_FINAL

    vs

    IBPR_FULL_RETRAIN

La evaluación debe ser prequential y globalmente cronológica.

## H1 --- Adaptación

Comparación primaria:

`Online - Stale`

Población PRIMARY:

usuarios warm-start que hayan recibido actualización antes del punto de evaluación.

Población ALL-WARM:

diagnóstico suplementario, no sustituto de PRIMARY.

Métrica primaria:

`NDCG@20`.

## H2 --- Eficiencia

Comparar:

- tiempo de `partial_fit_recent`;
- tiempo de `IBPR_FULL_RETRAIN.fit`.

Protocolo mínimo de timing, congelado antes de ejecutar:

- mismo hardware y entorno para ambas ramas;
- límites de threads fijados y registrados;
- reloj monotónico de pared;
- comparación pareada dentro del mismo seed/punto;
- carga inicial del dataset fuera de la medida principal;
- construcción de estructuras reportada por separado y también en una medida inclusiva;
- tiempos absolutos y `speedup = T_full / T_online`.

No eliminar overheads de la implementación real de OnlineIBPRMejorado de la medición principal. Los resultados describen esta implementación y este hardware; no se interpretan como complejidad asintótica.

## H3 --- Trade-off de calidad

Estimando primario:

`Delta_OF = Online - Full`.

`NDCG@20` será la métrica principal; las demás métricas serán secundarias.

Reportar `Recovery = (Online - Stale) / (Full - Stale)` solo como diagnóstico complementario y junto con su denominador. Si el denominador no permite una interpretación estable según la regla congelada antes de la campaña final, marcar el ratio como no interpretable.

No interpretar automáticamente como equivalencia o no-inferioridad.

## Réplicas e interpretación

Usar seeds como unidad principal de réplica.

Los puntos temporales dentro de una seed son medidas repetidas y no deben tratarse como observaciones estadísticamente independientes. Primero se agregará el estimando a través de puntos dentro de cada seed y después se resumirá entre seeds.

Objetivo operativo: cinco seeds finales. El número definitivo se fijará en FASE D por razones de presupuesto computacional, antes de inspeccionar cualquier resultado final.

Con cinco seeds:

- H1 se considerará consistente cuando el delta agregado `Online - Stale` sea positivo en al menos 4/5 seeds y su tendencia central sea positiva;
- H2 se considerará consistente cuando `speedup > 1` en al menos 4/5 seeds y la mediana sea mayor que 1;
- H3 se reportará descriptivamente sin umbral artificial de éxito;
- H4 exige `V_base == V_online` bit a bit en todos los seeds/puntos y cero rebuilds online del índice base.

Estas reglas son criterios descriptivos preespecificados, no pruebas formales de significancia.

---

# FASE F --- Evaluación H4 de indexabilidad

## F1. Preflight

Mantener un preflight sintético separado para validar:

- compatibilidad de `ANNMixin`;
- `MEASURE_DOT`;
- funcionamiento de FAISS;
- coherencia entre recuperación exacta e implementación de referencia;
- estabilidad del objeto/índice ante consultas.

El preflight no constituye evidencia H4.

## F2. H4 final

Primera condición:

`V_base == V_online` bit a bit.

Después evaluar:

### online_reused

`U_online` + índice construido una sola vez sobre `V_base`.

### full_stale

`U_full` + índice base no reconstruido.

### full_rebuilt

`U_full` + índice reconstruido sobre `V_full`.

Medir:

- recall ANN respecto de búsqueda exhaustiva;
- acuerdo posicional;
- coincidencia de conjuntos/orden;
- shortfall de candidatos;
- latencia;
- número real de builds/rebuilds;
- identidad/hash del índice cuando sea posible.

Interpretación permitida:

reutilización funcional del índice en el escenario evaluado.

Interpretación no permitida sin evidencia adicional:

escalabilidad industrial general.

---

# FASE G --- Diagnósticos y ablaciones

Los diagnósticos se ejecutarán solo cuando respondan a una anomalía o ayuden a interpretar un resultado final.

## Mantener como soporte

`compare_online_ibpr_original_vs_mejorado.py`

Propósito:

documentar diferencias de implementación y limitaciones de OnlineIBPR original.

No utilizar como baseline principal.

## Mantener como diagnóstico post-hoc

`ablation_a1_update_dynamics_online_ibpr_mejorado_v1_2.py`

Propósito:

explorar dinámicas de actualizaciones repetidas si H1-H3 muestran degradaciones que requieran explicación.

No utilizar sus resultados para retocar retrospectivamente la configuración congelada.

No crear nuevas ablaciones salvo que exista una pregunta concreta que no pueda responderse con la evidencia existente.

---

# FASE H --- Análisis científico y documentación

## Preguntas finales

1. ¿OnlineIBPRMejorado adapta información reciente frente a Stale?
2. ¿Cuál es su coste real frente a Full Retrain?
3. ¿Qué calidad conserva o recupera respecto de Full Retrain?
4. ¿Mantiene exactamente `V` y permite reutilizar el índice?
5. ¿Cuáles son sus limitaciones de alcance?

## Documentación final

    docs/
        PROBLEMA.md
        CONTRIBUCION.md
        ANALISIS_ALGORITMO.md
        PROTOCOLO_EXPERIMENTAL.md
        RESULTADOS.md
        LIMITACIONES.md
        REPRODUCIBILIDAD.md

---

# Criterio de cierre

El proyecto se considerará cerrado cuando:

- los invariantes obligatorios de implementación pasen y la localidad de `U` haya sido auditada/clasificada;
- `IBPR_FINAL` y `OnlineIBPRMejorado_FINAL` hayan sido seleccionados solo con datos de desarrollo;
- código, configuraciones y protocolo hayan sido congelados antes de mirar resultados finales;
- H1-H4 se hayan ejecutado reproduciblemente;
- las afirmaciones principales tengan evidencia directa;
- los resultados negativos o trade-offs estén documentados sin reinterpretación retrospectiva;
- las limitaciones warm-start e indexabilidad estén explícitas;
- no exista una pregunta necesaria para la contribución que permanezca sin prueba.
