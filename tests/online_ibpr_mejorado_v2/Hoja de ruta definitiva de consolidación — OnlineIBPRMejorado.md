# Hoja de ruta definitiva de consolidación — OnlineIBPRMejorado

## 1. Principio rector

A partir de ahora, el objetivo ya no es generar más experimentos.

El objetivo es:

> **Conservar o reconstruir únicamente los experimentos necesarios para demostrar, reproducir y defender la contribución central de OnlineIBPRMejorado.**

Un test existente puede ser:

```text
KEEP
REFACTOR
MERGE
RECREATE
SUPPORT
ARCHIVE
DELETE
```

Rehacer un test está permitido si mejora claridad, reproducibilidad o rigor.

No está permitido rehacerlo para buscar un resultado más conveniente.

---

## 2. Tesis central

> **OnlineIBPRMejorado es una extensión incremental de Indexable BPR que permite incorporar nuevas interacciones de usuarios conocidos mediante actualización parcial, con un coste computacional muy inferior al reentrenamiento completo y preservando los factores de ítems cuando se desea mantener reutilizable el índice top-k. Estas propiedades permiten proponerlo como componente online complementario entre reentrenamientos periódicos de IBPR dentro de una arquitectura híbrida offline-online.**

---

## 3. Las cuatro preguntas que deben quedar demostradas

### Q1 — ¿Por qué fue necesario OnlineIBPRMejorado?

Analizar las limitaciones del `OnlineIBPR` original:

```text
tripletas
negative sampling
history
warm-start
mini-batches
actualizaciones sucesivas
scoring/representación
control de V
```

---

### Q2 — ¿Funciona correctamente?

Debe existir un conjunto canónico de invariantes para demostrar:

```text
warm-start
actualización de U
V fija con update_V=False
negative sampling válido
update vacío = identidad
reproducibilidad
progresión de seeds
mappings coherentes
inputs inválidos
ausencia de leakage experimental
```

---

### Q3 — ¿Qué calidad/coste ofrece frente a IBPR?

Comparación principal:

```text
IBPR_STALE
      vs
OnlineIBPRMejorado
      vs
IBPR_FULL_RETRAIN
```

Calidad:

```text
NDCG@20
Recall@20
Precision@20
MAP
AUC
```

Eficiencia:

```text
partial update
full retraining
coste acumulado
speedup
```

No se requiere que Online gane siempre.

---

### Q4 — ¿Puede reutilizar el índice?

Demostrar:

```text
update_V=False
      ↓
V_online == V_base
      ↓
mismo índice válido
      ↓
0 rebuilds Online
```

con controles:

```text
Online + índice reutilizado
Full + índice stale
Full + índice reconstruido
```

---

# 4. Lo que NO tenemos que demostrar

Queda fuera del objetivo principal:

```text
Online siempre mejor que IBPR
equivalencia con Full Retrain
eliminar totalmente Full Retrain
granularidad óptima
frecuencia óptima
Adam persistente
n_epochs óptimo
max_steps óptimo
update_V=True óptimo
chunk óptimo
política híbrida universalmente óptima
cold-start
universalidad entre datasets
```

Estas preguntas van a:

```text
Limitaciones
Trabajo futuro
```

---

# 5. Decisiones congeladas

Salvo bug real, no se reabren:

```text
R900
O014
seeds [777,999]
NDCG@20 como principal
rating >= 3
update_V=False
warm-start
protocolo temporal
```

Podemos cambiar el código que ejecuta el experimento.

No podemos cambiar retrospectivamente el experimento científico para buscar mejores resultados.

---

# 6. Estado actual

```text
Implementación                         COMPLETADA
Tests funcionales                     COMPLETADOS / por consolidar
R900                                  CERRADO
O014                                  CERRADO
Stale vs Online vs Full               CERRADO
H4                                    CERRADO
A1                                    CERRADO / diagnóstico
A2                                    NO CONTINUAR
```

Siguiente etapa:

> **Poda y consolidación.**

---

# 7. Camino oficial

```text
FASE 1
INVENTARIO DE CÓDIGO
        ↓
FASE 2
AGRUPAR POR PREGUNTA CIENTÍFICA
        ↓
FASE 3
CLASIFICAR
KEEP / REFACTOR / MERGE / RECREATE /
SUPPORT / ARCHIVE / DELETE
        ↓
FASE 4
DISEÑAR PIPELINE CANÓNICO
        ↓
FASE 5
REFACTOR / RECREAR LO NECESARIO
        ↓
FASE 6
EJECUTAR UNA VEZ EL PIPELINE LIMPIO
        ↓
FASE 7
GENERAR RESULTADOS FINALES LIMPIOS
        ↓
FASE 8
AUDITAR EVIDENCIA
        ↓
FASE 9
DETECTAR HUECOS REALES
        ↓
       ¿hay alguno?
       /        \
     NO          SÍ
      ↓           ↓
 CERRAR      UNA prueba
             mínima
       \        /
        ↓      ↓
FASE 10
CIERRE EXPERIMENTAL
        ↓
FASE 11
TESIS
```

---

# 8. Clasificación de cada archivo

## KEEP

Ya es claro, correcto y necesario.

## REFACTOR

El experimento es válido, pero el código necesita limpieza.

## MERGE

Dos o más pruebas deberían convertirse en una.

## RECREATE

La prueba es necesaria, pero conviene implementarla limpia desde cero.

Mantiene:

```text
datos
seeds
configuración
protocolo
pregunta científica
```

## SUPPORT

Evidencia secundaria.

## ARCHIVE

Pilotos, versiones viejas, diagnósticos y ablaciones.

## DELETE

No aporta evidencia ni trazabilidad útil.

---

# 9. Pipeline objetivo

Idealmente terminaremos con algo cercano a:

```text
01_validate_implementation
        ↓
02_select_ibpr
        ↓
03_select_online_ibpr_mejorado
        ↓
04_compare_models_final
        ↓
05_validate_index_reuse
        ↓
06_build_final_report
```

No necesariamente serán exactamente seis archivos.

Lo importante es que exista **una sola ruta oficial**.

---

# 10. Experimento principal final

Idealmente `compare_models_final` debe generar en una ejecución:

```text
QUALITY
Stale vs Online vs Full

ADAPTATION
Online-Stale
Online-Full

EFFICIENCY
partial vs full
speedup

INVARIANTS
V frozen
```

Esto puede permitir fusionar varios tests actuales.

---

# 11. OnlineIBPR original

Se decidirá durante la auditoría.

Puede terminar como:

```text
SUPPORT
```

si el análisis existente es suficiente.

O integrarse en la comparación final:

```text
Stale
Original
Mejorado
Full
```

si concluimos que merece ser una comparación central.

No tomaremos esa decisión antes de ver todo el código.

---

# 12. A1

A1 queda como:

```text
ARCHIVE / SUPPORT DIAGNÓSTICO
```

Puede utilizarse en discusión y trabajo futuro.

No forma parte del pipeline reproductivo principal.

No se deriva A2.

---

# 13. ML10M

```text
DECISIÓN POST-PODA
```

Sólo se ejecutará si detectamos una necesidad real de demostrar escala/generalización.

Si se ejecuta:

```text
una validación
sin HPO
sin retuning
```

---

# 14. H5

No es obligatorio.

Puede proponerse la integración híbrida a partir de:

```text
adaptación incremental
+
coste muy inferior
+
V fija
+
índice reutilizable
+
Full Retrain periódico
```

H5 sólo será necesario si queremos demostrar una **política híbrida concreta**.

---

# 15. Matriz de evidencia final

Antes de cerrar debemos poder completar:

| Afirmación | Evidencia |
|---|---|
| Online original tiene limitaciones | análisis/diagnóstico |
| Mejorado corrige esas limitaciones | invariantes |
| R900 fue seleccionado reproduciblemente | selección IBPR |
| O014 fue seleccionado reproduciblemente | selección Online |
| Calidad vs Stale/Full caracterizada | comparación final |
| Online es mucho más barato que Full | comparación final |
| V permanece fija | invariantes/H4 |
| Índice reutilizable | H4 |
| Puede proponerse como capa híbrida | síntesis |
| Política híbrida óptima | trabajo futuro |

Si las primeras ocho están sólidamente cubiertas:

> **la experimentación es suficiente.**

---

# 16. Cuándo volver a ejecutar

Se vuelve a ejecutar si:

```text
RECREATE
REFACTOR relevante
MERGE
faltan outputs
hay un problema metodológico
queremos regenerar el conjunto final limpio
```

No se vuelve a ejecutar para:

```text
buscar mejores números
corregir un resultado negativo
volver a tunear O014
```

---

# 17. Cuándo crear una prueba nueva

Sólo si se cumplen las cuatro:

```text
1. falta evidencia para una afirmación central;
2. ningún test existente puede responderla;
3. la nueva prueba es directa y acotada;
4. está dentro del alcance de la tesis.
```

Si no:

```text
Trabajo futuro.
```

---

# 18. Estructura objetivo

Conceptualmente:

```text
tests/online_ibpr_mejorado/
│
├── validation/
├── selection/
│   ├── ibpr/
│   └── online/
├── experiments/
│   ├── final_comparison/
│   └── index_reuse/
├── support/
├── reporting/
├── results/
│   └── final/
├── docs/
└── archive/
    ├── pilots/
    ├── old_versions/
    ├── ablations/
    └── deprecated/
```

La estructura exacta se decidirá después del inventario.

---

# 19. Resultados finales

`results/final/` debe contener únicamente resultados del pipeline canónico.

Cada resultado debe relacionarse claramente con:

```text
script
SHA
protocolo
dataset
configuración
seed
timestamp
```

Los resultados anteriores pasan a archivo histórico.

---

# 20. Criterio de cierre

Cerramos experimentalmente cuando:

```text
hay un único pipeline;
no existen versiones ambiguas;
las afirmaciones centrales tienen evidencia;
los resultados finales provienen del pipeline limpio;
R900/O014 están congelados;
las limitaciones están documentadas;
el resto está clasificado como trabajo futuro.
```

Después:

```text
NO MÁS EXPERIMENTOS
        ↓
TESIS
```

salvo bug real.

---

# 21. Próximo paso

El siguiente paso exacto será que me pases **todos los códigos fuente de las pruebas**.

Haré:

```text
inventario
↓
clasificación
↓
dependencias
↓
redundancias
↓
propuesta de estructura
↓
qué conservar
↓
qué fusionar
↓
qué recrear
↓
qué archivar
↓
qué eliminar
↓
qué volver a ejecutar
```

Sólo después de eso tocaremos resultados.

---

## Regla final

> **No conservar un test sólo porque existe.  
> No crear un test sólo porque surge una nueva pregunta.  
> Conservar, refactorizar o recrear únicamente lo necesario para demostrar la historia científica central de la tesis.**