Este experimento quiere probar una idea muy concreta:

**si el rerank del modelo online realmente depende del retrieval previo y solo sirve para refinarlo, en vez de reemplazar la etapa de recuperación top-k.**

## Qué quiere probar el experimento

El código arma un escenario **warm-start**:

* entrena un **IBPR base** sobre un segmento `base`
* crea un **OnlineIBPRMejorado** con `warm-start` desde `U` y `V` del base
* adapta ese online con un segmento `adapt`
* evalúa sobre `test`, pero filtrando `adapt` y `test` para que solo queden **usuarios e ítems ya conocidos por la base**. El propio benchmark lo declara como `Scope = warm_start_known_entities` y `Cold-start = False`. 

Después compara tres cosas:

* **`Online_exhaustive_oracle`**: el ranking exhaustivo del modelo online adaptado sobre todo el catálogo, usado como **referencia ideal de calidad**.
* **`IBPR_indexed_topk`**: el retrieval base indexado con IBPR, usado como **baseline de serving**. 
* **`Hybrid_rerank_from_N_candidates`**: primero recupera `N` candidatos con el índice y luego los reranquea con el modelo online. En tu versión actual, esa recuperación ya usa `online.U[user_idx]`, o sea, el **usuario adaptado**, no el usuario viejo del base. 

Las métricas elegidas apuntan exactamente a esa hipótesis:

* `candidate_coverage_of_online_exhaustive@k`: si el retrieval ya recuperó lo que el online ideal querría en su top-k
* `agreement_with_online_exhaustive@k`: qué tan cerca queda el rerank del ranking exhaustivo ideal
* `subset_of_candidate_pool_rate`: si el rerank se mantiene dentro del conjunto recuperado, es decir, si realmente **refina** en vez de reemplazar retrieval. 

## Qué resultados salieron

En tu corrida:

* el **oracle exhaustivo** del online obtuvo `recall@k = 0.0469`, con `mean_ms = 0.0465`
* el **IBPR indexado base** obtuvo `recall@k = 0.0496`, con `agreement = 0.8288` respecto al oracle y `mean_ms = 0.7258`
* el **híbrido** ya con solo **20 candidatos** obtuvo:

  * `recall@k = 0.0469`
  * `coverage = 1.0000`
  * `agreement = 1.0000`
  * `subset_ok = 1.0000`
  * `mean_ms = 0.7294` 

Y eso se mantuvo igual en calidad para 40, 100, 200 y 400 candidatos:

* siempre `coverage = 1.0000`
* siempre `agreement = 1.0000`
* siempre `subset_ok = 1.0000`
* y el `recall@k` siguió en `0.0469`

Lo único que cambió al aumentar candidatos fue la latencia, que subió desde `0.7294 ms` con 20 hasta `0.8756 ms` con 400. 

## Qué significan esos resultados

El significado principal es este:

### 1. El rerank sí está refinando, no reemplazando

`subset_ok = 1.0000` en todos los casos significa que el resultado final del híbrido siempre salió del conjunto de candidatos recuperados. Entonces el rerank no “inventa” nuevos ítems fuera del pool; solo los reordena.

### 2. El retrieval ya está trayendo todo lo necesario

Como con solo 20 candidatos ya tienes `coverage = 1.0000`, eso significa que los ítems que el online exhaustivo querría en su top-k **ya estaban todos presentes** en los primeros 20 candidatos recuperados. 

### 3. Por eso el rerank reproduce exactamente al oracle

Como la cobertura ya es completa y el acuerdo también es `1.0000`, el rerank logra reconstruir exactamente el ranking ideal del online exhaustivo desde ese conjunto pequeño de candidatos. 

### 4. Aumentar el pool ya no aporta calidad

Pasar de 20 a 40, 100, 200 o 400 candidatos no mejora nada en cobertura, acuerdo ni recall. Solo aumenta el tiempo. Entonces, para esta corrida, **20 candidatos ya son suficientes**. 

## La observación importante sobre el `recall`

Aquí hay un punto clave:

el **baseline IBPR indexado** tuvo `recall@k = 0.0496`, que es un poco mayor que el `0.0469` del oracle online y del híbrido. 

Eso **no significa** que el experimento esté mal.

Significa que el benchmark no está tratando al oracle como “la verdad absoluta” de relevancia real, sino como **referencia ideal del ranking del modelo online adaptado**. El propio script lo etiqueta como `quality_oracle`.

Entonces la lectura correcta no es:

> “el híbrido siempre mejora al base en recall real”

sino esta:

> “el híbrido logra reproducir exactamente lo que el modelo online exhaustivo querría rankear, usando solo un pequeño conjunto de candidatos recuperados”

Y en esta corrida concreta, ese ranking ideal del modelo online no supera al base en `recall@k`.

## Conclusión final

Este experimento prueba correctamente, en escenario **warm-start**, que:

* el rerank **no reemplaza** al retrieval;
* depende totalmente del conjunto recuperado;
* y, con tu arquitectura actual, el retrieval ya es tan bueno que con solo 20 candidatos el híbrido reproduce exactamente el ranking exhaustivo del modelo online.

La conclusión práctica sería:

> En esta configuración, el sistema híbrido no necesita pools grandes de candidatos para aproximarse al ranking ideal del modelo online. El rerank cumple efectivamente el rol de refinamiento sobre un conjunto recuperado, y no de sustitución del retrieval. Sin embargo, el benchmark también muestra que, en esta corrida, el ranking ideal del modelo online no supera al baseline IBPR en recall real, por lo que el valor principal del experimento está en validar la función del rerank dentro de la arquitectura híbrida, más que en demostrar una mejora absoluta de calidad frente al base.
