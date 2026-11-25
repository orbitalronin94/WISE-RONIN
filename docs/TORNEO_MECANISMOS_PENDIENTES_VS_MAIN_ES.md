# Torneo de mecanismos pendientes: 20 hipótesis contra main

**Estado: protocolo preparado; no ejecutar durante la continuación 185.700 → 235.700.**  
**Alcance:** únicamente mecanismos todavía no integrados en el régimen de entrenamiento. No se reevalúan los órganos que ya forman parte del control `main`.

## Decisión de alcance

La inspección de `model/`, `train/`, `docs/ARQUITECTURA_ACTUAL_ES.md` y `docs/CEREBRO_A_CODIGO_HIPOTESIS_ES.md` distingue entre mecanismos ya presentes y mecanismos opcionales que todavía no forman parte del régimen de entrenamiento.

Quedan fuera del torneo porque ya pertenecen al control o porque no son una modificación del entrenamiento autoregresivo:

- Delta y sus trazas de elegibilidad/modulación retardada.
- Glucosa, dolor y placer como reguladores energéticos.
- Homeostasis y Top-K existentes, sin activar un segundo mecanismo experimental.
- Memoria elástica/lenta, sorpresa y AdaptiveChunking.
- Embedding vivo, router KAN/MLP y pesos compartidos.
- ZE/LC-NE: extensión ya implementada y optativa, no se introduce como novedad en esta tanda.
- ELM: se conserva como posible probe offline de representaciones, pero no entra en este torneo. Un probe que no modifica el entrenamiento no puede superar de forma causal los gates de loss de entrenamiento y pasos/segundo.

El torneo se limita a ocho mecanismos que ya tienen una base experimental en el código, pero cuya integración funcional en el entrenamiento sigue pendiente de validación/promoción:

1. **Synaptic tagging & capture** — tags locales y captura condicionada por recurso global (`tagging_enabled`).
2. **Umbral intrínseco adaptativo** — adaptación lenta y acotada del umbral (`intrinsic_threshold_enabled`).
3. **Poda sináptica simulada** — máscara reversible basada en magnitud/uso; no borra pesos (`simulated_pruning_enabled`).
4. **Memoria one-shot por incertidumbre** — escritura local rápida, acotada y separada del LR global (`one_shot_enabled`).
5. **Escalado sináptico homeostático** — escalado local acotado (`synaptic_scaling`).
6. **Replay priorizado/comprimido** — selección de registros consolidados y coste real de replay (`replay_compression`).
7. **Downscaling durante sueño** — renormalización acotada durante la fase offline (`sleep_downscale`).
8. **LTD protegida por tags** — debilitamiento limitado, condicionado por tags capturados (`ltd_enabled` + tagging).

Que una opción exista en el código no significa que esté integrada ni validada. Esta PR no activa ninguna de ellas en `main`; define cómo competirán contra el control.

## Las 20 hipótesis

Las ocho primeras miden cada mecanismo de forma aislada. Las doce restantes prueban interacciones con una justificación funcional. Cada candidata parte del mismo perfil neutro del control y activa únicamente los flags/ajustes indicados en su hipótesis.

| ID | Hipótesis | Mecanismos |
|---|---|---|
| H01 | Captura local de tags con recurso global | Tagging & capture |
| H02 | Umbral intrínseco adaptativo lento | Umbral intrínseco |
| H03 | Poda simulada por magnitud y uso | Poda simulada |
| H04 | Escritura one-shot gobernada por incertidumbre | One-shot |
| H05 | Escalado sináptico homeostático acotado | Escalado homeostático |
| H06 | Replay priorizado/comprimido | Replay priorizado |
| H07 | Downscaling moderado durante sueño | Sleep downscaling |
| H08 | LTD limitada y protegida por tags | LTD + tagging |
| H09 | Tags locales capturados durante replay priorizado | Tagging + replay |
| H10 | Tags capturados y protegidos durante downscaling | Tagging + sleep |
| H11 | Captura selectiva con LTD protegida | Tagging + LTD |
| H12 | Umbral adaptativo con escalado homeostático | Umbral + escalado |
| H13 | Umbral adaptativo con escritura one-shot | Umbral + one-shot |
| H14 | Poda simulada protegida por tags | Poda + tagging |
| H15 | Poda simulada gobernada por umbral adaptativo | Poda + umbral |
| H16 | One-shot condicionado por tags capturados | One-shot + tagging |
| H17 | One-shot con replay priorizado | One-shot + replay |
| H18 | Escalado homeostático durante consolidación | Escalado + sleep |
| H19 | Replay priorizado con downscaling | Replay + sleep |
| H20 | Consolidación de tres factores: tags, replay y LTD protegida | Tagging + replay + LTD |

No se añadirán mecanismos ya integrados para completar artificialmente las 20 hipótesis. Si una interacción no puede implementarse sin introducir una segunda variable no controlada, se marca como bloqueada y no se sustituye silenciosamente.

## Comparación obligatoria contra main

Cada hipótesis se compara con una ejecución de control desde el mismo checkpoint verificado, con la misma fuente de `main`. El control no activa ninguna candidata.

### Condiciones idénticas

- Checkpoint LUMEN de step **235.700**, una vez terminado y verificado el train actual.
- Mismos pesos, optimizer, scheduler, RNG, seed y estado persistente inicial.
- Mismo corpus, tokenizer, token cache, hashes y secuencia exacta de minibatches.
- Mismo batch, contexto, precisión, número de hilos, afinidad y perfil CPU.
- Mismo runner; control y candidata ejecutados secuencialmente con orden alternado para equilibrar calentamiento.
- 100 pasos por brazo en cribado, divididos en cinco ventanas de 20 pasos.
- Holdout congelado excluido del ajuste de parámetros, selección de umbrales y cálculo de señales.

### Gate binario de promoción

Una hipótesis solo es elegible si satisface **todos** estos requisitos frente a `main`:

1. **Loss media estrictamente menor** en los mismos 100 minibatches.
2. **Pasos/segundo estrictamente mayores**, midiendo tiempo de paso y contabilizando el mantenimiento periódico amortizado.
3. Loss de la última ventana de 20 pasos no peor que la del control.
4. Sin NaN/Inf, explosión de normas, colapso de actividad ni pérdida de estado persistente.
5. Tests de propiedades del mecanismo y equivalencia de señales compartidas superados.
6. Mejora reproducida en tres comparaciones emparejadas; no se promueve por una sola medición favorable.

Los empates no pasan. No se compensa una métrica con la otra: menor loss con menor throughput, o mayor throughput con peor loss, significa **NO MERGE**.

El throughput se calcula con el tiempo total de actualización del trainer, incluyendo el coste amortizado de replay/sueño/poda cuando corresponda. Se excluyen únicamente arranque del proceso y carga inicial de datos/modelo, de forma idéntica en ambos brazos. Se registran por separado forward, backward y mantenimiento cuando el profiler permita medirlos sin overhead desigual.

### Métricas que se conservan

- Loss por paso, por ventana y media total.
- Pasos/segundo y tiempo total de actualización.
- Coste del mecanismo y frecuencia real de activación.
- Actividad efectiva, unidades inactivas persistentes, entropía/Gini de uso y reactivación.
- Normas de parámetros, gradientes, trazas, tags y estados persistentes.
- Hashes de checkpoint, código, configuración, corpus, tokenizer y minibatches.
- Logs íntegros de ambos brazos y JSON de comparación, incluso cuando la candidata falla.

Los mecanismos periódicos deben ejecutarse al menos una vez para ser evaluables. Si no se produce un ciclo de sueño/replay/poda dentro del bloque, el resultado será `NOT_EXERCISED`, nunca una aprobación.

## Orden de ejecución y promoción

1. Esperar al cierre y verificación del checkpoint 235.700.
2. Congelar el SHA de `main`, checkpoint y artefactos de datos.
3. Ejecutar H01–H08 aisladas.
4. Ejecutar H09–H20 solo si sus componentes individuales pasan tests de propiedades; las combinaciones siguen teniendo que superar a `main` por sí mismas.
5. Repetir tres veces cada candidata que supere el primer cribado.
6. Generar un registro completo: aprobadas, pendientes, fallidas, no ejercitadas y conflictos.
7. Integrar exclusivamente las candidatas reproducibles que superen ambos gates y no entren en conflicto. Después medir el perfil combinado contra el mismo control.

**No se lanza automáticamente un train largo ni se modifica `main` por el mero hecho de completar el benchmark.** Cualquier continuación posterior requiere revisar los resultados y autorizar explícitamente el perfil aprobado.

## No-go y trazabilidad

- Esta PR es de preparación documental; no altera el trainer ni activa mecanismos.
- No modifica el entrenamiento en curso ni su checkpoint.
- No usa holdout para seleccionar hiperparámetros.
- No se rebaja el gate para obtener ganadores.
- Una hipótesis sin mejora simultánea queda pendiente/no elegible, con su artefacto preservado.
- El inventario debe actualizarse si una PR distinta integra alguno de estos mecanismos antes de ejecutar el torneo.

**Criterio rector:** cada mecanismo pendiente debe demostrar que aprende mejor y que lo hace más deprisa que `main`. Si no mejora ambas cosas, no entra.
