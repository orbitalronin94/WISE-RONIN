# Lumen-Ronin

**Experimento CPU-first: arquitectura Lumen recurrente frente a un Transformer de control, entrenados desde cero y evaluados bajo condiciones emparejadas.**

La línea experimental activa deja de ser WISE-RONIN. El objetivo ahora es determinar si los órganos de Lumen-Ronin aportan una mejora medible frente a un Transformer comparable, sin aceptar una penalización de velocidad que haga inviable entrenar y experimentar.

El repositorio conserva su nombre técnico actual (`WISE-RONIN`), pero la identidad y el plan de trabajo documentados aquí corresponden a **Lumen-Ronin**. No se reutilizarán checkpoints anteriores: ambos modelos comenzarán en **step 0**, con inicialización nueva.

> **Estado a 22 de noviembre de 2026:** fase de auditoría y optimización de rendimiento. No iniciar el entrenamiento largo hasta completar las pruebas de equivalencia, medir los cuellos de botella y verificar el corpus/tokenizer exactos que compartirán ambos modelos.

## Decisión de proyecto

- **WISE-RONIN:** línea abandonada; no es el objetivo de los próximos entrenamientos.
- **Lumen-Ronin:** arquitectura experimental principal.
- **Transformer:** baseline de control entrenado en paralelo desde cero.
- **Punto de partida:** step 0 para ambos. No importar pesos, momentos de AdamW, scheduler ni estado recurrente de checkpoints previos.
- **Prioridad inmediata:** mejorar pasos/s sin eliminar ni alterar silenciosamente las funciones de los órganos.
- **Entrenamiento largo:** bloqueado hasta cerrar la auditoría de rendimiento y la validación del conjunto de datos.

## Referencia de rendimiento y problema

En una medición previa de Lumen V3.1, ambos modelos tenían un tamaño cercano a 2,3 millones de parámetros. Lumen registró una pérdida de 6,353 frente a 7,838 del Transformer en aquel punto de comparación, pero necesitó 183 segundos por 100 pasos frente a 13,7 segundos: aproximadamente **13,3 veces más lento**.

Ese resultado es una señal de que hay que investigar, no una prueba definitiva de superioridad de calidad: ambos obtuvieron 0/155 en aquel holdout y no lograron generar respuestas válidas de forma fiable. La pérdida de entrenamiento no sustituye la evaluación de generaciones.

## Esquema de optimización previo al entrenamiento

Las optimizaciones se aplicarán en orden, midiendo cada una de forma aislada y conservando una ruta de referencia. No se aceptará una mejora de throughput si cambia la semántica del modelo, rompe gradientes o degrada la estabilidad.

### 1. Medición reproducible de la carga real

Antes de cambiar código, registrar por separado:

- Muestreo y construcción del batch.
- Forward de cada órgano y de la ruta completa.
- Cálculo de cada componente de pérdida.
- Backward.
- Gradient clipping.
- Actualización AdamW.
- Diagnósticos, comprobaciones de salud y snapshots.
- Replay/sueño.
- Escritura de checkpoints y evaluación.

Cada perfil debe declarar hardware/runner, versión de Python y PyTorch, hilos intra-op e inter-op, afinidad CPU, MKLDNN, foreach/fused AdamW, batch, longitud de secuencia, warmup y pasos medidos. Publicar mediana y dispersión, pasos/s y tokens/s. Separar tiempo de compilación inicial del throughput estable, pero informar ambos.

### 2. Bucle recurrente de EstadoDelta

**Sospecha prioritaria.** `EstadoDelta.forward` proyecta K/V/Q de forma conjunta, pero actualiza el estado recurrente con un bucle Python secuencial sobre cada token. La recurrencia real no puede paralelizarse sin más: el estado de cada token depende del anterior.

Plan:

1. Perfilar por separado proyecciones, normalizaciones, operaciones `bmm`, gates y actualización recurrente.
2. Reducir asignaciones temporales y conversiones redundantes dentro del bucle.
3. Probar compilación de la región recurrente, midiendo compilación, ejecución y coste de backward.
4. Evaluar variantes vectorizadas solo si preservan exactamente la dependencia temporal y se demuestra equivalencia numérica/funcional.
5. Mantener una implementación de referencia para comparar salidas, estados y gradientes.

No sustituir la recurrencia por una aproximación paralela solo para elevar pasos/s.

### 3. Homeostasis y Top-K

El órgano Homeostasis selecciona Top-K por token y calcula actividad efectiva por unidad. Actualmente mantiene estadísticas y puede materializar tensores densos o scatter operations.

Plan:

- Medir Top-K, gather, ReLU, scatter y actualización de EMA por separado.
- Comparar el camino de scatter fusionado con el camino de referencia.
- Evitar transferencias a CPU y conversiones a escalares en cada paso.
- Actualizar los resúmenes detallados en intervalos controlados, sin eliminar las señales necesarias para la homeostasis.
- Comprobar equivalencia de salida, actividad efectiva, bias y gradientes.

**Restricción:** no cambiar `k_top`, `m_target`, `eta` ni la regla de actualización como optimización de velocidad sin un experimento funcional separado.

### 4. Pérdidas auxiliares

La pérdida de Lumen incluye cross-entropy, JEPA, diversidad, router y penalización de repetición. El Transformer de control debe tener su objetivo claramente declarado; comparar solo la cross-entropy del Transformer contra la suma de pérdidas de Lumen no permite atribuir diferencias de tiempo a la arquitectura.

Plan:

- Cronometrar cada pérdida y su backward individualmente.
- Medir el coste de JEPA, diversidad, router y repetición.
- Mantener una comparación de objetivo común (next-token cross-entropy) para aislar el coste arquitectónico.
- Ejecutar después una ablación documentada de los auxiliares, midiendo calidad y velocidad.
- No retirar una pérdida de la configuración principal sin registrar el cambio y su efecto en holdouts.

### 5. Diagnósticos y comprobaciones numéricas

El trainer ejecuta comprobaciones exhaustivas de parámetros/estado del optimizador cada cierto número de pasos y recopila diagnósticos de órganos.

Plan:

- Mantener checks de loss y gradientes no finitos en cada actualización.
- Mantener escaneos completos de parámetros y estado del optimizador en un intervalo configurable y explícito.
- Separar telemetría barata por paso de snapshots detallados.
- No hacer `.item()`, `.cpu()` ni sincronizaciones innecesarias en el camino crítico.
- Registrar cualquier reducción de frecuencia de diagnóstico y conservar una auditoría completa en los checkpoints.

Nunca desactivar silenciosamente las protecciones numéricas para mejorar el benchmark.

### 6. AdamW y operaciones CPU

Medir las opciones reales del runner antes de fijar el perfil:

- Intra-op: comparar 1, 2 y 4 hilos.
- Inter-op: comparar configuraciones pequeñas y estables.
- MKLDNN: ON/OFF con la misma carga y versión de PyTorch.
- AdamW: `foreach` y `fused` cuando estén soportados.
- `zero_grad(set_to_none=True)` frente a reutilización de gradientes.
- Afinidad CPU y variables OMP/MKL.
- `torch.set_flush_denormal` como hipótesis secundaria.

No asumir que más hilos o una opción llamada fused siempre acelera. El perfil ganador debe elegirse por throughput estable y no por una única repetición.

### 7. torch.compile

Evaluar `default` y `reduce-overhead` en la misma revisión de código:

- Contabilizar tiempo de compilación y posibles recompilaciones.
- Medir throughput después del warmup.
- Verificar compatibilidad con estado recurrente, Top-K, backward y checkpoint/resume.
- Rechazarlo si la compilación no amortiza su coste en el tamaño real de los bloques.

### 8. Sueño y memoria

La configuración actual registra consolidación cada 500 pasos y hasta 50 pasos de replay. Se conserva inicialmente: no se desactiva para maquillar el throughput ni se cambia su frecuencia antes de medirla.

- Medir el coste del evento de sueño por separado y su coste amortizado.
- Medir el optimizador/replay, copias de memoria y cualquier sincronización.
- Comprobar que la memoria lenta recibe escrituras y que la consolidación realmente ocurre.
- Verificar que el estado transitorio se reinicia en los límites definidos y que el replay no contamina la evaluación.

### 9. Checkpoints y evaluación

- Medir serialización y tamaño de checkpoint aparte del tiempo de actualización.
- Mantener checkpoints recuperables con modelo, optimizador, scheduler, RNG, contador de tokens y hashes.
- Evaluar con `model.eval()`, `no_grad()` y estado recurrente reiniciado por prompt.
- No ejecutar replay, sueño ni escrituras de memoria durante evaluación.
- Mantener exactamente el mismo tokenizer y contrato de EOS entre entrenamiento y evaluación.

## Protocolo de comparación desde cero

Lumen-Ronin y Transformer deben ejecutarse como tareas paralelas dentro de una misma campaña emparejada, sin que una use más datos o un presupuesto distinto.

| Elemento | Condición inicial |
|---|---|
| Paso inicial | 0 para ambos |
| Semilla | 42 |
| Batch | 8 |
| Longitud de secuencia | 128 |
| Datos | Mismos ejemplos y mismo orden de muestreo reproducible |
| Tokenizer | El mismo artefacto y SHA-256 |
| Runner | Mismo tipo de runner CPU |
| Presupuesto | Mismo número de pasos y tokens vistos |
| Evaluación | Mismo holdout congelado, una evaluación emparejada al final de cada bloque |
| Recuperación | Checkpoint independiente para cada modelo |

El protocolo de bloques cortos sirve para comprobar que la velocidad y la calidad se sostienen. No se inicia una campaña larga hasta superar el smoke test, comprobar checkpoint/resume y revisar generaciones reales.

### Métricas obligatorias

- Pasos/s y tokens/s, con tiempos de warmup y compilación separados.
- Tiempo por etapa y coste amortizado del sueño.
- Loss de next-token y componentes auxiliares por separado.
- EOS correcto, longitud de salida, respuestas vacías y truncamiento.
- Exactitud factual, aritmética y razonamiento.
- Repetición, bucles, diversidad y fugas de plantillas.
- Porcentaje de respuestas en español y texto fuera de contexto.
- Estado de memoria, tasa de escritura, actividad efectiva y valores no finitos.
- Hashes de corpus, tokenizer, código, configuración y checkpoints.

La comparación debe incluir tanto calidad por paso como calidad por unidad de tiempo. Un modelo que mejora la pérdida pero necesita 13 veces más tiempo no se considera ganador automáticamente.

## Datos: estado y condición de uso

Existe una preparación anterior de corpus español licenciado y una mezcla con material histórico propio documentada en PR #64 y PR #65. La ejecución de mezcla procesó 147.627 filas históricas y seleccionó 10.341 grupos canónicos; los logs registraron 80.446 filas de train, 1.413 de validation y 2.431 de holdout.

Sin embargo, la ejecución [45008699964](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45008699964) falló en una aserción del manifiesto después de la extracción y no publicó el artefacto combinado. Por tanto, **ese conjunto no se considera todavía un artefacto de entrenamiento verificado**. Antes de usarlo hay que corregir la clave de validación, publicar el artefacto y verificar ficheros, manifiesto, hashes y separación de splits.

El corpus resultante es principalmente SFT/QA/instrucciones, texto paralelo y material sintético propio; no debe presentarse como pretraining general de todo el español. El mismo conjunto y tokenizer exactos se usarán en los dos modelos. No se entrenará con validation ni holdout.

## Estado de decisiones

| Decisión | Estado |
|---|---|
| Abandonar WISE-RONIN como línea activa | Adoptado |
| Lumen-Ronin frente a Transformer | Adoptado |
| Ambos desde step 0 | Adoptado |
| Preservar la semántica de los órganos durante optimización | Obligatorio |
| Medir cada etapa antes de modificarla | Obligatorio |
| Mantener sueño cada 500 pasos al inicio | Conservado provisionalmente |
| Homeostasis sin cambios funcionales iniciales | Conservado |
| Campaña larga antes de cerrar benchmark y corpus | No autorizada |

## Próximos hitos

1. Congelar el código y los inputs de referencia para tener una medición reproducible de ambos modelos.
2. Ejecutar el perfil por etapas y localizar los mayores costes reales.
3. Aplicar optimizaciones de ejecución de una en una, con pruebas de equivalencia y regresión.
4. Elegir un perfil CPU común y repetir el benchmark varias veces.
5. Verificar y publicar el corpus/tokenizer exactos que compartirán ambos modelos.
6. Ejecutar smoke test desde step 0, incluyendo backward, checkpoint y reanudación.
7. Lanzar bloques emparejados Lumen/Transformer desde step 0 y revisar calidad antes de ampliar duración.

## Código y workflows

- [Código de Lumen](model/lumen.py)
- [Estado Delta](model/delta_state.py)
- [Homeostasis](model/homeostasis.py)
- [Pérdidas](train/loss.py)
- [Entrenador](train/train.py)
- [Workflow de entrenamiento](.github/workflows/train.yml)
- [Workflow comparativo anterior](.github/workflows/pilot.yml)

Los workflows históricos y sus checkpoints son referencias de auditoría; no constituyen el punto de partida del nuevo experimento. Cualquier nuevo workflow debe agrupar la preparación, las pruebas y la comparación necesarias sin crear matrices de workflows redundantes.

---

*Lumen-Ronin no tiene que demostrar que puede ejecutar más operaciones por segundo a cualquier precio. Tiene que demostrar que sus órganos conservan su función, que el modelo aprende y que la mejora compensa su coste computacional frente a un Transformer de control.*
