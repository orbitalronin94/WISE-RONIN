# Lumen-Ronin

**Lumen-Ronin es el modelo principal de este repositorio:** una arquitectura recurrente CPU-first con memoria Delta, Homeostasis, presupuesto energético, glucosa y órganos de regulación afectiva. El Transformer se conserva como baseline de control e instrumento de comparación, no como modelo principal.

El repositorio mantiene el nombre técnico WISE-RONIN, pero el proyecto activo, la documentación y la línea de entrenamiento corresponden a **Lumen-Ronin**. La arquitectura que se promueve es ronin_lumen_v3_2_parallel_delta; no se eliminan sus órganos para perseguir velocidad. Las optimizaciones deben preservar sus funciones y demostrar equivalencia o mejora con pruebas y mediciones.

> **Estado experimental — 22 de noviembre de 2026, checkpoint 16.000:** Lumen obtuvo **786/2.431 (32,332%)** en el holdout español congelado; el Transformer obtuvo **761/2.431 (31,304%)**. Lumen aventajó al baseline por 25 respuestas (+1,028 puntos porcentuales). La comparación emparejada aún no es estadísticamente concluyente (McNemar exacto bilateral p=0,329); tampoco se rechaza el azar ponderado del conjunto (31,619%; p=0,231). Se trata de una señal favorable, no de una victoria demostrada.

## Decisión de proyecto

- **Modelo principal:** Lumen-Ronin, arquitectura recurrente ronin_lumen_v3_2_parallel_delta.
- **Baseline:** Transformer pequeño, conservado para controles y comparaciones cuando sean informativas.
- **Checkpoint de trabajo:** Lumen step 16.000, con pesos, buffers de los órganos y estado de AdamW preservados.
- **Siguiente bloque:** 5.000 pasos adicionales, step 16.000 → 21.000, solo Lumen.
- **Scheduler:** warmup de 100 pasos desde el LR agotado, meseta a 3e-5 hasta step 20.000 y decaimiento coseno suave hasta 1,5e-5 en step 21.000. El LR no vuelve a cero en este bloque.
- **Reguladores:** glucosa, dolor y placer siguen siendo señales independientes y acotadas; los órganos afectivos modulan el presupuesto interno, no el LR ni los momentos del optimizador.
- **Criterio de avance:** holdout congelado, precisión y NLL informadas por separado, integridad de checkpoint y throughput. La loss de entrenamiento por sí sola no acredita generalización.

## Holdout español de step 16.000

La evaluación usa 2.431 preguntas congeladas (1.931 de tres opciones y 500 de cuatro), el mismo tokenizer y contexto para ambos modelos, y puntuación por NLL condicional media de cada respuesta. No hubo actualizaciones durante la evaluación.

| Modelo | Aciertos | Accuracy | NLL medio |
|---|---:|---:|---:|
| **Lumen-Ronin** | **786/2.431** | **32,332%** | 7,1953 |
| Transformer baseline | 761/2.431 | 31,304% | **6,8550** |

Lumen gana en clasificación (+25 aciertos), mientras que el Transformer conserva menor NLL medio. Son señales distintas: la NLL no debe utilizarse como sustituto de la precisión. La diferencia de precisión no alcanza significación en el contraste emparejado (McNemar exacto bilateral p=0,329), y Lumen tampoco supera de forma significativa el azar ponderado (p=0,231). El resultado justifica continuar el experimento de Lumen, no proclamar superioridad estadística.

## Referencia de rendimiento y problema

En una medición previa de Lumen V3.1, ambos modelos tenían un tamaño cercano a 2,3 millones de parámetros. Lumen registró una pérdida de 6,353 frente a 7,838 del Transformer en aquel punto de comparación, pero necesitó 183 segundos por 100 pasos frente a 13,7 segundos: aproximadamente **13,3 veces más lento**. En el piloto fresco del 22 de noviembre, el Transformer completó 1.000 pasos en 141,8 s; Lumen completó solo sus primeros 100 pasos en 179,7 s antes de que el job se cancelara.

Ese resultado es una señal de que hay que investigar, no una prueba definitiva de superioridad de calidad: ambos obtuvieron 0/155 en aquel holdout y no lograron generar respuestas válidas de forma fiable. La pérdida de entrenamiento no sustituye la evaluación de generaciones.

## Autopsia de los logs existentes (22 de noviembre)

No hace falta volver a ejecutar la batería de benchmarks ya disponible. Los resultados existentes permiten separar los costes dominantes de los secundarios y descartar varias optimizaciones que se habían activado en el perfil llamado `fast`.

### Evidencia directa del entrenamiento emparejado

En [run 45003881342](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45003881342), Lumen y Transformer continuaron desde el mismo step 8.250, con el mismo corpus/tokenizer, batch 8, contexto 128 y cuatro hilos. El entrenamiento de ambos jobs terminó correctamente y el holdout emparejado también; el run global quedó marcado como fallido por probes secundarios de LR, no por los dos entrenamientos principales.

| Modelo | Tiempo típico por 100 pasos | Forward | Backward | Overhead Python |
|---|---:|---:|---:|---:|
| Lumen | 102–110 s | 30–32 s | 68–72 s | 3,0–3,2 s |
| Transformer | 13,8–14,0 s | 3,86–3,90 s | 8,57–8,76 s | 1,3–1,35 s |

La brecha real en ese run es de aproximadamente **7,5–8 veces**. No la explican la carga del batch, los checkpoints, el sueño ni el logging: la mayor parte del tiempo está dentro del forward/backward de Lumen. Sus cuatro capas Delta ejecutan una recurrencia secuencial de 128 posiciones por capa; cada posición contiene varias operaciones pequeñas dependientes del estado anterior y autograd debe conservar/trazar esa cadena.

Los órganos no aparecen muertos: los logs muestran `write=1.000` y `dead=0.00%` en las cuatro capas. Por tanto, no hay evidencia para reducir actividad, eliminar Homeostasis ni debilitar Delta como atajo de velocidad.

### Qué ya se probó y qué dicen los resultados

| Experimento | Resultado registrado | Conclusión |
|---|---|---|
| [Pipeline real: eager](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44941577031) | 1,0725 pasos/s; 0,93242 s/paso | Referencia del perfil por etapas |
| Pipeline sin JEPA/repetición | 1,1117 pasos/s; 0,89949 s/paso | Solo ~3,7% más rápido. Las pérdidas auxiliares no explican la brecha principal |
| Pipeline sin snapshots de salud | 0,7822 pasos/s | No produjo una mejora; no hay base para quitar telemetría |
| Pipeline `torch.compile` (benchmark corto anterior) | 2,8558 pasos/s; compilación informada de 1,455 s | Resultado prometedor, pero no se reprodujo así en el trainer completo posterior |
| [A/B real eager vs compile](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45003557749) | Eager: 50 pasos en 57,18 s. Compiled: 50 pasos en 1.423,42 s | El trainer completo incurrió en unos 22,5 min de compilación inicial y pausas adicionales de compilación |
| [Ablación de fastpaths](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44954484448) | Delta pre-normalizado: 0,3208 pasos/s; scatter fusionado: 0,5238 pasos/s; control: 0,6629 pasos/s | Ambos fastpaths empeoran el throughput y deben apagarse |
| Perfil de cuatro hilos del mismo benchmark | 0,7249 pasos/s frente a 0,6629 del control | Mejora medida de ~9,3%; conservar 4 hilos, con los demás ajustes de ese perfil |
| [Sueño/replay](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45008983356) | Checkpoint 3000: ocupación 0 y 0 registros de memoria lenta | En el entrenamiento real el sueño se salta por falta de registros; no es el cuello de botella |

El benchmark por etapas de [44941577031](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44941577031) también muestra que batch y serialización son insignificantes frente al cómputo: batch ronda 0,00014 s, optimizer 0,005–0,007 s y serializar un checkpoint ~0,025 s. En eager, forward y backward suman ~0,906 s de los ~0,932 s por paso. El objetivo debe ser abaratar la celda recurrente y su backward, no optimizar la carga de datos o el guardado.

### Correcciones concretas deducidas

1. **Desactivar los dos fastpaths que salieron perdedores.** Volver a `delta_pre_normalize=false` y `homeostasis_fused_scatter=false`. La prueba de paridad numérica confirma que las variantes preservan la salida/gradiente dentro de tolerancia, pero la medición de throughput demuestra que son más lentas; paridad no equivale a optimización.
2. **Restaurar el perfil CPU medido, no el perfil llamado “fast”.** El perfil actual `lumen_cpu_profile_fast.json` activa `foreach=true`, `interop_threads=1`, afinidad desactivada y los dos fastpaths perdedores. El perfil de cuatro hilos ganador medido usó 4 intra-op, 2 inter-op, MKLDNN activado, foreach desactivado y afinidad fijada. Esos valores son la referencia disponible más sólida para el siguiente entrenamiento.
3. **No retirar JEPA/repetición ni snapshots.** La ablación de auxiliares solo ganó ~3,7%; eliminarla cambia el objetivo por una ganancia pequeña. Desactivar snapshots no aceleró el perfil medido. Ninguna de las dos acciones resuelve el coste dominante.
4. **No tratar sueño/replay como culpable.** En el checkpoint inspeccionado no había memoria lenta consolidada y el trainer omitía el sueño; cambiar el intervalo de 500 pasos no recuperaría el tiempo perdido.
5. **No activar `torch.compile` indiscriminadamente en el trainer.** El benchmark corto anterior no representa el coste de integración observado en el A/B posterior. En el A/B real hubo una compilación inicial enorme y más pausas de compilación. Tras estabilizarse, el tramo final compilado sí rondó 1,8 pasos/s frente a ~0,87 del eager de ese A/B; el coste de arranque/recompilación implica un punto de equilibrio aproximado de 2,3k pasos. Solo tiene sentido en una campaña larga si se evita recompilar y se amortiza ese arranque; no en smoke tests ni bloques cortos.
6. **Atacar la recurrencia Delta, no sus órganos.** El coste está concentrado en forward/backward. La solución de arquitectura/ejecución debe reducir el número de operaciones pequeñas y nodos de autograd por token, idealmente fusionando el cuerpo recurrente y su backward en una unidad compilada/optimizada que conserve exactamente el orden temporal, las escrituras y las puertas. No convertir la recurrencia en una operación paralela aproximada.

### Perfil de partida que queda documentado

- 4 hilos intra-op y 2 inter-op.
- MKLDNN activado.
- AdamW `foreach=false`.
- Afinidad CPU fijada a los cuatro cores asignados.
- `delta_pre_normalize=false`.
- `homeostasis_fused_scatter=false`.
- JEPA, repetición, diversidad, router, Homeostasis y Delta conservados.
- Sueño en 500 pasos, sin cambiarlo mientras no haya registros de memoria lenta.

Este perfil es una selección basada en resultados existentes, no una afirmación de que la brecha arquitectónica ya esté resuelta. La prioridad pendiente es optimizar la ejecución recurrente de Delta y su backward sin cambiar la función de los órganos.

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

La recurrencia depende del estado previo, pero su actualización tiene estructura algebraica: para coeficientes de puerta/decaimiento fijados, el estado de cada instante puede expresarse como una suma causal ponderada. La nueva ruta calcula las interacciones K-K y Q-K con productos matriciales por lotes y resuelve las puertas sorpresa-dependientes mediante seis refinamientos de punto fijo. Así evita materializar el estado completo en cada token y reduce los nodos de autograd del bucle Python.

La ruta secuencial exacta se conserva mediante `parallel_scan=False` y se usa para la auditoría de equivalencia. La ruta paralela es una evaluación iterativa de la recurrencia no lineal, no una identidad exacta tras un número finito de refinamientos; por ello se exige comparar salidas, sorpresa, estado final y gradientes con tolerancias explícitas. Se mantiene la misma fórmula de escritura y el mismo rango de estado. El benchmark local CPU del módulo Delta (batch 8, contexto 128, dimensión 256, rango 64) mostró una mejora de orden de magnitud en forward+backward, pero esa cifra no se extrapola al modelo completo: el runner de Actions debe confirmarla.

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

## Protocolo de comparación inicial desde cero (histórico)

La campaña emparejada inicial desde step 0 ya cumplió su función de control. El resultado actual que gobierna la siguiente fase es el holdout de step 16.000: Lumen continúa como modelo principal y el Transformer queda como referencia histórica de comparación. No se exige duplicar cada continuación con un Transformer.

| Elemento | Condición inicial |
|---|---|
| Paso inicial | 0 para ambos en la campaña inicial (histórico) |
| Semilla | 42 |
| Batch | 8 |
| Longitud de secuencia | 128 |
| Datos | Mismos ejemplos y mismo orden de muestreo reproducible |
| Tokenizer | El mismo artefacto y SHA-256 |
| Runner | Mismo tipo de runner CPU |
| Presupuesto | Mismo número de pasos y tokens vistos |
| Evaluación | Mismo holdout congelado, una evaluación emparejada al final de cada bloque |
| Recuperación | Checkpoint independiente para cada modelo |

El protocolo inicial permitió validar los dos puntos de partida y construir una referencia emparejada. La campaña actual ya dispone de checkpoint Lumen 16.000, estado de AdamW recuperable y holdout congelado. La continuación autorizada es Lumen 16.000 → 21.000 con LR en meseta y decaimiento final no nulo.

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

La ejecución antigua [45008699964](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45008699964) falló al publicar su artefacto y queda como antecedente histórico. La campaña posterior [45014470774](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45014470774) publicó el corpus/tokenizer español congelado; la continuación [45021399600](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45021399600) verificó manifiesto, ficheros y las 2.431 filas del holdout antes de entrenar y evaluar step 16.000. Es ese artefacto posterior, no el intento fallido antiguo, el que utiliza la continuación actual.

El corpus resultante es principalmente SFT/QA/instrucciones, texto paralelo y material sintético propio; no debe presentarse como pretraining general de todo el español. El mismo conjunto y tokenizer se mantienen para reproducibilidad y comparación con el baseline. No se entrena con validation ni holdout.

## Estado de decisiones

| Decisión | Estado |
|---|---|
| Abandonar WISE-RONIN como línea activa | Adoptado |
| Lumen-Ronin frente a Transformer | Adoptado |
| Campaña emparejada inicial desde step 0 | Completada; referencia histórica |
| Preservar la semántica de los órganos durante optimización | Obligatorio |
| Medir cada etapa antes de modificarla | Obligatorio |
| Mantener sueño cada 500 pasos al inicio | Conservado provisionalmente |
| Homeostasis sin cambios funcionales iniciales | Conservado |
| Continuación Lumen 16.000 → 21.000 | Autorizada; holdout final incluido |

## Próximos hitos

1. Completar la continuación Lumen 16.000 → 21.000 con AdamW y buffers de los órganos preservados.
2. Confirmar en logs la meseta de LR a 3e-5 y el suelo coseno de 1,5e-5; comprobar que no se agota el scheduler antes del último paso.
3. Evaluar Lumen step 21.000 sobre el mismo holdout español congelado, informando accuracy y NLL por separado.
4. Comparar el resultado con Lumen 16.000 y con el Transformer 16.000 como referencias, sin confundir diferencia numérica con significación estadística.
5. Optimizar velocidad de Delta solo mediante cambios medidos que preserven la semántica y los órganos.

## Dos hipótesis de arquitectura: Lumen-Ronin frente al Transformer

Esta comparación no enfrenta dos nombres para la misma receta. Enfrenta dos maneras distintas de organizar el cómputo, la memoria y la adaptación.

El Transformer representa una de las grandes victorias de la ingeniería moderna: atención causal, bloques densos, operaciones matriciales altamente paralelizables y una ruta de escalado que aprovecha extraordinariamente bien aceleradores y grandes presupuestos de datos y cómputo. Su éxito es real y está demostrado. Pero su principio organizador dominante sigue siendo calcular relaciones entre posiciones mediante atención y transformar representaciones a través de bloques repetidos. Es una solución potente; no es la única forma imaginable de construir una máquina que aprenda.

Lumen-Ronin parte de otra pregunta: **¿podemos obtener una conducta útil organizando el modelo como un sistema de órganos con responsabilidades diferenciadas, estados persistentes y regulación interna, en lugar de confiar casi todo el trabajo a una pila uniforme de bloques de atención?**

No afirmamos que el cerebro sea un plano de ingeniería que haya que copiar literalmente, ni que la neurociencia haya demostrado que esta arquitectura vaya a ganar. Tomamos inspiración de principios funcionales —plasticidad, memoria, homeostasis, especialización, regulación de recursos y consolidación— y los convertimos en mecanismos computables que se pueden medir, refutar y modificar.

### Anatomía comparada

| Dimensión | Lumen-Ronin | Transformer de control |
|---|---|---|
| Principio organizador | Órganos especializados que cooperan sobre una representación compartida | Bloques causales repetidos de atención y red feed-forward |
| Dependencia temporal | Estado recurrente Delta actualizado secuencialmente por token | Atención causal sobre el contexto; las posiciones de una secuencia se procesan en paralelo dentro de cada bloque |
| Memoria de trabajo | Estado asociativo recurrente Delta, con escritura condicionada por sorpresa | Contexto representado por activaciones y claves/valores de atención durante el forward |
| Compresión de estado | Delta usa una proyección ortogonal de rango 64 sobre dimensión 256 para reducir el estado asociativo | No usa esta memoria recurrente de bajo rango; conserva la representación contextual del bloque de atención |
| Regulación de actividad | Homeostasis por unidad, EMA de actividad efectiva, sesgo adaptativo y Top-K | LayerNorm y transformaciones densas; no incorpora el mismo lazo homeostático adaptativo |
| Presupuesto | Presupuesto aprendido que distribuye energía entre Delta, memoria y cabezas; reserva de glucosa acotada que responde a la mejora de loss | No tiene un presupuesto energético explícito ni realimentación de glucosa |
| Memoria de largo plazo | Banco lento de slots, escritura selectiva por sorpresa y recuperación asociativa; componente elástico aprendido | No contiene un banco externo equivalente en esta implementación; el conocimiento se expresa en pesos y contexto |
| Procesamiento de salida | Router que mezcla una cabeza spline/KAN y una MLP | Proyección de salida desde la representación transformada |
| Aprendizaje auxiliar | JEPA/predicción latente, diversidad, router y penalización de repetición, además de next-token | Baseline de next-token; no incorpora esos órganos ni el mismo conjunto de pérdidas auxiliares |
| Vocabulario | Embedding vivo con contabilidad de uso y propuestas no destructivas de poda | Embedding estándar |
| Consolidación | Mecanismo de sueño/replay disponible en el trainer y banco de memoria | Sin mecanismo equivalente en el baseline |
| Control experimental | Diagnósticos de escritura, actividad efectiva, energía, sorpresa y estado de memoria | Diagnósticos convencionales de loss, parámetros y optimizador |

La tabla describe las implementaciones presentes en este repositorio, no todos los Transformers existentes ni todas las variantes posibles de Lumen. En particular, el módulo dual ZE V3.2 es opt-in y no forma parte del camino Lumen V1 que se compara por defecto.

### Los órganos de Lumen, uno por uno

**1. VivoEmbedding — vocabulario con observabilidad.** Mantiene un contador de uso por token y puede proponer candidatos sin uso para una poda posterior. No modifica el tokenizer ni elimina IDs por su cuenta: la integridad del vocabulario congelado tiene prioridad sobre una poda automática.

**2. EstadoDelta — memoria asociativa recurrente.** Proyecta K, V y Q, predice el contenido asociado a una clave desde el estado actual y calcula el error de predicción. La sorpresa derivada de ese error gobierna la escritura; una puerta aprendible y el presupuesto Delta modulan cuánto se incorpora. El estado recurrente conserva la misma regla de decaimiento y escritura de rango uno. La implementación de entrenamiento actual evalúa esa recurrencia mediante una forma cerrada causal y seis refinamientos de punto fijo sobre las puertas dependientes del estado; la ruta token a token permanece como referencia exacta. La versión comprime el espacio de claves mediante una proyección ortogonal fija de rango 64 (frente a dimensión 256), reduciendo el tamaño de la matriz de estado de 256×256 a 64×256 por capa.

**3. Homeostasis — regulación de actividad.** Selecciona las unidades Top-K y mide la actividad que realmente sobrevive a la puerta. Una media móvil exponencial de esa actividad efectiva ajusta un sesgo por unidad para aproximarse al objetivo de actividad. No es una simple máscara de sparsity: intenta cerrar un lazo de regulación sobre qué unidades están participando de verdad.

**4. Presupuesto — asignación de recursos.** Una capa aprendida reparte el presupuesto entre memoria Delta, memoria elástica/lenta y cabezas de salida. La reserva de glucosa es un buffer acotado: en cada ventana de comprobación, una mejora relativa de loss superior al umbral añade reserva; si no hay mejora, decae suavemente. La reserva escala la energía disponible, pero no cambia el learning rate. Es una hipótesis de control experimental, no una afirmación de que la glucosa biológica se comporte así en una neurona artificial.

**5. MemoriaElastica — plasticidad y banco lento.** Una MLP aporta una vía elástica cuya contribución se modula por sorpresa. Los eventos de sorpresa alta pueden escribirse en slots persistentes; la recuperación se realiza por similitud con las activaciones actuales y el uso de slots se sigue mediante estadísticas. La escritura está restringida al entrenamiento para evitar que la evaluación modifique la memoria entre prompts.

**6. Heads — especialización de la transformación final.** Un router aprendido mezcla dos rutas: una cabeza spline de tipo KAN, que interpola puntos aprendibles, y una MLP convencional. La salida se escala con el presupuesto asignado a las cabezas.

**7. JEPA y objetivos auxiliares.** Además de predecir el siguiente token, Lumen puede aprender una predicción de representación latente a varios pasos, regularizar diversidad de activaciones y uso del router, y penalizar patrones de repetición. Estas pérdidas se registran por separado porque una suma de objetivos distinta no debe confundirse con una comparación limpia de la arquitectura.

**8. Sueño y replay.** El trainer incluye una fase periódica de consolidación/replay. Su valor no se da por supuesto: se mide su coste y se comprueba que exista memoria que consolidar. Si no hay registros, no se debe atribuir al sueño una mejora o una penalización que no ha ocurrido.

### Neurociencia frente a escalado: la pregunta incómoda

La historia reciente de la IA ha premiado una estrategia muy eficaz: tomar una arquitectura general, aumentar datos, parámetros y cómputo, y explotar hardware cada vez más capaz. El Transformer ha sido extraordinariamente compatible con esa estrategia. La atención convierte gran parte del trabajo en álgebra matricial densa, y esa regularidad es una ventaja industrial inmensa.

Pero una ventaja de escalado no equivale a una teoría completa de la inteligencia.

Los cerebros no funcionan como una única operación homogénea repetida millones de veces. Son sistemas físicos, recurrentes, distribuidos y energéticamente restringidos, con memoria en múltiples escalas, mecanismos de adaptación local, regulación, especialización y coordinación. La neurociencia tampoco ofrece una receta cerrada: sigue habiendo debate sobre cómo emergen la memoria, la predicción, la atención, la conciencia y el aprendizaje. Lo que sí ofrece es un repertorio de problemas y principios que la ingeniería de redes puede explorar.

La provocación de Lumen es ésta: **quizá hemos confundido una arquitectura muy escalable con una arquitectura cognitivamente inevitable.** Quizá una parte del progreso futuro no consista únicamente en alimentar la misma familia de bloques con más cómputo, sino en descubrir mejores mecanismos de organización: cuándo escribir, qué conservar, qué regular, qué especializar y cómo asignar recursos según el estado del sistema.

Decir que el Transformer «nació obsoleto» sería ir demasiado lejos. No lo sabemos, y su rendimiento empírico impide despacharlo con una consigna. La crítica más seria es otra: el Transformer estándar no intenta reproducir muchos de los principios de organización que la biología ha desarrollado bajo restricciones extremas de energía, memoria y adaptación continua. Eso no lo invalida; delimita una decisión de diseño que merece ser cuestionada.

Lumen tampoco puede reclamar superioridad por llevar nombres de órganos. Una metáfora biológica no es una explicación, y una arquitectura más compleja puede ser peor: más lenta, más frágil, más difícil de entrenar y más propensa a introducir objetivos auxiliares que distorsionen la comparación. Si sus órganos no mejoran aprendizaje, robustez, memoria o eficiencia medidos, habrá que simplificarlos o descartarlos. La inspiración neurocientífica solo tiene valor cuando se traduce en mecanismos operativos y resultados reproducibles.

### Qué debe demostrar el experimento

La comparación emparejada desde cero es el primer juicio, no el veredicto final. Debe separar tres preguntas:

1. **Aprendizaje:** ¿Lumen reduce la pérdida de next-token y mejora las generaciones reales, o solo optimiza sus pérdidas auxiliares?
2. **Organización:** ¿los órganos hacen un trabajo observable —escrituras útiles, actividad homeostática estable, recuperación de memoria, asignación de presupuesto— o son complejidad decorativa?
3. **Economía:** ¿qué calidad obtiene cada modelo por paso, por token y por segundo de CPU? Una arquitectura inspirada en el cerebro que necesite diez veces más cómputo para obtener lo mismo no habrá resuelto el problema de eficiencia.

Para que la comparación sea honesta, hay que publicar por separado la cross-entropy común, las pérdidas propias de Lumen, los parámetros entrenables, tokens vistos, pasos/s, tokens/s, consumo de CPU/tiempo, métricas de generación y hashes de datos y tokenizer. También hay que declarar que el baseline actual es un Transformer pequeño y concreto —decoder causal de cuatro capas, dimensión 192, cuatro cabezas y FFN 576—, no «el Transformer» en abstracto.

**La tesis de trabajo no es que la biología tenga razón por ser biología. Es que sus soluciones contienen hipótesis de ingeniería que la escala, por sí sola, no responde.** Lumen debe ganarse su lugar en los logs, no en la retórica.

### Nueva ruta Delta paralela (22 de noviembre)

La implementación mantiene la recurrencia de decaimiento y escritura rank-one. Con coeficientes fijados, la actualización se resuelve como una suma causal; las lecturas se expresan mediante matrices de Gram K-K y Q-K. Como la sorpresa y la puerta dependen del estado previo, se aplican seis refinamientos de punto fijo. La versión secuencial exacta permanece disponible como referencia. Las pruebas comparan salida, sorpresa, estado y gradientes; la mejora de velocidad del módulo aislado observada en CPU local es una hipótesis de rendimiento que debe confirmarse en el runner completo. Homeostasis también evita ordenar los índices Top-K, porque su orden no interviene en la función.

---

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


### Aplicación directa del perfil medido (22 de noviembre)

Se actualizaron en `main` los dos perfiles CPU para que tanto el perfil normal como el alias `fast` utilicen 4 hilos intra-op, 2 inter-op, MKLDNN activado, AdamW `foreach=false`, afinidad CPU y ambos fastpaths experimentales desactivados. Commits: `9377772` y `ced7d54`. No se modificó la semántica de Delta ni de Homeostasis: los logs no contienen una implementación alternativa de la recurrencia que haya demostrado acelerar forward y backward manteniendo las mismas escrituras y gradientes. No se debe sustituir por una aproximación paralela.
