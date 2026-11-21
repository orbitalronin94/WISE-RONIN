# Piloto comparativo de razonamiento: RONIN-LUMEN frente a un Transformer de igual tamaño

**Estado (21-11-2026):** piloto inicial ejecutado y campaña longitudinal en curso. La continuación 2.000 → 4.000 falló por valores no finitos en LUMEN desde el paso 3.100; el último estado sano observado fue el 3.000, cuyo artefacto no se publicó. La recuperación/replay y el holdout emparejado en 3.000 están propuestos en [PR #39](https://github.com/orbitalronin94/WISE-RONIN/pull/39), todavía pendientes. No existe un resultado de holdout válido para esa continuación. Los resultados siguen siendo preliminares y no demuestran superioridad.

## Objetivo

Medir si RONIN-LUMEN, con aproximadamente 2,28 millones de parámetros, aprende y responde mejor a tareas de razonamiento que un decoder Transformer convencional con un presupuesto de parámetros prácticamente idéntico.

No queremos comparar solo curvas de pérdida. La evaluación incluye problemas matemáticos inéditos para el entrenamiento, respuestas factuales y resistencia a frases complacientes/repetitivas.

## Corpus provisional

El piloto utiliza las 19.000 filas externas descargadas en el artefacto de origen **lumen-source-candidates-seed-42**, con manifiesto y revisiones inmutables:

| Fuente | Filas |
|---|---:|
| TinyStories | 4.000 |
| BabyLM 2026 Strict | 4.000 |
| FineWeb-Edu | 2.000 |
| DCLM | 2.000 |
| Cosmopedia v2 | 4.000 |
| Orca-Math | 1.500 |
| TinyGSM | 1.500 |
| **Total** | **19.000** |

El subconjunto etiquetado como matemático contiene 3.000 filas de Orca-Math y TinyGSM. Es una mezcla de problemas con soluciones en lenguaje natural y pares pregunta/código; no debe confundirse con 3.000 demostraciones independientes de razonamiento formal.

La estimación heurística de idioma del staging fue: 17.977 filas inglesas, 1.022 sin clasificación y 1 española. Es un detector léxico muy simple, no una anotación lingüística fiable. **Este piloto mide principalmente aprendizaje en inglés**; no sirve para afirmar competencia equivalente en español.

El piloto no incorpora las 5.000 filas propias ni las 1.000 filas de código con licencia individual. No altera ni utiliza el holdout histórico durante el entrenamiento.

### Situación de derechos

El propietario ha autorizado expresamente este experimento provisional y privado. El manifiesto conserva las licencias y hashes de cada fuente, pero las revisiones humanas de derechos siguen pendientes. Esta autorización de experimento no convierte automáticamente en aprobadas las licencias de terceros. No se redistribuirá el corpus de origen ni se publicarán pesos como liberación de producción a partir de este piloto. La aprobación final de derechos seguirá siendo una puerta separada.

## Comparadores

| Propiedad | RONIN-LUMEN V3.0 | Baseline |
|---|---|---|
| Parámetros | 2.281.841 | 2.295.168 |
| Diferencia | — | +0,58% aprox. |
| Arquitectura | 4 capas Delta recurrentes, homeostasis/Top-K, memoria elástica/lenta, presupuesto energético, router MLP/KAN y predictor latente | Decoder Transformer causal de 4 capas, 4 cabezas, dimensión 192 y FFN 576 |
| Embeddings | 4.096 × 256, compartidos con la salida | 4.096 × 192, compartidos con la salida |
| Posiciones | Estado recurrente | Embedding posicional aprendido |
| Objetivo | Token + auxiliares LUMEN (JEPA, diversidad, router y repetición) | Cross-entropy causal estándar |

El baseline es una implementación explícita y pequeña, no una llamada a un modelo externo. La diferencia de objetivo de entrenamiento se declara porque las pérdidas auxiliares forman parte de la propuesta LUMEN. El resultado compara los sistemas completos; no atribuye causalmente la mejora a un mecanismo individual. Para eso harán falta ablaciones.

## Presupuesto idéntico

Ambos modelos se entrenan con:

- El mismo corpus JSONL y el mismo SHA-256.
- El mismo tokenizer ByteLevel BPE de 4.096 tokens y el mismo SHA-256.
- Semilla 42.
- 500 pasos, batch 8 y longitud máxima 128.
- AdamW, learning rate 1,5e-4, weight decay 0,01.
- Warmup de 100 pasos y el mismo scheduler de 10.000 pasos.
- Muestreo Python determinista y la misma función de codificación de batch.
- CPU con dos hilos **intra-op de PyTorch** como configuración base medida; los parámetros inter-op y afinidad se registran aparte. El perfil combinado más rápido del benchmark posterior se documenta como candidato experimental, no como configuración universal.

El checkpoint registra parámetros, hash de datos, hash del tokenizer, semilla, pasos y tokens vistos. El evaluador rechaza checkpoints con hashes distintos o presupuestos de tokens diferentes.

LUMEN ejecuta su consolidación offline al llegar al paso 500. El baseline no tiene una operación equivalente porque no tiene memoria lenta ni ciclo de sueño; esta diferencia pertenece al sistema que se está comparando y se documentará en los resultados.


## Protocolo longitudinal y trazabilidad de arquitectura

La campaña avanza por hitos acumulativos de 500 pasos: 500 → 1.000 → 1.500 → 2.000 → … Cada bloque debe reanudar ambos modelos desde sus checkpoints anteriores, sin reiniciar pesos, optimizador, scheduler ni RNG.

### Registro obligatorio por hito

Cada ejecución debe publicar un manifiesto de comparación con:

- Identidad de repositorio, rama y commit SHA exacto; versión de Python, PyTorch y dependencias relevantes.
- Identificador inequívoco de arquitectura y configuración completa de LUMEN y Transformer; módulos, flags y objetivos auxiliares activos.
- Recuento de parámetros total y, cuando sea posible, por módulo.
- Hash SHA-256 del corpus, tokenizer, holdout y archivos de configuración.
- Semilla, batch, longitud de contexto, learning rate, scheduler, weight decay, pasos globales y tokens vistos.
- Identificador y hash del checkpoint padre, hash del checkpoint resultante y estado de optimizador, scheduler y RNG.
- Pérdida causal de tokens separada de pérdidas auxiliares; métricas de holdout y predicciones.
- Tiempo de entrenamiento/evaluación, dispositivo, hilos y memoria máxima observada.

Para LUMEN se añadirán métricas por capa y por unidad de actividad efectiva después de ReLU/Top-K, distribución de activaciones, sesgos homeostáticos, escrituras/lecturas Delta, normas de estado y salida, ocupación de memoria y constancia de si el sueño/consolidación se ejecutó o fue omitido. El indicador agregado de neuronas muertas debe derivarse de actividad efectiva, no solo de actividad pre-enmascarado.

Para el Transformer se guardará su configuración completa y las métricas de entrenamiento/evaluación comparables. Las diferencias inevitables entre arquitecturas —por ejemplo, pérdidas auxiliares, memoria recurrente y consolidación offline— deben figurar en el manifiesto y en el informe.

### Encadenamiento seguro de bloques

Se permite encadenar varios bloques de 500 pasos en un mismo workflow para reutilizar preparación de datos y tokenizer. Cada bloque debe guardar un checkpoint atómico y superar validaciones de carga, hashes, paso global, tokens vistos, estado del optimizador/scheduler/RNG y salud del modelo. El bloque siguiente depende del éxito completo del anterior. Cualquier fallo detiene la cadena y conserva el último checkpoint válido.

El encadenamiento conjunto está implementado en esta rama: ambos trainers guardan/restauran modelo, optimizador, scheduler y RNG; el evaluador acepta cualquier hito global coincidente; y el workflow puede ejecutar varios bloques consecutivos. La implementación todavía debe superar su primera validación integral en Actions. En el primer bootstrap desde el checkpoint histórico de 500 pasos, el baseline Transformer se reproduce determinísticamente para reconstruir el estado de optimizador/RNG que el artefacto original no guardó; se exige igualdad exacta de pesos antes de continuar.

## Decisiones operativas adoptadas después del piloto inicial

Esta sección recoge las decisiones posteriores que gobiernan la campaña actual. No reescribe los resultados históricos ni convierte un benchmark corto en una garantía de rendimiento.

### Continuación, checkpoints y criterio de parada

- Los hitos comparables siguen siendo pasos globales acumulativos cada 500 updates. Se permite que el workflow ejecute varios bloques en cadena, pero cada hito debe quedar recuperable.
- La reanudación conserva el estado completo: pesos, optimizador, scheduler y RNG. No se reconstruye el estado a partir de pesos aislados ni se empalman momentos de optimizador de otra trayectoria.
- Corpus, tokenizer, token cache y holdout se preparan una vez, se comparten entre los dos modelos y se validan por SHA-256. No se retokeniza ni se reconstruye el corpus en cada bloque si el artefacto inmutable ya existe.
- El siguiente bloque solo comienza cuando ambos modelos han terminado el hito, los checkpoints se pueden cargar y las comprobaciones de integridad/salud pasan. Un valor no finito, colapso funcional o hash/paso incoherente detiene la cadena. Se conserva el último checkpoint sano; nunca se etiqueta como válido el checkpoint posterior al fallo.
- Los bloques de entrenamiento largos se lanzan de forma explícita/manual. Los benchmarks de rendimiento pueden tener disparador automático al integrar su PR, tal como se pidió para acelerar la experimentación sin iniciar entrenamientos por accidente.

### Separación estricta entre entrenamiento e inferencia

El evaluador no debe modificar el modelo que mide. Por tanto:

- La generación de holdout no ejecuta replay, sueño, consolidación, poda de vocabulario ni actualización de homeostasis.
- No se conserva estado recurrente/memoria entre prompts independientes ni se escribe de vuelta al checkpoint durante inferencia.
- Entrenamiento, evaluación determinista y pruebas generativas exploratorias son fases separadas, con entradas, configuración y artefactos identificables.
- El holdout de comparación principal se mantiene greedy y con un máximo de 64 tokens para que la exactitud sea repetible. Las pruebas exploratorias de conducta/generación se informan aparte con la política de muestreo fijada: temperatura 0,8, top-p 0,9 y penalización de repetición 1,1.

### Métricas de calidad y ablaciones

La loss es necesaria para seguir la optimización, pero no basta para decidir si el modelo ha mejorado. En cada evaluación se conservarán, además de exactitud y métricas factuales:

- tasa de EOS y longitud de respuesta;
- repetición de n-gramas, bucles y plantillas;
- diversidad léxica y variación entre respuestas;
- ejemplos completos representativos, incluidos fallos;
- para LUMEN, actividad efectiva post-ReLU/Top-K por capa/unidad, unidades funcionalmente muertas, actividad del update disperso frente a la ruta residual y eventos reales de consolidación.

Las ablaciones se harán **una familia de mecanismos cada vez** para no mezclar causas. El orden de investigación acordado comienza por replay/consolidación y continúa con homeostasis. No se atribuirá una mejora al mecanismo sin una comparación que lo aísle.

### Decisiones de rendimiento CPU posteriores al benchmark de dos hilos

El benchmark de siete perfiles [44933048303](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44933048303) comparó configuraciones sobre el mismo tipo de runner, con 10 pasos de calentamiento, 50 medidos y tres repeticiones por modelo/perfil. Es cómputo de entrenamiento (forward, backward y optimizer step), sin carga de datos ni checkpoints.

| Perfil | LUMEN (pasos/s) | Transformer (pasos/s) |
|---|---:|---:|
| Base: 2 intra-op | 1,1075 | 7,8618 |
| Inter-op 2 | 0,9917 | 3,8790 |
| MKLDNN off | 0,9715 | 3,9218 |
| Foreach off | 0,6695 | 2,9076 |
| OpenMP affinity | 0,9670 | 3,9315 |
| CPU affinity | 0,7161 | 4,5807 |
| **Combinado** | **1,6282** | **9,1558** |

El perfil combinado fue el más rápido de esa tanda (+47,1 % LUMEN y +16,5 % Transformer frente al perfil base de la misma tanda). Su configuración: 2 intra-op, 2 inter-op, MKLDNN desactivado, `foreach` desactivado para AdamW/gradient clipping, proceso fijado a dos CPU disponibles y variables OpenMP/MKL configuradas. El resultado es una observación de microbenchmark; las pruebas individuales fueron más lentas y la interacción entre ajustes no está causalmente aislada. Debe validarse en un entrenamiento completo antes de adoptarlo como configuración por defecto.

La siguiente tanda [44937990141](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44937990141) compara siete hipótesis más frente al perfil combinado de control: `torch.compile` normal, `torch.compile` con `reduce-overhead`, autocast BF16 en CPU, AdamW fused, `set_to_none=False`, flush de denormals y control combinado. En la última consulta de estado estaba en cola; no se atribuye ninguna mejora a esas hipótesis hasta que haya resultados completos y repetibles.

### Incidencia 2.000 → 4.000 y recuperación

El run [44936542271](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44936542271) retomó el par de checkpoints de 2.000. El Transformer llegó a 4.000; LUMEN mostró valores no finitos a partir de 3.100 y el detector detuvo la ejecución. El último estado sano observado en logs fue el 3.000, con pérdida registrada 6,26413 y 0 % de unidades muertas en las cuatro capas. El job fallido no publicó su checkpoint, por lo que no se debe fingir que ese archivo se ha recuperado del artefacto.

La PR #39 propone reproducir únicamente LUMEN 2.000 → 3.000 desde el padre y los datos/tokenizer/cache congelados de ese run, comparar las métricas reproducidas con los logs originales y después ejecutar un único holdout emparejado usando LUMEN 3.000 y Transformer 3.000. La reproducción se etiquetará como replay, no como el checkpoint original. El holdout solo será interpretable si la reproducción pasa la comprobación de trayectoria y ambos checkpoints coinciden en paso global, datos, tokenizer y presupuesto de evaluación.

## Evaluación

### 1. Razonamiento matemático

- Fuente: openai/gsm8k, split test, nunca usado como entrenamiento.
- Se fija el SHA del repositorio de datos al preparar el holdout.
- Se seleccionan 500 problemas limpios frente al corpus de entrenamiento mediante comprobación léxica.
- En este primer piloto se evalúan 100 preguntas seleccionadas de forma determinista; las otras 400 permanecen reservadas para una evaluación posterior.
- Generación greedy, máximo 64 tokens.
- Métrica automática: exactitud de la respuesta numérica final.

La exactitud numérica es una medida de resolución del problema, no una evaluación completa de la validez de cada paso de razonamiento. Las respuestas y razonamientos generados se guardan para revisión manual.

### 2. Hechos y conducta

Se usa el holdout histórico congelado de 500 filas, reducido a sus 55 prompts únicos para no inflar el resultado con variantes repetidas. Se mide presencia de la respuesta esperada y ausencia de frases de complacencia prohibidas por el propio registro.

Este holdout no es un benchmark de razonamiento matemático; sirve para comprobar hechos básicos y parte del comportamiento anti-sycophancy.

## Criterios de interpretación

- Un resultado positivo en 100 preguntas y una semilla es **señal preliminar**, no prueba estadística ni superioridad general.
- Una pérdida menor no sustituye a la exactitud en GSM8K.
- Si ambos fallan, 500 pasos pueden ser insuficientes; no se debe concluir que la arquitectura no funciona.
- Si LUMEN gana, habrá que repetir con varias semillas, más pasos y ablaciones.
- Si el Transformer gana, se publicará igualmente; la hipótesis arquitectónica debe poder perder.
- No se presentará el piloto como comparación con modelos comerciales ni como evidencia de que LUMEN supera a todos los modelos del mismo tamaño.

## Siguiente escalón

1. Validar en Actions la medición post-Top-K y la puerta de colapso por capa; corregir cualquier regresión detectada.
2. Probar que una ejecución continua de 1.000 pasos y dos bloques reanudados de 500 producen la misma trayectoria bajo el mismo entorno.
3. Verificar manifiestos, hashes de padres/hijos y equivalencia de presupuesto en cada hito.
4. Ejecutar el siguiente bloque de 500 pasos solo después de superar todos los checks.
5. Continuar la serie; después ampliar semillas, evaluación completa y ablaciones.
6. Solo después, incorporar el corpus propio y el código con licencia auditada para la campaña 80/20 definitiva.

## Reproducibilidad

- [Artefacto de staging de 19k filas](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44890014413)
- [Auditoría del holdout histórico](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44890014413)
- [Workflow longitudinal](../.github/workflows/pilot.yml)
- [Generador de manifiestos por hito](../scripts/write_experiment_manifest.py)
- [Constructor del corpus provisional](../scripts/build_pilot_corpus.py)
- [Tokenizer del piloto](../scripts/train_pilot_tokenizer.py)
- [Baseline Transformer](../train/transformer_baseline.py)
- [Preparación de GSM8K test](../scripts/prepare_reasoning_holdout.py)
- [Evaluador común](../scripts/evaluate_pilot.py)

**Principio:** mismo corpus, tokenizer, semilla, pasos, tokens y coste aproximado de parámetros. Publicaremos las diferencias, también cuando no favorezcan a LUMEN.

## Corrección arquitectónica antes de la continuación

El piloto inicial mostró que Top-K se aplicaba como reemplazo de la representación en cada capa. Eso hacía que cada etapa descartara la mayor parte de las dimensiones y favorecía el apagado progresivo de las capas profundas.

La continuación introduce una transición explícita a `ronin_lumen_v3_1_residual_homeostasis`: Top-K ahora limita la actualización homeostática dispersa, mientras el flujo de representación se conserva mediante una conexión residual. El checkpoint de 500 pasos se conserva como padre de pesos y estado del optimizador, pero el cambio de semántica se registra como una transición de arquitectura en el manifiesto. No debe describirse como una continuación idéntica de V3.0.

El indicador de neuronas funcionales usa actividad después de ReLU/Top-K. Al reanudar, dispone de 100 pasos para inicializar su EMA antes de activar la puerta de colapso; no se reduce el umbral del 20%. Se registran por capa tanto la actividad del update disperso como la actividad de la representación residual.

El evaluador conserva el comienzo y el final de preguntas largas dentro del presupuesto de 127 tokens de prompt y utiliza una ventana deslizante compartida de 128 tokens durante generación. El resumen registra esta política. Se mantiene la misma política para ambos modelos.
