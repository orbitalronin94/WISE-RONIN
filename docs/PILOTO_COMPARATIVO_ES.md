# Piloto comparativo de razonamiento: RONIN-LUMEN frente a un Transformer de igual tamaño

**Estado:** primer bloque de 500 pasos ejecutado; campaña longitudinal por bloques de 500 definida. Los resultados son preliminares y no demuestran superioridad.

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
- CPU con dos hilos.

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
