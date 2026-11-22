# Experimento alternativo: neuroplasticidad y chunking adaptativo

## Propósito

Evaluar una versión experimental de Lumen inspirada en seis familias de mecanismos del libro *Cerebro a código*, sin alterar `main`, desactivar órganos ni reutilizar checkpoints históricos. La comparación emparejada enfrenta esta variante de Lumen con el Transformer de control, ambos inicializados desde el paso 0.

## Qué se reutiliza y qué se añade

No se vuelven a implementar mecanismos que ya existen en el repositorio:

| Principio | Implementación en esta rama |
|---|---|
| Delta paralelo | Se conserva el solucionador paralelo por punto fijo existente. No se afirma que sea un parallel scan asociativo exacto. |
| Predictive coding | Se conserva la sorpresa y el umbral que regula las escrituras Delta. |
| Homeostasis | Se mantienen los cuatro órganos homeostáticos y sus Top-K. |
| Neuromodulación | Dolor y placer siguen regulando el presupuesto; además modulan las trazas persistentes. |
| Replay/consolidación | Se habilita el sueño/replay existente a los pasos 500 y 1000 si hay registros en memoria. |
| Chunking adaptativo | Nuevo controlador compartido: sorpresa alta reduce el tamaño de chunk; sorpresa baja permite chunks mayores. El resumen se añade como residual acotado. |
| Trazas de elegibilidad | Nuevo estado persistente por capa Delta, formado por coincidencias clave/escritura y conservado entre minibatches. La señal retardada es placer menos dolor, acotada y de ganancia pequeña. |

## Guardarraíles

- Ningún órgano se desactiva.
- El chunking no sustituye la secuencia token a token: añade un resumen residual limitado al 2,5 % de la norma de entrada.
- Las trazas se almacenan como buffers, se actualizan sin gradiente y se limitan antes de acumularse. Esta es una aproximación asociativa del principio de elegibilidad: **no** implementa todavía una regla local de tres factores que actualice directamente cada parámetro sináptico.
- Los estados recurrentes transitorios se reinician en cada minibatch; las trazas de elegibilidad sí persisten y forman parte del checkpoint.
- El modulador de elegibilidad está acotado a ±0,25.
- El Transformer no recibe chunking, trazas ni órganos de Lumen: permanece como baseline independiente.
- El experimento usa inicialización fresca. No carga pesos, optimizador ni RNG de campañas anteriores.

## Protocolo emparejado

- Lumen: arquitectura `ronin_lumen_v3_3_neuroplasticity`.
- Transformer: baseline decoder-only existente.
- Paso inicial: 0.
- Horizonte: 1.000 actualizaciones por modelo.
- Seed: 42.
- Batch: 8.
- Contexto: 128 tokens.
- Corpus y tokenizer: artefacto español v2 compartido, idénticos para ambos modelos.
- Checkpoints: cada 250 pasos; recuperación cada 100 pasos para Lumen.
- Logs: cada 100 pasos.
- Replay: habilitado; se registra si no hay memoria suficiente para ejecutarlo.
- Comparación: loss, NLL, tokens vistos, tiempo por bloque, pasos/s, memoria, estabilidad numérica y, después, holdout común.

El corpus v2 debe terminar su construcción y superar las comprobaciones de manifiesto, hashes y splits antes de que arranque el entrenamiento. No se permite sustituirlo silenciosamente por v1.

## Interpretación

Este es un paquete combinado: si gana o pierde, el resultado no identifica qué mecanismo causó la diferencia. Una victoria solo justificaría benchmarks de ablación posteriores (trazas, chunking y replay por separado). No se promociona nada a `main` sin tests, revisión de checkpoints y evaluación común.

La ejecución de 1.000 pasos es un piloto de aprendizaje y rendimiento, no evidencia suficiente de generalización. El holdout se mantiene separado y debe evaluarse después de que ambos modelos terminen.
