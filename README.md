# RONIN-LUMEN

**Arquitectura experimental de modelos pequeños: estado recurrente gobernado por sorpresa, actividad escasa, homeostasis, memoria de varios tiempos y consolidación fuera de línea. CPU primero.**

RONIN-LUMEN nace del refactor de WISE-RONIN V2. No es una carrera por entrenar el modelo más grande ni una promesa de superar a los modelos fundacionales. La pregunta es más concreta: **¿podemos diseñar un modelo pequeño que aprenda sin quedar atrapado en unas pocas respuestas, plantillas o atractores dominantes?**

> **Estado científico:** arquitectura implementada en primera versión; todavía no está validada como sistema de aprendizaje. Las hipótesis se tienen que ganar con entrenamiento real, holdouts congelados, controles y ablaciones.

## Índice

- [Qué estamos intentando cambiar](#qué-estamos-intentando-cambiar)
- [La arquitectura](#la-arquitectura)
- [Qué aporta la inspiración neurocientífica](#qué-aporta-la-inspiración-neurocientífica)
- [Qué tiene de propio el trabajo](#qué-tiene-de-propio-el-trabajo)
- [Estado real y próximos hitos](#estado-real-y-próximos-hitos)
- [Documentación y código](#documentación-y-código)
- [Reproducir el smoke test](#reproducir-el-smoke-test)
- [Principios de ingeniería](#principios-de-ingeniería)


## Protocolo longitudinal y trazabilidad arquitectónica

La comparación se desarrollará en bloques acumulativos de 500 pasos: 500, 1.000, 1.500, 2.000 y sucesivos. Cada hito debe poder reconstruirse y compararse con el Transformer sin depender de notas manuales. El workflow conserva manifiestos JSON por hito, hashes de checkpoints padre/hijo, configuración de arquitectura, actividad neuronal efectiva y eventos reales de consolidación.

Para cada ejecución se conservará un manifiesto que identifique el commit exacto, versión/configuración de cada arquitectura, módulos y flags activos, recuento de parámetros, hashes del corpus y tokenizer, semilla, presupuesto de tokens, checkpoint padre, hashes de checkpoints, estado de optimizador/scheduler/RNG, métricas de evaluación y coste de CPU/memoria.

En LUMEN se registrará además la actividad efectiva post-ReLU/Top-K por capa y por unidad, estado y escrituras Delta, actividad de memoria y consolidación realmente ejecutada. En el Transformer se registrarán su configuración completa, parámetros, estado de entrenamiento y métricas equivalentes disponibles. Las diferencias arquitectónicas —incluidas las pérdidas auxiliares y el ciclo de sueño— quedarán explícitas, no ocultas bajo una etiqueta de «mismo entrenamiento».

El workflow permite encadenar bloques de 500 pasos y reutiliza mediante caché el corpus, tokenizer y holdout ya preparados. Solo restaura el checkpoint padre exacto, nunca una versión aproximada. Cada bloque valida hashes, paso global, estado de reanudación y salud neuronal antes de habilitar el siguiente; un fallo detiene la cadena. La reanudación del Transformer y la evaluación por hitos ya están implementadas en esta rama, pendientes de validación integral en Actions.


## Configuración CPU: dos hilos

**La configuración de entrenamiento CPU queda fijada en 2 hilos intra-op de PyTorch** (`torch.set_num_threads(2)`). No es una preferencia estética ni una extrapolación: es la opción que ganó el primer benchmark controlado en los runners CPU estándar de GitHub Actions.

### Evidencia del benchmark

Ejecución [44932082530](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44932082530): LUMEN y Transformer medidos en el mismo runner, con batch 8, secuencia 128, 3 pasos de calentamiento y 20 pasos medidos por configuración. Se midió el cómputo de entrenamiento (forward, backward y actualización del optimizador); la carga de datos quedó fuera de esta prueba.

| Hilos intra-op | LUMEN (pasos/s) | Transformer (pasos/s) |
|---:|---:|---:|
| 1 | 0,968 | 4,441 |
| **2** | **1,530** | **9,477** |
| 4 | 0,842 | 5,860 |

Dos hilos dieron el mayor rendimiento en **ambos modelos**: aproximadamente un 58 % más de pasos/s que un hilo en LUMEN y un 113 % más en Transformer. Frente a cuatro hilos, la mejora fue de aproximadamente un 82 % en LUMEN y un 62 % en Transformer.

La conclusión operativa es clara: en este entorno de CPU compartida/limitada, aumentar el paralelismo a cuatro hilos no se tradujo en más trabajo útil por segundo. El coste de coordinación y la contención por recursos pueden explicar parte del resultado, pero el benchmark no aísla esas causas; por eso documentamos la medición, no una explicación causal que aún no hemos demostrado.

### Qué significa y qué no significa

- Los dos hilos son **intra-op**: controlan el paralelismo de las operaciones tensoriales de PyTorch. No significan dos procesos de entrenamiento ni dos hilos de inter-op.
- La decisión se aplica por igual a LUMEN y al Transformer de control, para mantener las condiciones comparativas.
- No cambia arquitectura, batch, longitud de secuencia, optimizador, semilla, corpus, tokenizer, checkpoints ni protocolo de holdout.
- Es una decisión de rendimiento para los runners CPU estándar de GitHub Actions, no una afirmación de que dos hilos sean óptimos en cualquier CPU o máquina.
- La medición inicial es un benchmark de cómputo corto, no el tiempo total de un entrenamiento con lectura de datos y checkpoints.

El benchmark de siete perfiles [44933048303](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44933048303) encontró que el perfil combinado superó a los otros seis en esa prueba corta: 1,628 pasos/s para LUMEN y 9,156 para el Transformer. Frente al perfil base de la misma tanda (1,108 y 7,862 pasos/s), supone +47,1 % y +16,5 %, respectivamente.

Ese perfil ganador mantiene 2 hilos intra-op y añade 2 inter-op, MKLDNN desactivado, `foreach` desactivado y afinidad del proceso a dos CPU disponibles, con variables OpenMP/MKL fijadas. **Es el mejor candidato medido en el microbenchmark, no una mejora todavía demostrada en un entrenamiento completo**: la prueba excluye carga de datos, guardado de checkpoints y variación entre runners. Por eso se conserva como perfil candidato, no como una verdad universal ni como motivo para cambiar silenciosamente una ejecución en curso.

La siguiente tanda [44937990141](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44937990141) compara siete hipótesis adicionales contra ese perfil de control: `torch.compile` por defecto, `torch.compile` con `reduce-overhead`, autocast BF16 CPU, AdamW fused, reutilización de gradientes (`set_to_none=False`), flush de denormals y control combinado. En la última consulta de estado seguía en cola; no hay resultados que permitan elegir ganador todavía. [Workflow de benchmark CPU](.github/workflows/benchmark-cpu-throughput.yml).



## Decisiones de entrenamiento, evaluación y calidad

Las decisiones que afectan a la validez del experimento quedan separadas de las optimizaciones de velocidad:

- **Comparación emparejada:** LUMEN y el Transformer usan el mismo corpus congelado, tokenizer, semilla 42, batch 8, contexto máximo 128, secuencia de tokens y pares de checkpoints con el mismo paso global. Sus diferencias arquitectónicas y objetivos auxiliares se declaran; no se finge que sean el mismo algoritmo.
- **Progresión verificable:** los hitos científicos son acumulativos y se publican en checkpoints de 500 pasos. Se reanuda el estado completo (pesos, optimizador, scheduler y RNG), no solo los pesos. Se valida el padre exacto, hashes, tokens vistos y salud antes de continuar.
- **Parada ante degradación:** cualquier `NaN`/valor no finito, colapso de actividad o fallo de integridad detiene la cadena. El checkpoint posterior al fallo no se considera válido; se conserva el último estado sano y no se salta por encima del hueco.
- **Datos y holdout separados:** corpus y tokenizer se preparan una vez y se reutilizan por hash. El holdout histórico de 500 filas permanece fuera del entrenamiento; GSM8K test se fija por hash, con 500 problemas limpios, 100 evaluados de forma determinista en el piloto inicial y 400 reservados. La evaluación usa generación greedy con máximo 64 tokens; resultados y respuestas se conservan para inspección.
- **Entrenamiento separado de inferencia:** replay/sueño, consolidación, poda de vocabulario y mutaciones de estado propias del entrenamiento no se ejecutan durante la generación de evaluación. La inferencia no conserva estado entre prompts ni altera el checkpoint. Así se evita que el evaluador cambie el modelo que pretende medir.
- **Calidad más allá de la loss:** además de pérdida y exactitud, se vigilan EOS, diversidad, repetición de n-gramas, bucles/plantillas y ejemplos representativos. Las pruebas generativas exploratorias usan una política de muestreo fijada (temperatura 0,8, top-p 0,9 y penalización de repetición 1,1) y se informan aparte del holdout greedy.
- **Ablaciones controladas:** se cambia una familia de mecanismos cada vez. El orden de investigación acordado empieza por replay/consolidación y después homeostasis; no se atribuye una mejora a un órgano sin una ablación que lo aísle.
- **Automatización con intención distinta:** los benchmarks de rendimiento pueden lanzarse automáticamente al integrar su PR para que la evidencia llegue pronto; los entrenamientos largos siguen siendo explícitos/manuales. Un merge documental o de benchmark no debe iniciar accidentalmente un entrenamiento.

### Incidencia longitudinal más reciente

La continuación 2.000 → 4.000 [run 44936542271](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44936542271) no produjo un hito válido completo: LUMEN registró valores no finitos desde el paso 3.100 y el detector detuvo la ejecución; el último estado sano observado en los logs fue el paso 3.000. El job fallido no publicó ese checkpoint. El Transformer sí llegó al paso 4.000, pero eso no crea un par emparejado válido en 4.000. La [PR #39](https://github.com/orbitalronin94/WISE-RONIN/pull/39) propone reproducir únicamente el tramo sano hasta 3.000 desde el padre 2.000 y, tras verificar la reproducción, ejecutar un único holdout emparejado en 3.000. Hasta que esa recuperación termine, no se declara un holdout de esta continuación ni se presenta el checkpoint reproducido como el original.


## Qué estamos intentando cambiar

La receta dominante de muchos modelos de lenguaje sigue una familia arquitectónica cuyo punto de inflexión moderno fue el Transformer de 2017: capas apiladas de autoatención y redes feed-forward, entrenadas para predecir el siguiente token. Esa arquitectura fue un avance decisivo; no sostenemos que sea obsoleta ni que escalarla sea inútil. Lo que rechazamos es asumir que **más parámetros, más datos y más cómputo son la única respuesta** a los problemas de estabilidad y degradación de un modelo pequeño.

WISE-RONIN V2 mostró un problema práctico: reducir la pérdida no garantizaba conservar variedad ni conducta útil. El modelo podía adquirir una voz reconocible y, al continuar entrenando, derivar hacia respuestas repetitivas, plantillas o un atractor KOAN. LUMEN convierte esa experiencia en una pregunta de arquitectura: ¿qué mecanismos de escritura, actividad, memoria y consolidación pueden limitar esa deriva?

## La arquitectura

```text
Token
  -> Embedding BPE
  -> Presupuesto por token
  -> Estado Delta recurrente gobernado por sorpresa
  -> Homeostasis + Top-K
  -> Memoria elástica / banco lento
  -> Router
       -> MLP de lenguaje
       -> KAN de funciones spline
  -> Logits de token
  -> Objetivos auxiliares latentes durante entrenamiento

En intervalos de entrenamiento:
  -> replay / consolidación offline ("sueño")
  -> reinicio de estados rápidos
```

| Mecanismo | Función prevista | Pregunta experimental |
|---|---|---|
| Estado Delta con puerta de sorpresa | Actualizar el estado cuando el error de predicción lo justifica; decaer si no | ¿Reduce escrituras inútiles sin perder contexto? |
| Homeostasis | Regular la frecuencia de activación de las unidades | ¿Evita que unas pocas neuronas monopolicen la representación? |
| Top-K | Mantener una fracción pequeña de activaciones | ¿Mejora selectividad y coste sin destruir información? |
| Presupuesto de energía | Repartir capacidad entre estado, memoria y cabeza | ¿Evita que un subsistema absorba todo el cómputo? |
| Memoria elástica y lenta | Separar adaptación rápida de patrones consolidados | ¿Reduce interferencia y olvido? |
| Router MLP/KAN | Separar generación lingüística de operaciones | ¿Aporta especialización medible o colapsa a una sola ruta? |
| Sueño / replay | Reprocesar registros internos sin nuevos ejemplos externos | ¿Consolida sin degradar lo aprendido? |
| Predicción latente JEPA-style | Añadir una señal auxiliar sobre representaciones futuras | ¿Aporta algo por encima de la pérdida de token? |

El modelo actual tiene **2.281.841 parámetros** según el recuento integrado registrado en CI. Cumple el techo de 6 millones, pero todavía no alcanza el objetivo de diseño de 3–4 millones. El tamaño final en disco de 20–30 MB tampoco está demostrado por ese recuento.

### Qué aporta la inspiración neurocientífica

La inspiración no es decorativa: se usa para formular mecanismos concretos y preguntas falsables. La correspondencia entre biología y código es una **abstracción de ingeniería**, no una simulación fiel del cerebro. Las fuentes y sus límites están desarrollados en [Neurociencia convertida en decisiones de arquitectura](docs/NEUROCIENCIA_Y_ARQUITECTURA_ES.md).

En resumen:
- Plasticidad homeostática: regular actividad para que el aprendizaje no empuje el sistema hacia actividad descontrolada o silencio.
- Codificación escasa: representar cada entrada con pocas unidades activas.
- Codificación predictiva y error de predicción: usar la discrepancia entre lo esperado y lo observado como señal de actualización.
- Sistemas de memoria complementarios: distinguir adaptación rápida e integración lenta.
- Sueño y homeostasis sináptica: explorar consolidación y renormalización offline.

No todo procede de neurociencia: la regla Delta tiene una genealogía propia en modelos secuenciales modernos; KAN es una propuesta matemática; JEPA es una línea de aprendizaje de representaciones. La documentación identifica cada procedencia sin apropiarse del trabajo ajeno.

## Qué tiene de propio el trabajo

La contribución que reivindicamos es **la selección, traducción e integración de estas ideas en un sistema pequeño, ejecutable en CPU y sometido a restricciones reales de memoria, tiempo y reproducibilidad**. No afirmamos haber inventado la homeostasis, la codificación escasa, la consolidación, la predicción latente, KAN ni las reglas Delta.

La decisión de diseño es distinta de limitarse a reducir una Transformer o aumentar su tamaño: repartir explícitamente la computación entre estado recurrente, memoria rápida/lenta y rutas especializadas; controlar la actividad; y tratar la degradación como objetivo de arquitectura y evaluación, no solo como un problema de limpieza del dataset.

La prueba del mérito no será el lenguaje grandilocuente: será que las ablaciones demuestren qué componente mejora qué métrica, a qué coste y en qué condiciones.

## Estado real y próximos hitos

<badge>Arquitectura integrada · datos en preparación</badge>

- **PR #13 — integrada:** arquitectura, módulos, trainer CPU, checkpoint/resume y smoke tests. [Ver PR #13](https://github.com/orbitalronin94/WISE-RONIN/pull/13).
- **PR #14 — abierta:** gates de datos, procedencia, licencias y tokenizer. [Ver PR #14](https://github.com/orbitalronin94/WISE-RONIN/pull/14).
- **Staging externo:** workflow [44890014413](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44890014413) completado. Artefacto de 19.000 candidatos; auditoría automática: 0 duplicados normalizados, 0 coincidencias exactas con holdout y 0 coincidencias largas de subcadena. El holdout conserva 500 filas y 55 prompts únicos.
- **No es corpus final:** faltan 5.000 filas propias curadas y 1.000 filas de código con licencia comprobada. Las fuentes externas descargadas siguen con `license_reviewed=false` y `rights_confirmed=false`; el resultado léxico no sustituye revisión humana ni semántica.
- **Tokenizer:** todavía no hay tokenizer final construido a partir del corpus aprobado.
- **Campaña longitudinal:** bloques encadenables de 500 pasos, con LUMEN (2.281.841 parámetros) frente a Transformer (2.295.168), datos/tokenizer/semilla y presupuesto de tokens equiparados. El workflow cachea el corpus, tokenizer y holdout, y conserva solo el par de checkpoints padre exacto para evitar descargas y almacenamiento redundantes. [Protocolo comparativo](docs/PILOTO_COMPARATIVO_ES.md) · [Workflow longitudinal](.github/workflows/pilot.yml).
- **Primer bloque comparativo:** el workflow [44900623884](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44900623884) completó los entrenamientos de 500 pasos de LUMEN y Transformer, preparó el holdout GSM8K, ejecutó la evaluación y publicó un artefacto. El diagnóstico mostró colapso progresivo de actividad en capas profundas de LUMEN; el siguiente bloque queda condicionado a corregirlo y validar la medición post-Top-K.
- **Producción:** sigue bloqueada por las revisiones de licencias, las 5.000 filas propias, las 1.000 filas de código auditadas y el tokenizer final.

El piloto provisional no reemplaza la campaña 80/20 definitiva. Sus resultados se publicarán con sus límites y no se confundirán con los del corpus de producción.

## Documentación y código

El [índice de documentación](docs/README.md) organiza las especificaciones, auditorías, datos y experimentos. Entradas principales:

- [Documento maestro V1](DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md)
- [Especificación V1](SPEC_RONIN_LUMEN_V1.md)
- [Neurociencia y decisiones de arquitectura](docs/NEUROCIENCIA_Y_ARQUITECTURA_ES.md)
- [Protocolo comparativo, longitudinal y de trazabilidad](docs/PILOTO_COMPARATIVO_ES.md)
- [Auditoría de la especificación](docs/SPEC_AUDIT.md)
- [Auditoría del bucle de entrenamiento](docs/TRAIN_LOOP_REFERENCE_AUDIT.md)
- [Extensión experimental V3.2 ZE](docs/RONIN_LUMEN_V3_2_ZE.md)
- [Procedencia y preparación de datos](data/README.md)
- [Contrato del corpus propio (español)](data/owner/README_ES.md) · [versión original](data/owner/README.md)
- [Workflow de entrenamiento CPU](.github/workflows/train.yml)
- [Workflow del piloto comparativo](.github/workflows/pilot.yml)
- [Workflow de preparación de datos](.github/workflows/prepare-data.yml)

## Reproducir el smoke test

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
python -m train.train --dummy-m1 --steps 10 --phase 1
python -m train.train --dummy-m1 --resume checkpoints/step_10.pt --steps 10 --phase 1
```

El modo dummy solo comprueba forward, optimización, checkpoint y reanudación. Está limitado a diez pasos y no usa el corpus real.

El entrenamiento de un bloque con datos reales requiere el corpus y el tokenizer aprobados:

```bash
python -m train.train --steps 500 --phase 1
```

No ejecutar ese comando hasta cerrar los gates de datos, licencias, tokenizer y holdout. Los workflows de GitHub Actions son la vía reproducible prevista para los bloques CPU.

## Principios de ingeniería

### Carmack–Naka–Chapman: diseñar desde la restricción

- **Carmack:** comprender el coste real de memoria, caché, ancho de banda y operaciones antes de añadir complejidad.
- **Yuji Naka:** exprimir cada byte, rutina y recurso disponible mediante reutilización y diseño específico para la máquina.
- **Colin Chapman:** simplificar antes de añadir peso.

Aplicado aquí: CPU como objetivo, no como plan B; arquitectura antes que fuerza bruta; cada parámetro y cada fila de datos debe justificar su coste; y ningún benchmark se acepta sin controles, holdout y límites declarados.

---

*RONIN-LUMEN no pretende imitar un cerebro ni demostrar que una arquitectura pequeña vence a una grande. Pretende convertir ideas sobre aprendizaje, estabilidad y memoria en mecanismos computables, baratos y falsables.*
