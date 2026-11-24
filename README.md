# Lumen-Ronin
### Una arquitectura pequeña. Una pregunta enorme: ¿y si aprender no fuera solo escalar?

> **LUMEN es un perro.** Un perro de Pavlov que aprende lento, pero aprende. Que saliva, que tiene hambre, que le duele, que le da placer. Y un perro, aunque sea más lento que un loro para repetir, es más listo que un loro para aprender.

Lumen-Ronin es un experimento de arquitectura recurrente, CPU-first y de pocos millones de parámetros. En lugar de limitarse a apilar bloques Transformer y aumentar cómputo, explora otra pregunta: **¿pueden un estado asociativo, la homeostasis, un presupuesto energético y varias escalas de memoria mejorar la manera en que aprende un modelo pequeño?**

No presentamos una metáfora como un descubrimiento. Cada órgano es código, cada hipótesis debe medirse y cada resultado debe sobrevivir a un holdout. La épica está en la pregunta y en el trabajo de responderla, no en exagerar las conclusiones.

> **La velocidad es ingeniería. El aprendizaje es arquitectura.**
>
> No basta con procesar más tokens por segundo. Importa cómo se transforma la experiencia en conocimiento, cómo se conserva y cómo se adapta el sistema.

El repositorio conserva el nombre técnico **WISE-RONIN**. El proyecto activo es **Lumen-Ronin**; un Transformer pequeño permanece como baseline de control.

---

## Índice

- [Estado actual](#estado-actual)
- [La tesis: el perro y el loro](#la-tesis-el-perro-y-el-loro)
- [Qué estamos construyendo](#qué-estamos-construyendo)
- [Arquitectura actual](#arquitectura-actual)
- [La campaña de entrenamiento: de cero a 400k](#la-campaña-de-entrenamiento-de-cero-a-400k)
- [Resultados emparejados](#resultados-emparejados)
- [Corpus y evaluación](#corpus-y-evaluación)
- [Benchmark de optimización CPU](#benchmark-de-optimización-cpu)
- [Integración progresiva de neuroplasticidad](#integración-progresiva-de-neuroplasticidad)
- [Cómo se investiga](#cómo-se-investiga)
- [Documentación y código](#documentación-y-código)
- [Límites y compromisos](#límites-y-compromisos)

## Estado actual

<sup>Actualizado: 23 de noviembre de 2026 · rama `main`</sup>

<details>
<summary><strong>La campaña activa</strong></summary>

**Último holdout emparejado promovido: LUMEN step 219.700 / Transformer step 219.750**, run [45257284512](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45257284512), sobre las mismas 2.431 preguntas congeladas.

- Lumen: **831/2.431 (34,18 %)**, NLL media **5,7441**.
- Transformer: **805/2.431 (33,11 %)**, NLL media **5,5147**.
- Lumen obtiene **26 aciertos más (+1,07 puntos porcentuales)**; Transformer conserva una NLL **0,2294 menor**.
- Ambos aciertan 456 preguntas; solo Lumen acierta 313; solo Transformer acierta 289; ambos fallan 1.198.
- Los checkpoints están separados por 50 actualizaciones: Lumen 219.700 y Transformer 219.750. La diferencia se declara expresamente; no se presenta como igualdad exacta de pasos.

El holdout y sus predicciones están preservados en el [artefacto 14859702424](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45257284512/artifacts/14859702424). La continuación previa desde 185.700 se interrumpió: Lumen conservó su último checkpoint en 219.700, mientras Transformer llegó a 235.700. Se seleccionaron los dos checkpoints más próximos con procedencia verificable para la comparación emparejada.

**Promoción:** el par 219.700/219.750 queda fijado como referencia experimental y padre histórico. La primera continuación emparejada sí completó el bloque hasta 229.700 en ambos modelos. La campaña corregida parte ahora de esos checkpoints finales verificados y tiene como objetivo común el paso 400.000. Se conserva el estado del optimizador y el RNG; la nueva transición de LR se reinicia explícitamente desde el estado real del optimizador.

**Tasa de aprendizaje de la campaña:** se ha fijado `5e-5` como LR objetivo para ambas arquitecturas. Es un cambio distinto para cada una: LUMEN venía de `3e-5` (objetivo un 66,7 % mayor), mientras que el Transformer venía de `7.5e-5` (objetivo un 33,3 % menor). No se aplica como una tasa constante abrupta: el scheduler cosenoidal parte del LR efectivo guardado en cada checkpoint y transiciona hacia `5e-5` durante la nueva campaña; se conservan los momentos del optimizador. Por ello, el efecto debe juzgarse por separado en cada arquitectura mediante loss por ventanas, estabilidad, throughput y holdout, no suponiendo de antemano que una vaya a beneficiarse más.

La entrada común sigue siendo el snapshot congelado de 250.000 filas (200k v2 + 50k personalidad factual), tokenizer v2 de 4.096 IDs, seed 42, batch 8 y contexto 128. El perfil runtime aprobado de Lumen se extrae del propio checkpoint, incluyendo plasticidad y fast paths Delta. Corpus v3 sigue excluido: su manifiesto mantiene `training_authorized=false`.
</details>

## La tesis: el perro y el loro

El loro representa la fluidez: producir secuencias deprisa y parecer competente por lo que repite. El perro representa otra apuesta: un sistema cuyo aprendizaje está regulado por señales internas, consecuencias, recursos y memoria.

En Lumen, esa metáfora tiene correspondencias técnicas concretas:

- **Glucosa:** reserva de energía interna que responde a mejoras verificadas de la loss.
- **Dolor:** señal acotada de regresión; reduce de forma acotada el multiplicador energético interno.
- **Placer:** señal independiente de mejora; eleva de forma moderada el multiplicador energético interno.
- **Homeostasis:** regula la actividad efectiva de las unidades.
- **Delta:** escribe en un estado asociativo cuando la sorpresa lo justifica.
- **Memoria:** separa adaptación elástica y registros lentos.

No son emociones humanas, no implican experiencia subjetiva y no convierten al modelo en un organismo. Son mecanismos computacionales inspirados en preguntas estudiadas por la neurociencia.

## Qué estamos construyendo

No buscamos demostrar que un modelo pequeño vaya a derrotar a los grandes por decreto. Investigamos si un sistema pequeño puede aprender de manera más estable y útil cuando su arquitectura incorpora algo más que capacidad bruta.

La hipótesis central es de ingeniería: quizá no todo el progreso dependa de multiplicar parámetros y operaciones. Quizá también importe cómo se asigna el presupuesto, cuándo se escribe en memoria, qué actividad se regula y cómo se conserva lo aprendido.

> **No esperaremos al hardware del futuro para estudiar principios del aprendizaje.** No necesitamos neuronas vivas, un chip neuromórfico ni un ordenador cuántico para experimentar con homeostasis, sorpresa, presupuestos y memoria. Podemos construir aproximaciones con CPU, código y modelos pequeños.

La biología tardó millones de años en descubrir ciertos trucos. Nosotros podemos estudiarlos con Python. Y cualquiera puede intentarlo.

## Arquitectura actual

La arquitectura activa se identifica en los checkpoints como `ronin_lumen_v3_2_parallel_delta`.

| Subsistema | Función |
|---|---|
| Embedding vivo | Vocabulario de 4.096 IDs, dimensión 256 y seguimiento de uso |
| Estado Delta ×4 | Memoria asociativa recurrente de rango 64, guiada por sorpresa |
| Homeostasis ×4 | Regulación de actividad efectiva y Top-K (16/32) |
| Presupuesto energético | Distribuye energía entre Delta, memoria y salida |
| Glucosa | Reserva energética acotada alimentada por mejoras de loss |
| Dolor y placer | Reguladores independientes de regresión y mejora |
| Memoria elástica/lenta | MLP rápido y banco de 128 slots |
| Cabeza de salida | Router entre una rama KAN spline y una MLP |
| Embedding/output | Pesos compartidos |

**Configuración de ejecución del bloque 61k→66k:** cuatro hilos intra-op, dos inter-op, MKLDNN activado, AdamW `foreach=false`, afinidad CPU fijada, Delta pre-normalization desactivada y dos refinamientos paralelos. El valor base de refinamientos en `CONFIG` es seis; el perfil de ejecución lo sobrescribe a dos.

La extensión dual-state ZE está disponible en el código, pero no se activa en el entrenamiento actual. El sueño/replay también está implementado, pero el workflow 61k→66k usa `--no-sleep`. No atribuimos a un checkpoint mecanismos que no ejecutó.

La descripción completa, con responsabilidades, fórmulas, parámetros y distinción entre código disponible y código activo, está en [Arquitectura actual](docs/ARQUITECTURA_ACTUAL_ES.md).

## La campaña de entrenamiento: de cero a 400k

El linaje emparejado se inició desde cero y avanzó mediante continuaciones verificadas. Cada bloque conserva checkpoints y estado de optimizador; los holdouts comparan ambos modelos sobre las mismas 2.431 preguntas.

| Tramo | Run de entrenamiento | Estado |
|---|---|---|
| 0 → 1k | [45012493289](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45012493289) | Completado; holdout [45013766761](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45013766761) |
| 1k → 11k | [45014470774](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45014470774) | Completado |
| 11k → 16k | [45021399600](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45021399600) | Completado; órganos afectivos integrados |
| 16k → 21k | [45025897082](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45025897082) | Completado |
| 21k → 26k | [45030262114](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45030262114) | Completado |
| 26k → 31k | [45034338841](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45034338841) | Completado |
| 31k → 41k | [45037602119](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45037602119) | Checkpoints guardados; gate final del workflow falló; holdout separado |
| 41k → 61k | [45044892658](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45044892658) | Completado y evaluado |
| 61k → 66k | [45061486739](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45061486739) | Completado; holdout 66k |
| 66k → 73,5k (corpus v2) | [45071952349](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45071952349) | Completado; holdout v2 de 2.431 preguntas |
| 73,5k → 133,5k (corpus v2) | [45082812702](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45082812702) | Transformer completó 133,5k; Lumen guardó 133,2k antes de la cancelación del job |
| 133,2k → 140,7k (v2 + personalidad) | [45136184396](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45136184396) | Ambos modelos completaron +7.500 pasos; holdout emparejado completado |
| 140,7k → 155,7k (v2 + personalidad) | [45149684913](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45149684913) | Ambos modelos completaron +15.000 pasos; holdout emparejado completado; step 155.700 promovido |
| 185,7k → 219,7k/235,7k (continuación interrumpida) | [45230206330](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45230206330) | Lumen conservó 219.700; Transformer completó 235.700; se verificaron los padres 219.700/219.750 para el holdout |
| Holdout 219,7k/219,75k | [45257284512](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45257284512) | 2.431 preguntas; Lumen 831, Transformer 805; par promovido |
| 219,7k/219,75k → 229,7k | [45262377661](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45262377661) | Ambos modelos completaron el primer bloque de 10.000 pasos y publicaron checkpoints 229.700 |
| Reanudación 229,7k → 239,7k | [45262377661](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45262377661) | El job de continuación falló al validar la reanudación: discrepancia entre el LR guardado en `training_config` y el solicitado; además se corrigió la ruta canónica del checkpoint entre artefactos (`checkpoint/` frente a `checkpoints/`). No fue un fallo de loss ni evidencia de inestabilidad de ninguna arquitectura |
| Recuperación 229,7k → 400k | [Workflow de continuación](https://github.com/orbitalronin94/WISE-RONIN/actions/workflows/continue-paired-219700-300000.yml) | Reanuda desde los dos checkpoints 229.700; LR objetivo `5e-5` en ambos; validación explícita de padre, paso, arquitectura, optimizador, RNG y scheduler antes de cada bloque |

El primer piloto de 500 pasos detectó un colapso de escrituras en las capas superiores. No se oculta ni se suma al linaje actual: fue una prueba de arranque que ayudó a corregir la ruta residual y la migración homeostática.

La crónica completa explica qué cambió en cada tramo, qué runs fallaron después de producir checkpoints válidos y qué ramas fueron únicamente benchmarks: [Crónica experimental, step 0 → actualidad](docs/CRONICA_EXPERIMENTAL_ES.md).

## Resultados emparejados

Último holdout completo: **Lumen 219.700 vs Transformer 219.750**, run [45257284512](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45257284512), con el holdout congelado de 2.431 preguntas.

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| **Lumen-Ronin** | **831/2.431** | **34,18 %** | 5,7441 |
| Transformer baseline | 805/2.431 | 33,11 % | **5,5147** |

Lumen supera al control por **26 respuestas (+1,07 puntos porcentuales)**. Transformer obtiene una NLL media inferior por 0,2294. Por tanto, la ventaja de Lumen está en la elección correcta; la distribución probabilística del Transformer sigue siendo mejor según esta NLL. No se deben colapsar ambas métricas en una única afirmación de victoria total.

La comparación es casi emparejada, con 50 actualizaciones de diferencia entre checkpoints. Ambos modelos utilizaron el mismo tokenizer, contexto y holdout, sin actualizaciones durante la evaluación. Conteos emparejados: ambos correctos 456; solo Lumen correcto 313; solo Transformer correcto 289; ambos incorrectos 1.198.

[Artefacto completo con predicciones por pregunta](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45257284512/artifacts/14859702424). La ficha de promoción y procedencia está en [experiments/step_219700_promotion.json](experiments/step_219700_promotion.json).
## Corpus y evaluación

La referencia de trabajo más reciente para la próxima campaña es [Corpus v3.2 — estado, manifiestos y auditoría](docs/CORPUS_V3_2_NEXT_TRAIN.md). Está enlazada para continuar la revisión, pero **no está autorizada aún para entrenamiento** (`training_authorized=false`); la campaña de 400k mantiene el snapshot v2 congelado.


Los tramos 66k→73,5k y 73,5k→133,2k utilizaron el corpus español v2 congelado, preparado con seed 42. Los tramos 133,2k→140,7k y 140,7k→155,7k utilizaron la misma base v2 más 50.000 ejemplos de personalidad factual adaptados a Lumen-Ronin; no incorporaron literatura, M-Personas ni los currículos sintéticos adicionales del candidato v3. El manifiesto registra 9.357.876 filas de entrenamiento, 1.413 de validación y 2.431 de holdout. La muestra determinista compartida de entrenamiento contiene 200.000 filas (20k v1, 72k CSIC y 108k TDX); tokenizer de 4.096 IDs, PAD=0 y EOS presente. Hash del tokenizer: `d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe`. CSIC y TDX se incorporaron bajo CC BY 4.0; CoWeSe quedó excluido por derechos de los textos de origen.

| Split | Filas |
|---|---:|
| Entrenamiento | 70.105 |
| Validación | 1.413 |
| Holdout | 2.431 |

El holdout contiene 1.931 preguntas SIQA-es y 500 OpenBookQA-es. Está separado del entrenamiento y no alimenta glucosa, dolor ni placer. La preparación aplica deduplicación, limpieza, filtros por componente y una política explícita de licencias.

La procedencia, fuentes, manifiesto, hashes y estado de la ampliación están en [data/README.md](data/README.md). La ampliación se aplicará por igual a Lumen y Transformer; el siguiente ciclo no incorporará v3 hasta que su gate de admisión se cierre explícitamente.

**Corpus v3 congelado y próximo paso:** v3.0.0 queda como la siguiente ampliación planificada para el próximo train emparejado. El experimento comprobará si seguir escalando y diversificando el corpus de forma gradual mejora la generalización, en lugar de asumir que más datos siempre ayudan. La auditoría automática está completada, pero no equivale a una certificación perfecta: se mantienen pendientes la revisión de near-duplicates y respuestas repetidas, el examen de los 41 marcadores potencialmente sensibles y la trazabilidad individual de parte de la literatura. El manifiesto conserva `training_authorized=false` como registro del estado de admisión de la auditoría; antes de ejecutar el train se deberá cerrar y documentar explícitamente el gate de datos. [Ficha y composición v3](docs/CORPUS_V3_ES.md) · [Auditoría 45142717502](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45142717502).

## Benchmark de optimización CPU

El 23 de noviembre de 2026 se ejecutó un cribado paralelo de **20 hipótesis de optimización** sobre Lumen-Ronin. Cada candidata y su control partieron del mismo checkpoint promovido (step 140.700), corpus v2 + personalidad, tokenizer, seed 42, scheduler y configuración de entrenamiento; se midieron 100 pasos por brazo, en cinco ventanas de 20 pasos. El experimento completo está en [GitHub Actions run 45158484545](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45158484545).

**Ganador del cribado: desactivar MKLDNN (mkldnn_enabled=false)**, manteniendo cuatro hilos intra-op, dos inter-op, AdamW foreach=false y afinidad CPU fijada.

| Métrica | Control | MKLDNN off | Cambio |
|---|---:|---:|---:|
| Tiempo total / 100 pasos | 31,213 s | 30,487 s | **−2,38 %** |
| Tiempo backward | 20,022 s | 19,592 s | **−2,19 %** |
| Training loss media | 3,2091804 | 3,2091800 | −0,00001 % |

El segundo resultado fue homeostasis_fused_scatter (+1,96 % de velocidad); el tercero, affinity_off (+1,47 %). Las variantes de dos y seis hilos empeoraron notablemente el tiempo en este runner. Las sustituciones Delta por bmm, la eliminación de diagnósticos y no_grad para el resumen no mejoraron el resultado de forma consistente. Por tanto, no se han promovido cambios en los mecanismos Delta ni se han desactivado órganos.

**Importante: es una selección provisional, no una victoria que haya superado el gate.** El protocolo había fijado un mínimo de +3 % de aceleración y una regresión máxima de loss del 0,5 %; ninguna de las 20 candidatas alcanzó el umbral de velocidad (0/20 elegibles). Se adopta MKLDNN off como mejor configuración observada para el siguiente ciclo, pero el margen del 2,38 % procede de una sola comparación de 100 pasos y puede incluir variación del runner. No es un holdout ni evidencia de mejor calidad o generalización. Debe confirmarse con repeticiones más largas antes de considerarlo una optimización robusta.

La configuración activa queda registrada en config/lumen_cpu_profile.json. El informe agregado y los informes individuales se conservaron como artefactos de la ejecución. El benchmark también incorporó una migración de compatibilidad para los checkpoints step-140.700 anteriores a las trazas persistentes de elegibilidad: únicamente inicializa a cero esos buffers nuevos ausentes y mantiene estricta la comprobación de cualquier otra clave inesperada o ausente.
## Integración progresiva de neuroplasticidad

La [PR #71](https://github.com/orbitalronin94/WISE-RONIN/pull/71) investiga una siguiente generación de mecanismos para Lumen-Ronin. Su objetivo transversal es **compartir señales y cálculos entre órganos**, evitando EMAs y operaciones redundantes, y coordinar mejor cuándo se escribe, consolida, debilita o protege lo aprendido.

La propuesta incluye un contexto compartido de sorpresa/incertidumbre; una señal común de evolución de la loss para glucosa, dolor y placer; perfiles acotados de modulación de plasticidad; etiquetado y captura sináptica; umbrales intrínsecos adaptativos; poda simulada y reversible; LTD protegida; memoria one-shot; escalado homeostático; replay priorizado/comprimido y downscaling durante sueño. El ELM permanece como probe offline, sin atribuirle efectos de entrenamiento. Las analogías farmacológicas son únicamente inspiración computacional, no equivalencias biológicas.

**La intención es que, tarde o temprano, podamos incorporar todos los cambios que demuestren ser útiles**, aunque algunos necesiten varias rondas de diseño, optimización y validación. No se activarán todos por decreto: cada candidato debe demostrar menor loss y mayor throughput en una comparación emparejada, sin empeorar estabilidad. Los que no pasen quedan pendientes con sus logs y podrán revisarse contra una referencia posterior. Después se evaluará la composición conjunta y su estabilidad en entrenamiento y holdout.

**Resultado del torneo de 18 candidatos (run 45175046624):** las 18 evaluaciones emparejadas finalizaron y solo 2 superaron todos los criterios: `07_disinhibition_gain` (escala de inhibición 0,80) y `09_eligibility_window` (retención de elegibilidad 0,98). Los otros 16 quedan pendientes, conservando artefactos y motivos; no se activan. La aprobación corresponde al paquete de base compartida más cada delta, no a una atribución causal aislada del delta.

El piloto integrado de 1.000 pasos se ejecutó desde el checkpoint original 155.700 hasta 156.700, usando únicamente esos dos mecanismos aprobados. Finalizó correctamente, guardó checkpoint y logs, y registró `dead=0,00 %` en el último paso. Es un piloto de integración, no una prueba de superioridad en holdout.

La PR #71 se fusionó mediante squash en `main` (commit `4e42507f146106937b4b2c6ae7f2d53cd6a8e70c`). El siguiente experimento continúa **Lumen y Transformer durante 15.000 pasos desde sus checkpoints originales 155.700**, con el mismo corpus v2 + personalidad, tokenizer, seed y configuración de batch/longitud. Solo Lumen recibe los dos mecanismos aprobados; Transformer permanece como control. El resultado decisivo será el holdout emparejado posterior.

La intención sigue siendo incorporar progresivamente todos los mecanismos que demuestren utilidad. Los pendientes no se descartan: podrán corregirse y volver a competir contra una referencia actualizada. El alcance, protocolo y registro de hipótesis están en [Integración progresiva de neuroplasticidad](docs/INTEGRACION_PROGRESSIVA_NEUROPLASTICIDAD_ES.md) y en la [PR #71](https://github.com/orbitalronin94/WISE-RONIN/pull/71).

## Cómo se investiga

El proyecto no acepta una mejora solo porque baje la loss o porque un workflow se llame `fast`.

1. **Mismo punto de partida:** comparar checkpoints, corpus, tokenizer, seed, batch y contexto.
2. **Medir por separado:** accuracy, NLL, loss, throughput, estabilidad y comportamiento generativo.
3. **Ablaciones explícitas:** cambiar un mecanismo cada vez y conservar una ruta de referencia.
4. **Preservar órganos:** no desactivar un órgano como atajo de rendimiento sin un experimento funcional separado.
5. **Verificar checkpoints:** pesos, buffers, RNG, optimizador, scheduler, hashes y paso real.
6. **Publicar los fallos:** distinguir un job fallido de un checkpoint inválido; no ocultar reintentos, cancelaciones ni duplicados.

Los microbenchmarks de 100 o 500 pasos sirven para filtrar hipótesis de entrenamiento y velocidad. No sustituyen al holdout ni prueban generalización.

## Documentación y código

### Lectura recomendada

- [Arquitectura actual](docs/ARQUITECTURA_ACTUAL_ES.md) — anatomía del modelo y configuración realmente activa.
- [Crónica experimental](docs/CRONICA_EXPERIMENTAL_ES.md) — linaje desde cero, cada continuación y sus holdouts.
- [Neurociencia y decisiones de arquitectura](docs/NEUROCIENCIA_Y_ARQUITECTURA_ES.md) — referencias, analogías y límites.
- [Cerebro a Código: hipótesis falsables](docs/CEREBRO_A_CODIGO_HIPOTESIS_ES.md) — seis líneas de investigación traducidas a experimentos, con referencias DOI verificadas, límites y criterios de rechazo.
- [Glucosa, dolor y placer](docs/PAIN_PLEASURE_ORGANS_ES.md) — evolución de los reguladores afectivos.
- [Experimento de neuroplasticidad](docs/EXPERIMENTO_NEUROPLASTICIDAD_ES.md) — rama aislada, chunking adaptativo, trazas de elegibilidad y protocolo Lumen vs Transformer desde cero.
- [Documento maestro V1](DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md) — especificación de diseño.
- [Especificación V1](SPEC_RONIN_LUMEN_V1.md) — contrato arquitectónico.
- [Índice de documentación](docs/README.md).

### Código principal

- [Modelo integrado](model/lumen.py)
- [Estado Delta](model/delta_state.py)
- [Homeostasis](model/homeostasis.py)
- [Memoria](model/memory.py)
- [Presupuesto y glucosa](model/energy.py)
- [Dolor](model/pain.py)
- [Placer](model/pleasure.py)
- [Router y cabezas KAN/MLP](model/heads.py)
- [Pérdidas](train/loss.py)
- [Entrenador y checkpoints](train/train.py)
- [Baseline Transformer](train/transformer_baseline.py)

## Límites y compromisos

- Lumen no es un organismo ni una prueba de conciencia.
- Las analogías biológicas no demuestran equivalencia con procesos neuronales.
- Un holdout de 2.431 preguntas no representa toda la inteligencia ni todos los dominios.
- La NLL y la accuracy deben leerse por separado.
- Un benchmark corto no demuestra una mejora sostenida.
- El código opcional no se considera activo si el workflow no lo habilita.
- Los resultados negativos, fallos y retrocesos forman parte de la investigación.

**La promesa de Lumen no es que ya haya resuelto el aprendizaje. Es que se atreve a formular otra arquitectura, exponerla a controles y dejar que los resultados decidan.**


## Optimización de throughput: ganador OpenMP (23-11-2026)

En el torneo emparejado de 18 perfiles CPU ([run 45197657760](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45197657760)), la mejor configuración fue **OpenMP ACTIVE + PROC_BIND=close + PLACES=cores + KMP_BLOCKTIME=0**, manteniendo cuatro hilos intra-op, dos inter-op, el perfil de órganos y todos los mecanismos LUMEN. En la pareja de 100 pasos obtuvo **3,4415 pasos/s frente a 3,2161** del control (**+7,01 %**); loss final idéntica (2,9238), actividad muerta idéntica (0 %) y diagnósticos de las cuatro capas conservados. Es una medición de throughput, no una afirmación de mejor aprendizaje ni de generalización.

Se integra únicamente este ganador; no se incorporan las otras 17 hipótesis del torneo. La configuración se aplica al entorno de ejecución de **ambos modelos**, LUMEN y Transformer, para que la comparación de calidad siga siendo emparejada. Ningún órgano, cálculo funcional o mecanismo se elimina, desactiva ni reduce. El siguiente entrenamiento continúa ambos checkpoints desde **170.700 hasta 185.700 (+15.000 pasos)** con el corpus, tokenizer, token cache, batch y scheduler congelados. Workflow: [LUMEN vs Transformer — OMP winner, 15k](.github/workflows/lumen-transformer-throughput-winner-15k.yml).
