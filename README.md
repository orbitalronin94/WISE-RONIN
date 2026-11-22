# Lumen-Ronin
### Una arquitectura pequeña. Una pregunta enorme: ¿y si aprender no fuera solo escalar?

> **LUMEN es un perro.** Un perro de Pavlov que aprende lento, pero aprende. Que saliva, que tiene hambre, que le duele, que le da placer. Y un perro, aunque sea más lento que un loro para repetir, es más listo que un loro para aprender.

Lumen-Ronin es un experimento de arquitectura recurrente, CPU-first y de pocos millones de parámetros. En lugar de limitarse a apilar bloques Transformer y aumentar cómputo, explora otra pregunta: **¿pueden un estado asociativo, la homeostasis, un presupuesto energético y varias escalas de memoria mejorar la manera en que aprende un modelo pequeño?**

No presentamos una metáfora como un descubrimiento. Cada órgano es código, cada hipótesis debe medirse y cada resultado debe sobrevivir a un holdout. La épica está en la pregunta y en el trabajo de responderla, no en exagerar las conclusiones.

El repositorio conserva el nombre técnico **WISE-RONIN**. El proyecto activo es **Lumen-Ronin**; un Transformer pequeño permanece como baseline de control.

---

## Índice

- [Estado actual](#estado-actual)
- [La tesis: el perro y el loro](#la-tesis-el-perro-y-el-loro)
- [Qué estamos construyendo](#qué-estamos-construyendo)
- [Arquitectura actual](#arquitectura-actual)
- [La campaña de entrenamiento: de cero a 66k](#la-campaña-de-entrenamiento-de-cero-a-66k)
- [Resultados emparejados](#resultados-emparejados)
- [Corpus y evaluación](#corpus-y-evaluación)
- [Cómo se investiga](#cómo-se-investiga)
- [Documentación y código](#documentación-y-código)
- [Límites y compromisos](#límites-y-compromisos)

## Estado actual

<sup>Actualizado: 22 de noviembre de 2026 · rama `main`</sup>

<details>
<summary><strong>La campaña activa</strong></summary>

- Último checkpoint emparejado completado y evaluado: **step 61.000**.
- Continuación activa: **61.000 → 66.000**, run [45061486739](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45061486739).
- Quality gate: correcto.
- Transformer: entrenamiento hasta 66.000 completado.
- Lumen: entrenamiento todavía en curso al corte de esta documentación.
- Holdout final de 66.000: pendiente; no se atribuyen resultados antes de que termine.

El bloque actual mantiene todos los órganos activos. Lumen usa el perfil seleccionado de dolor ×1,50 y placer ×1,50 sobre los ajustes previos, con Delta en dos refinamientos paralelos. El Transformer conserva su configuración baseline. Ambos parten de sus checkpoints 61k del run [45044892658](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45044892658), el mismo corpus/tokenizer, batch 8, contexto 128 y seed 42.

Existe otro run 41k→61k en paralelo ([45054687034](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45054687034)); no es el padre utilizado por la continuación actual. La procedencia de los checkpoints se mantiene explícita para no mezclar ejecuciones.

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

## La campaña de entrenamiento: de cero a 66k

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
| 61k → 66k | [45061486739](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45061486739) | En curso; holdout pendiente |

El primer piloto de 500 pasos detectó un colapso de escrituras en las capas superiores. No se oculta ni se suma al linaje actual: fue una prueba de arranque que ayudó a corregir la ruta residual y la migración homeostática.

La crónica completa explica qué cambió en cada tramo, qué runs fallaron después de producir checkpoints válidos y qué ramas fueron únicamente benchmarks: [Crónica experimental, step 0 → actualidad](docs/CRONICA_EXPERIMENTAL_ES.md).

## Resultados emparejados

Último holdout completo disponible: **step 61.000**, run [45044892658](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45044892658).

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen-Ronin | 795/2.431 | 32,70 % | **6,1771** |
| Transformer baseline | **824/2.431** | **33,90 %** | 6,4988 |

La lectura honesta es doble:

- El Transformer aventaja a Lumen por 29 respuestas en accuracy.
- Lumen obtiene una NLL media 0,3217 menor.

No hay una victoria total. Lumen conserva una ventaja en NLL; el baseline mantiene una pequeña ventaja en aciertos. La diferencia de accuracy no debe proclamarse concluyente sin el análisis emparejado correspondiente.

La serie histórica completa está en la [crónica experimental](docs/CRONICA_EXPERIMENTAL_ES.md). Las generaciones libres se documentan aparte porque no son comparables directamente con el test de opción múltiple.

## Corpus y evaluación

El entrenamiento emparejado actual utiliza un corpus español congelado, preparado con seed 42:

| Split | Filas |
|---|---:|
| Entrenamiento | 70.105 |
| Validación | 1.413 |
| Holdout | 2.431 |

El holdout contiene 1.931 preguntas SIQA-es y 500 OpenBookQA-es. Está separado del entrenamiento y no alimenta glucosa, dolor ni placer. La preparación aplica deduplicación, limpieza, filtros por componente y una política explícita de licencias.

La procedencia, fuentes, manifiesto y hashes están en [data/README.md](data/README.md).

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
- [Glucosa, dolor y placer](docs/PAIN_PLEASURE_ORGANS_ES.md) — evolución de los reguladores afectivos.
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
