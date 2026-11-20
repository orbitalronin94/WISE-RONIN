# WISE-RONIN

**Un modelo pequeño con una voz deliberada: factual, respondón y antisicofántico.**

WISE-RONIN es un experimento reproducible de entrenamiento de un Transformer decoder-only compacto. Investiga cuánto comportamiento reconocible puede aprender un modelo pequeño con corpus curados, currículos sintéticos controlados y evaluación separada. No se presenta como un modelo generalista ni como una inteligencia ya conseguida.

> El enano no necesita una corona. Necesita que el hecho sea cierto y que no lama.

## Índice

- [Estado actual](#estado-actual)
- [Resultados y límites](#resultados-y-límites)
- [Arquitectura](#arquitectura)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Datos y procedencia](#datos-y-procedencia)
- [Entrenar](#entrenar)
- [Evaluar](#evaluar)
- [GitHub Actions](#github-actions)
- [Reproducibilidad y decisiones](#reproducibilidad-y-decisiones)
- [Documentación](#documentación)
- [Licencia y uso](#licencia-y-uso)

## Estado actual

- Trainer Transformer en PyTorch, con checkpoint/resume y SFT.
- Primer checkpoint SFT de 10.000 pasos completado en CPU mediante GitHub Actions.
- Corpus sintético ampliado de 147.627 filas, con informe de composición y trazabilidad por `fact_id`.
- Constructor de un currículo equilibrado de 80.000 filas, con reserva de filas no seleccionadas.
- Segundo entrenamiento: continuación desde el checkpoint 10k, dos épocas sobre el currículo equilibrado. **Actualmente en ejecución** en [GitHub Actions run 44721903295](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44721903295).
- Evaluadores greedy y sampling. Sus métricas automáticas son señales de diagnóstico, no sustitutos de revisión semántica.

## Resultados y límites

### Primer SFT: 10.000 pasos

[Run 44709299157](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44709299157) terminó correctamente.

| Métrica | Resultado |
|---|---:|
| Actualizaciones | 10.000 |
| Ejemplos tras deduplicación exacta | 88.348 |
| Parámetros efectivos | 4.267.520 |
| Vocabulario BPE efectivo | 1.297 |
| Loss final registrada | 0,16234 |
| Tiempo de entrenamiento | 1.562 s (aprox. 26 min) |

La loss descendente confirma que el objetivo SFT se optimizó; **no demuestra por sí sola** factualidad, diversidad ni seguimiento de instrucciones.

### Primer holdout greedy

El evaluador heurístico marcó 35/500 casos (7%). El conjunto contiene 500 registros, pero solo 55 preguntas distintas, repetidas con variaciones. Además, el modelo mostró colapso de plantillas y errores factuales evidentes. Por tanto, ese 7% no debe leerse como una tasa de éxito semántico ni como 500 pruebas independientes.

El siguiente ciclo añade sampling y métricas separadas para coincidencia con el ancla factual, diversidad literal y concentración de motivos. La coincidencia literal también es solo una aproximación: requiere inspección humana.

### Lo que todavía no afirmamos

- No afirmamos que las 147.627 filas sintéticas sean 147.627 hechos independientes.
- El generador base contiene 402 anclas de conocimiento factual; sus variantes no crean nuevos hechos.
- No afirmamos que el modelo sea fiable para uso general.
- No activamos ORPO sin pares reales `chosen/rejected`; inventar preferencias falsearía el experimento.

## Arquitectura

| Parámetro | Configuración |
|---|---:|
| Parámetros efectivos del checkpoint 10k | 4.267.520 |
| Dimensión | 256 |
| Capas decoder | 6 |
| Contexto arquitectónico | 1.024 tokens |
| Query heads / KV heads | 8 / 2 |
| Head dimension | 32 |
| SwiGLU hidden | 640 |
| Posición | RoPE |
| Normalización | RMSNorm |
| Embeddings | Compartidos (tied) |
| Tokenizer del primer checkpoint | ByteLevel BPE, 1.297 tokens efectivos |

La cifra de parámetros y el vocabulario de la tabla son los medidos en el primer checkpoint, no los valores nominales de los primeros borradores. La memoria real de entrenamiento supera el tamaño de los pesos por gradientes, estados del optimizador, activaciones y buffers.

Implementación: [`train_wise_ronin.py`](train_wise_ronin.py).

## Estructura del repositorio

El árbol conserva en la raíz los ficheros de datos heredados porque los workflows actuales los referencian directamente. No se han movido ni renombrado para evitar romper la reproducibilidad.

```text
WISE-RONIN/
├── README.md
├── train_wise_ronin.py                 # modelo, SFT, checkpoints y resume
├── evaluate_wise_ronin.py              # evaluación greedy
├── evaluate_wise_ronin_sampling.py     # sampling, diversidad y motivos
├── generate_wise_ronin_expanded.py     # generador determinista de corpus
├── build_balanced_80k.py               # currículo equilibrado + reserva
├── audit_wise_ronin_dataset.py         # auditoría del corpus heredado
├── build_wise_ronin_dataset.py         # constructor de dataset original
├── Train-Wise-Ronin (1).py             # wrapper de compatibilidad
├── dataset_enano_70mb_parte*.jsonl     # corpus SFT heredado, tres partes
├── holdout_wise_ronin_500.jsonl        # evaluación separada
├── wise-ronin_v2_pesos.npz             # pesos históricos; no usados por el SFT actual
├── data/
│   └── expanded/
│       ├── README.md
│       ├── wise_ronin_expanded.jsonl
│       └── wise_ronin_expanded_report.json
├── docs/
│   ├── REPOSITORY_GUIDE.md
│   └── EXPERIMENTS.md
└── .github/
    └── workflows/
        ├── wise-ronin-first-10k.yml
        ├── wise-ronin-expand-dataset.yml
        ├── wise-ronin-dataset-audit.yml
        ├── wise-ronin-balanced-80k.yml
        └── wise-ronin-balanced-80k-validation.yml
```

Los datos equilibrados y los checkpoints del segundo ciclo se generan como artefactos de Actions; no se confunden con los ficheros versionados en `data/expanded/`.

## Datos y procedencia

### Corpus heredado

Tres ficheros JSONL, con 147.627 filas declaradas en origen. La primera ejecución eliminó duplicados exactos y entrenó con 88.348 ejemplos. La deduplicación exacta no equivale a deduplicación semántica.

### Corpus ampliado

[`data/expanded/wise_ronin_expanded.jsonl`](data/expanded/wise_ronin_expanded.jsonl) contiene 147.627 registros generados determinísticamente desde un banco integrado de hechos y operaciones aritméticas exactas.

- 402 anclas factuales.
- 9.939 tareas aritméticas base.
- 10.341 grupos independientes por `fact_id`.
- 147.627 prompts y respuestas literales únicas.
- Variantes de formulación y estilo; no nuevas unidades de conocimiento.

Informe: [`wise_ronin_expanded_report.json`](data/expanded/wise_ronin_expanded_report.json). Detalles y advertencias: [`data/expanded/README.md`](data/expanded/README.md).

### Currículo equilibrado 80k

El constructor selecciona exactamente 80.000 filas con estos objetivos:

| Categoría | Filas |
|---|---:|
| Geografía factual | 10.000 |
| Química factual | 15.000 |
| Conocimiento general | 15.000 |
| Sumas | 10.000 |
| Restas | 10.000 |
| Multiplicaciones | 10.000 |
| Divisiones | 10.000 |
| **Total** | **80.000** |

Las variantes añadidas de geografía y conocimiento general conservan el `fact_id` original. La ampliación aritmética de divisiones crea operaciones exactas nuevas. Las filas fuente no seleccionadas se conservan en una reserva separada; el builder no sobrescribe el corpus de origen.

## Entrenar

### Dependencias

```bash
python -m pip install torch tokenizers
```

### Primer SFT desde cero

```bash
python train_wise_ronin.py \
  --data dataset_enano_70mb_parte1.jsonl dataset_enano_70mb_parte2.jsonl dataset_enano_70mb_parte3.jsonl \
  --steps 10000 --batch-size 4 --max-seq-len 512 --lr 3e-4 \
  --warmup-steps 500 --save-every 1000 --output-dir artifacts
```

### Continuación sobre el currículo equilibrado

El workflow automatizado descarga el checkpoint de 10k, construye y valida el dataset equilibrado y ejecuta dos épocas con batch 8 y learning rate 1.5e-4. Para 80.000 filas y batch 8 son 10.000 actualizaciones por época: 20.000 adicionales, hasta el paso global 30.000.

El workflow preserva las filas no seleccionadas y reutiliza el tokenizer del checkpoint. No activa ORPO porque el corpus no contiene preferencias etiquetadas.

## Evaluar

Evaluación greedy:

```bash
python evaluate_wise_ronin.py --help
```

Evaluación sampling:

```bash
python evaluate_wise_ronin_sampling.py --help
```

El evaluador sampling registra generaciones, coincidencia literal con el ancla esperada, respuestas únicas y concentración de motivos. Hay que leer cada métrica con sus límites:

1. **Factualidad:** la coincidencia literal puede fallar ante paráfrasis y aceptar coincidencias superficiales.
2. **Diversidad:** contar respuestas únicas no mide calidad ni pertinencia.
3. **Motivos:** detectar aperturas, coletillas u objetos repetidos es diagnóstico de colapso estilístico.
4. **Holdout:** 500 filas no significan 500 preguntas independientes; el conjunto actual contiene preguntas repetidas.

No ajustar el modelo mirando las respuestas del holdout ni incorporarlo al entrenamiento.

## GitHub Actions

Los workflows se ejecutan desde la pestaña **Actions**. Consulta [` .github/workflows`](.github/workflows) y la [guía de workflows](.github/workflows/README.md) para el propósito y los disparadores de cada uno.

- **First 10k SFT:** entrenamiento inicial desde cero.
- **Expand dataset:** genera el corpus ampliado determinista.
- **Dataset audit:** inspecciona duplicados y distribución del corpus heredado.
- **Balanced 80k:** reanuda el checkpoint 10k y entrena dos épocas.
- **Balanced 80k validation:** valida el builder y los invariantes en PR.

Los checkpoints, informes de evaluación y datasets de trabajo se publican como artefactos de Actions cuando el workflow correspondiente termina correctamente.

## Reproducibilidad y decisiones

- Semillas explícitas en los workflows.
- El holdout se mantiene fuera del entrenamiento y del tokenizer.
- Los constructores producen informes con recuentos y categorías.
- Los datos de origen y los sobrantes se conservan.
- Los checkpoints incluyen configuración y estado necesario para reanudar.
- No se fabrican pares de preferencias para habilitar ORPO.
- Los cambios de estructura se harán de forma gradual para no romper rutas consumidas por workflows.

## Documentación

- [Guía del repositorio y responsabilidades de cada archivo](docs/REPOSITORY_GUIDE.md)
- [Registro de experimentos y resultados](docs/EXPERIMENTS.md)
- [Ficha del corpus ampliado](data/expanded/README.md)
- [Informe JSON del corpus ampliado](data/expanded/wise_ronin_expanded_report.json)
- [Workflow de entrenamiento equilibrado](.github/workflows/wise-ronin-balanced-80k.yml)

## Licencia y uso

Úsalo, estúdialo, rómpelo y mejora el experimento. Documenta los resultados reales, incluidos los malos. No confundas una arquitectura pequeña con una inteligencia ya conseguida.

**Rey muerto, rey puesto. El rotulador sigue seco; ahora toca entrenar al que lo sostiene.**
