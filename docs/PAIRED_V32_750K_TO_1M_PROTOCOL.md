# Continuación pareada v3.2.005: step 750.000 → 1.000.000

## Padres inmutables

Los dos modelos continúan desde los artefactos step-750.000 del run [45480785478](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45480785478). No se inicializan pesos nuevos ni se reinician los momentos de AdamW.

| Modelo | Arquitectura | Parámetros | Checkpoint |
|---|---|---:|---|
| LUMEN | ronin_lumen_v3_3_neuroplasticity | 2.281.841 | step_750000.pt |
| Transformer | decoder_transformer_4l_d192_h4_ff576_tied_v1 | 2.295.168 | step_750000.pt |

El workflow vuelve a comprobar step, arquitectura, parámetros, optimizer, scheduler, RNG, perfil de CPU y hashes antes de iniciar cualquier bloque.

## Datos y evaluación

- Corpus: Spanish v3.2.005, 279.613 filas.
- Corpus SHA-256: `0f7db9d6c4264f68fd6a12a9fe5bfa708e4fc30c1df176de5a4f53081bfe79aa`.
- Tokenizer SHA-256: `d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe`.
- Holdout sellado: 2.431 filas.
- Holdout SHA-256: `e21ae7e292d603f1b8073d8114572c92fb0c6448bfaed9066413ddce09e5a359`.
- El corpus y el tokenizer permanecen idénticos durante toda la continuación.

Resultado de referencia en step 750.000, evaluado con el mismo holdout:

| Modelo | Correctas | Accuracy | Mean answer NLL |
|---|---:|---:|---:|
| LUMEN | 837 / 2.431 | 34,43 % | 5,4239 |
| Transformer | 815 / 2.431 | 33,53 % | 5,4986 |

## Scheduler y ejecución

- Rango: 750.000 → 1.000.000.
- LR objetivo: `3e-5`; LR del checkpoint padre al cierre del segmento previo: `1.5e-5`.
- Nueva transición coseno: inicio 750.000, fin 1.000.000, warmup 100 pasos, mínimo 50 % del LR objetivo.
- AdamW, momentos, pesos, RNG y estado del modelo se restauran desde el padre.
- Batch 8, contexto 128, seed 42.
- CPU: 4 hilos intra-op; LUMEN usa 2 inter-op y su perfil fijado; Transformer mantiene el perfil de ejecución del baseline.
- 25 bloques secuenciales de 10.000 pasos por modelo. Cada bloque publica un checkpoint final; los checkpoints de recuperación se guardan cada 100 pasos en LUMEN.
- LUMEN y Transformer avanzan en cadenas independientes y pueden ejecutarse en paralelo.
- Al terminar ambos, el evaluador compara los dos step-1.000.000 contra el mismo holdout sellado.

## Validación de reanudación

El CI [45572460861](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45572460861) ejecutó ambos entrenadores reales con los artefactos originales y un bloque temporal de un paso: step 750.000 → 750.001. Ambos conservaron optimizer, scheduler, RNG, corpus y tokenizer, y escribieron checkpoints válidos con la nueva transición 750.000 → 1.000.000.

Ese smoke test no publica checkpoints de entrenamiento ni modifica los artefactos originales.

## Relación con Corpus V4

Esta continuación no cambia de corpus: termina la línea v3.2.005 que ya está en curso. Corpus V4 se congela y autoriza por separado; sus datos no se mezclan en estos checkpoints ni se cambia el tokenizer bajo el modelo.
