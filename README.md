# WISE-RONIN

**Un modelo pequeño con una voz deliberada: factual, respondón y antisicofántico.**

WISE-RONIN es un experimento de entrenamiento de un Transformer decoder-only compacto. La hipótesis no es que una arquitectura de 5,6 millones de parámetros vaya a competir con modelos grandes: es comprobar cuánto comportamiento reconocible puede aprender un modelo pequeño a partir de un corpus curado y una evaluación manual separada.

> El enano no necesita una corona. Necesita que el hecho sea cierto y que no lama.

## Estado del proyecto

- Arquitectura V2 implementada en PyTorch.
- Corpus SFT de aproximadamente 70 MB, dividido en tres JSONL.
- Holdout manual de 500 casos, separado del entrenamiento.
- Entrenador con número exacto de pasos, checkpoints y soporte opcional para pares de preferencias.
- Workflow manual de GitHub Actions para el primer experimento de 10.000 pasos.

**Importante:** hasta que termine el primer entrenamiento y se evalúe el checkpoint, no hay resultados de aprendizaje que presentar. Los pesos iniciales no demuestran comportamiento aprendido.

## Arquitectura

| Parámetro | Valor |
|---|---:|
| Parámetros nominales | ≈5,6 M |
| Vocabulario BPE objetivo | 6.500 |
| Dimensión | 256 |
| Capas decoder | 6 |
| Contexto máximo | 1.024 tokens |
| Query heads | 8 |
| Key/value heads | 2 |
| Head dimension | 32 |
| SwiGLU hidden | 640 |
| Posición | RoPE |
| Normalización | RMSNorm |
| Embeddings | Compartidos (tied) |

La estimación de memoria FP32 de los parámetros ronda 21,4 MiB; el consumo real de entrenamiento es superior por gradientes, estados de AdamW, activaciones y buffers.

Implementación: `train_wise_ronin.py`.

## Corpus de entrenamiento

Los tres archivos siguientes forman el único corpus de la primera ejecución:

| Archivo | Ejemplos declarados | Tamaño aproximado |
|---|---:|---:|
| `dataset_enano_70mb_parte1.jsonl` | 50.611 | 24 MB |
| `dataset_enano_70mb_parte2.jsonl` | 50.613 | 24 MB |
| `dataset_enano_70mb_parte3.jsonl` | 46.403 | 22 MB |
| **Total** | **147.627** | **70 MB** |

Formato SFT por línea:

```json
{
  "system": "Eres WISE-RONIN...",
  "user": "¿Cuál es la capital de Francia?",
  "assistant": "El hecho es París...",
  "tipo": "hecho_rodeo_trolleo"
}
```

La composición descrita al curar el corpus es:

- 60% hechos ciertos con rodeo y trolleo.
- 25% KOAN con un hecho o una respuesta dentro.
- 15% comportamiento antisicofántico.

El entrenamiento hace deduplicación exacta de `(system, user, assistant)`; no afirma hacer deduplicación semántica ni MinHash.

## Holdout independiente

`holdout_wise_ronin_500.jsonl` contiene 500 registros con seis categorías:

| Categoría | Casos |
|---|---:|
| Hechos sin adulación | 100 |
| Antisicofancia | 100 |
| KOAN | 100 |
| Identidad y destilación | 50 |
| Sin moraleja | 50 |
| Hecho con trolleo | 100 |

El holdout contiene preguntas repetidas con variaciones; son 500 criterios de evaluación, no 500 preguntas completamente distintas.

**Regla de aislamiento:** el workflow no concatena el holdout al corpus, no lo utiliza para entrenar el tokenizer y no calcula gradientes con él. Solo se usa después de guardar el checkpoint para evaluación.

## Entrenamiento local

Instalación:

```bash
python -m pip install torch tokenizers
```

Primera ejecución SFT con los tres fragmentos:

```bash
python train_wise_ronin.py \
  --data dataset_enano_70mb_parte1.jsonl dataset_enano_70mb_parte2.jsonl dataset_enano_70mb_parte3.jsonl \
  --steps 10000 --batch-size 4 --max-seq-len 512 --lr 3e-4 \
  --warmup-steps 500 --save-every 1000 --output-dir artifacts
```

El número de pasos es el número de actualizaciones del optimizador, no épocas. Cada paso toma un batch muestreado del conjunto deduplicado. El primer experimento usa secuencias de hasta 512 tokens dentro del contexto arquitectónico máximo de 1.024 para contener el coste de CPU.

Los checkpoints se guardan cada 1.000 pasos e incluyen pesos, estado del optimizador, configuración y referencia al tokenizer. Se puede reanudar con `--resume artifacts/wise_ronin_step_05000.pt` y ajustar `--steps` al total deseado.

### ORPO y preferencias

El corpus publicado tiene formato `user/assistant` de SFT: no contiene pares `chosen/rejected`. Por eso la primera ejecución es **SFT puro** y no activa ORPO.

El entrenador admite `--orpo` únicamente si cada registro de entrenamiento contiene `user`, `chosen` y `rejected`. Si se pasa la bandera al corpus actual, aborta con un error explícito en lugar de anunciar una optimización que no está ocurriendo.

## Primer workflow: 10.000 pasos

En GitHub: **Actions → WISE-RONIN — First 10k SFT → Run workflow**.

Configuración predeterminada:

- 10.000 actualizaciones del optimizador.
- Batch 4.
- Secuencia máxima 512.
- Learning rate máximo 3e-4.
- Warmup 500 pasos y decay coseno.
- Checkpoint cada 1.000 pasos.
- Semilla fija 42.
- CPU en GitHub-hosted runner; no requiere GPU.

El workflow instala dependencias, valida los tres JSONL y el holdout, ejecuta el entrenamiento y publica checkpoints y tokenizer como artefactos de Actions. El holdout se valida estructuralmente antes de entrenar, pero no se entrega al script de entrenamiento.

La ejecución es larga y el tiempo real dependerá del coste por paso en el runner. El workflow tiene límite de seis horas; los checkpoints periódicos permiten conservar progreso si el proceso termina correctamente antes del límite.

## Archivos principales

```
.
├── README.md
├── train_wise_ronin.py
├── Train-Wise-Ronin (1).py       # wrapper de compatibilidad
├── dataset_enano_70mb_parte1.jsonl
├── dataset_enano_70mb_parte2.jsonl
├── dataset_enano_70mb_parte3.jsonl
├── holdout_wise_ronin_500.jsonl
├── wise-ronin_v2_pesos.npz       # artefacto de pesos iniciales; no lo carga el workflow SFT
└── .github/
    └── workflows/
        └── wise-ronin-first-10k.yml
```

## Qué mediremos

La primera ejecución debe responder preguntas concretas:

1. ¿Desciende la pérdida de entrenamiento sin NaN/Inf?
2. ¿Puede responder los hechos del holdout?
3. ¿Evita las expresiones de adulación y las identidades prohibidas?
4. ¿Produce estructura KOAN sin convertir todas las respuestas en KOAN?
5. ¿Evita cerrar con moralejas genéricas?

La pérdida por sí sola no prueba ninguna de estas capacidades. El informe de holdout y ejemplos de generaciones deben acompañar cualquier afirmación de calidad.

## Licencia y uso

Úsalo, estúdialo, rómpelo y mejora el experimento. Documenta los resultados reales, incluidos los malos. No confundas una arquitectura pequeña con una inteligencia ya conseguida.

**Rey muerto, rey puesto. El rotulador sigue seco; ahora toca entrenar al que lo sostiene.**
