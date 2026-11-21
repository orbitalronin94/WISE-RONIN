# Documentación de RONIN-LUMEN

Índice principal en castellano. Los documentos normativos se mantienen en su idioma de origen cuando corresponde; los resúmenes de arquitectura, operación y resultados se ofrecen en español. Las afirmaciones experimentales deben leerse junto con su estado y sus límites.

## Empieza aquí

1. [README del proyecto](../README.md) — propósito, arquitectura, estado y reproducción.
2. [Documento maestro V1](../DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md) — especificación normativa completa.
3. [Neurociencia y decisiones de arquitectura](NEUROCIENCIA_Y_ARQUITECTURA_ES.md) — bibliografía, traducción a mecanismos y límites de las analogías.

## Arquitectura y diseño

- [Especificación RONIN-LUMEN V1](../SPEC_RONIN_LUMEN_V1.md)
- [Extensión experimental V3.2 ZE / LC-NE](RONIN_LUMEN_V3_2_ZE.md)
- [Auditoría de la especificación](SPEC_AUDIT.md) — contradicciones, decisiones pendientes y estado de implementación.
- [Auditoría del bucle de entrenamiento](TRAIN_LOOP_REFERENCE_AUDIT.md) — qué está implementado, qué está incompleto y qué exige validación.
- [Protocolo del piloto comparativo](PILOTO_COMPARATIVO_ES.md) — comparación de LUMEN con un Transformer de parámetros equiparados y evaluación GSM8K.

## Datos, licencias y evaluación

- [Preparación y procedencia de datos](../data/README.md)
- [Datos propios: formato y requisitos](../data/owner/README_ES.md)
- [Holdout congelado](../data/holdout_500.jsonl)
- [Preparación de bibliotecas en GitHub Actions](../.github/workflows/prepare-data.yml)
- [Entrenamiento CPU en GitHub Actions](../.github/workflows/train.yml)
- [Piloto comparativo de 500 pasos](../.github/workflows/pilot.yml)

## Código por responsabilidad

- [Modelo integrado](../model/lumen.py)
- [Estado Delta](../model/delta_state.py)
- [Homeostasis y Top-K](../model/homeostasis.py)
- [Memoria](../model/memory.py)
- [Presupuesto de energía](../model/energy.py)
- [Router y cabezas MLP/KAN](../model/heads.py)
- [Sueño / replay](../train/sleep.py)
- [Pérdidas auxiliares](../train/loss.py)
- [Entrenador y checkpoints](../train/train.py)
- [Baseline Transformer de 2,3 M de parámetros](../train/transformer_baseline.py)
- [Constructor del corpus piloto](../scripts/build_pilot_corpus.py)
- [Tokenizer del piloto](../scripts/train_pilot_tokenizer.py)
- [Preparación del holdout GSM8K](../scripts/prepare_reasoning_holdout.py)
- [Evaluador común](../scripts/evaluate_pilot.py)
- [Entrenamiento del tokenizer](../scripts/train_tokenizer.py)

## Estado de validación

- Integración de arquitectura: [PR #13](https://github.com/orbitalronin94/WISE-RONIN/pull/13), fusionada.
- Gates de corpus/tokenizer: [PR #14](https://github.com/orbitalronin94/WISE-RONIN/pull/14), abierta.
- Staging externo y auditoría léxica: [Actions run 44890014413](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44890014413).

**Regla de lectura:** una implementación que compila no equivale a una hipótesis validada; un smoke test dummy no equivale a entrenamiento; una auditoría léxica no demuestra ausencia de fuga semántica; y una etiqueta de licencia de dataset no reemplaza la revisión de derechos de sus contenidos.
