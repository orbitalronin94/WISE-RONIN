# Índice de documentación — Lumen-Ronin

La documentación distingue entre arquitectura implementada, configuración realmente ejecutada, especificación normativa e historia experimental. La portada del repositorio resume el estado actual; los documentos siguientes contienen el detalle.

## Empieza aquí

1. [README del proyecto](../README.md) — tesis, estado actual, resultados y accesos rápidos.
2. [Arquitectura actual](ARQUITECTURA_ACTUAL_ES.md) — componentes, parámetros y qué está activo o desactivado en el entrenamiento 185,7k (último holdout y train activo).
3. [Crónica experimental](CRONICA_EXPERIMENTAL_ES.md) — historia desde el arranque a step 1.000, continuaciones hasta 61k y bloque actual a 66k.
4. [Neurociencia y decisiones de arquitectura](NEUROCIENCIA_Y_ARQUITECTURA_ES.md) — literatura, traducción a mecanismos y límites de las analogías.

## Diseño y especificaciones

- [Documento maestro V1](../DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md) — intención y especificación normativa del refactor.
- [Especificación RONIN-LUMEN V1](../SPEC_RONIN_LUMEN_V1.md)
- [Extensión experimental V3.2 ZE / LC-NE](RONIN_LUMEN_V3_2_ZE.md)
- [Auditoría de la especificación](SPEC_AUDIT.md) — contradicciones, decisiones pendientes y estado de implementación.
- [Auditoría del bucle de entrenamiento](TRAIN_LOOP_REFERENCE_AUDIT.md) — qué ejecuta el trainer y qué exige validación.
- [Glucosa, dolor y placer](PAIN_PLEASURE_ORGANS_ES.md) — diseño e integración histórica de los reguladores afectivos.

## Datos y evaluación

- [Corpus español v1: procedencia, splits y hashes](../data/README.md)
- [Protocolo del piloto comparativo](PILOTO_COMPARATIVO_ES.md)
- [Holdout congelado](../data/holdout_500.jsonl) — referencia histórica; la evaluación emparejada actual utiliza el holdout español de 2.431 filas preparado como artefacto.
- [Preparación de datos](../.github/workflows/prepare-data.yml)
- [Continuación emparejada 61k→66k](../.github/workflows/continue-61000-66000-paired-organs-delta2.yml)

## Código por responsabilidad

- [Modelo integrado](../model/lumen.py)
- [Configuración base](../model/config.py)
- [Estado Delta](../model/delta_state.py)
- [Embedding](../model/embedding.py)
- [Homeostasis y Top-K](../model/homeostasis.py)
- [Memoria](../model/memory.py)
- [Presupuesto y glucosa](../model/energy.py)
- [Dolor](../model/pain.py)
- [Placer](../model/pleasure.py)
- [Router y cabezas KAN/MLP](../model/heads.py)
- [Extensión dual-state ZE](../model/ze.py)
- [Sueño / replay](../train/sleep.py)
- [Pérdidas](../train/loss.py)
- [Entrenador y checkpoints](../train/train.py)
- [Baseline Transformer](../train/transformer_baseline.py)

## Regla de lectura

Una implementación que compila no equivale a una hipótesis validada. Un smoke test no equivale a entrenamiento. Un microbenchmark no equivale a holdout. Un checkpoint guardado tras un workflow fallido debe verificarse y evaluarse por separado. Los mecanismos opcionales solo se atribuyen a un entrenamiento si su workflow los habilitó.

- [Torneo de 20 hipótesis sobre mecanismos pendientes](TORNEO_20_MECANISMOS_PENDIENTES_ES.md) — alcance, combinaciones y gate loss + throughput.
