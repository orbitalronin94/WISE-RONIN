# Corpus v3.2 — referencia para la próxima campaña

La rama de trabajo más reciente del corpus es [`corpus-v3-2-100mb-acceptance`](https://github.com/orbitalronin94/WISE-RONIN/tree/corpus-v3-2-100mb-acceptance), fijada en el commit `695243a9502c14bf8d106552756c75654baccfc9` en el momento de esta referencia.

Enlaces directos:
- [README y estado de la rama](https://github.com/orbitalronin94/WISE-RONIN/blob/corpus-v3-2-100mb-acceptance/README.md)
- [Manifiesto de fuentes v3.2](https://github.com/orbitalronin94/WISE-RONIN/blob/corpus-v3-2-100mb-acceptance/data/corpus_v3_2_sources.json)
- [Disposiciones de auditoría v3.2](https://github.com/orbitalronin94/WISE-RONIN/blob/corpus-v3-2-100mb-acceptance/data/corpus_v3_2_dispositions.json)
- [Workflow de admisión incremental de bloques de 100 MiB](https://github.com/orbitalronin94/WISE-RONIN/blob/corpus-v3-2-100mb-acceptance/.github/workflows/)

## Estado de autorización

**No usar todavía en entrenamiento.** El manifiesto de disposiciones declara expresamente `training_authorized: false`. Las fuentes permanecen sujetas a revisión de versión inmutable, derechos/licencias, particiones reales, idioma, calidad, privacidad y deduplicación frente a v3.1 y entre fuentes.

Esta referencia deja localizable la última línea de trabajo para preparar el siguiente train; no promueve la rama, no congela un artefacto entrenable ni altera retrospectivamente los experimentos que utilizaron el corpus v2. Antes de entrenar con v3.2 habrá que completar la auditoría, publicar hashes y recuentos efectivos, congelar el artefacto final y cambiar la autorización de forma explícita.
