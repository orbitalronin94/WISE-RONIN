# Corpus v4 — auditoría de diversidad y admisión

## Estado

Auditoría inicial realizada sobre el artefacto acumulativo de la cadena v3.2 descargado desde GitHub Actions. Se inspeccionaron íntegramente los cinco deltas v3.2.001–v3.2.005 (JSONL), sus manifiestos y el índice de deduplicación incluido en el artefacto.

**Límite de alcance:** el ZIP de la cadena contiene los deltas v3.2, no el JSONL completo del padre v3.1. Por tanto, las cifras por fuente que siguen describen los 81.825 registros añadidos en v3.2; no se presentan como una auditoría léxica completa de las 279.613 filas acumuladas.

## Hallazgos sobre v3.2

| Incremento | Fuente | Filas | Tamaño delta |
|---|---|---:|---:|
| v3.2.001 | BSC-LT/m-personas | 32.040 | 104,86 MB |
| v3.2.002 | BSC-LT/m-personas | 3.447 | 12,60 MB |
| v3.2.003 | Kukedlc/Personas_Spanish-40k | 35.997 | 29,02 MB |
| v3.2.004 | somosnlp/instruct-legal-refugiados-es | 9.091 | 35,89 MB |
| v3.2.005 | StrataSynth/stratasynth-personas-spain | 1.250 | 3,86 MB |
| **Total delta** | | **81.825** | **186,22 MB** |

Distribución por tipo de tarea:
- persona_dialogue: 35.487
- persona_profile: 35.997
- synthetic_persona_voice: 1.250
- contextual_legal_qa: 9.091

Así, **72.734/81.825 (88,9%)** de las nuevas filas v3.2 son perfiles, diálogos o voces de persona; solo 11,1% son QA jurídico contextual. El incremento añade volumen, pero no la diversidad temática que necesitamos. v3.2.001 es el único delta cercano al límite de 100 MiB; los otros cuatro son mucho menores. La cadena es acumulativa, no cinco bloques de 100 MB.

La cadena declara 279.613 filas acumuladas (197.788 de v3.1 + 81.825 deltas v3.2). Si la cifra operativa de la rama de entrenamiento es aproximadamente 180.000, hay que reconciliarla contra el manifiesto/artifact exacto antes de fijar el multiplicador. El objetivo mínimo solicitado se interpreta como 10× el corpus congelado efectivo, con 100× como objetivo aspiracional sujeto a datos elegibles reales.

## Diagnóstico de atractores de LUMEN

Se inspeccionaron las 100 respuestas libres de LUMEN del holdout del paso 417.500. Entre los patrones recurrentes observables aparecen:
- «Contexto documental»
- «democracia» y variantes de «democrático»
- «música»
- «deformación»
- «Latinoamericana»
- «Estados miembros» / «Unión Europea»

Se incorpora una lista de admisión dura, insensible a mayúsculas y acentos. **Cualquier documento que contenga cualquiera de esos términos o frases se rechaza completo**, aunque el documento pueda ser correcto en otros aspectos. Es una decisión deliberadamente estricta solicitada para cortar la recurrencia de esos atractores; no implica que esos temas sean intrínsecamente malos.

Aplicación de la lista a los 81.825 registros deltas v3.2:
- 15.936 filas contienen al menos un término/frase bloqueado (19,5%).
- «Contexto documental» aparece en las 9.091 filas del bloque de QA jurídico, por lo que ese bloque entero queda fuera con la regla actual.
- «música» aparece en 3.885 filas.
- Los recuentos por término se solapan y no deben sumarse.

La regla se implementa en scripts/filter_corpus_attractors.py y la lista queda versionada en data/corpus_v4_attractor_denylist.json. El filtro conserva las filas rechazadas en un fichero separado con la causa; no las destruye.

## Problemas adicionales detectados

1. **Concentración temática:** el corpus v3.2 añade mayoritariamente perfiles y diálogos de persona. Hace falta ciencia, historia, geografía, tecnología, ingeniería, matemáticas, naturaleza, economía, artes, vida cotidiana, documentación técnica y conversación general.
2. **Datos legales sintéticos o derivados:** el manifiesto de v3.2.004 indica respuestas generadas desde contexto y validez legal no verificada independientemente. No debe presentarse como autoridad legal vigente.
3. **Campos de persona:** v3.2.005 declara campos de origen excluidos, pero el texto de ejemplo todavía contiene nombre, edad y circunstancias familiares. Se debe auditar el texto serializado, no confiar solo en la lista de campos excluidos; minimizar o eliminar identificadores biográficos innecesarios.
4. **Calidad OCR:** libros y prensa históricos pueden aportar enorme diversidad, pero requieren filtros de OCR, idioma, longitud, duplicación y fragmentos incompletos.
5. **Compatibilidad del entrenador:** train/dataset.py exige prompt y completion y trunca a max_len=128; los deltas v3.2 contienen text. Hace falta un adaptador explícito antes de entrenar.
6. **Licencias:** la licencia del código del dataset no basta; se debe registrar la licencia del texto subyacente por documento o colección y excluir fuentes incompatibles/no verificadas.

## Objetivo de corpus v4

- **Mínimo:** 10× el número real de filas del corpus de entrenamiento congelado (si son 180.000, al menos 1,8 M).
- **Aspiracional:** 100× (si son 180.000, 18 M), solo si la ingestión real produce suficientes documentos únicos, útiles y con derechos verificables.
- El entrenador actual usa max_len=128 y espera prompt/completion, mientras los deltas v3.2 serializan text. El adaptador debe convertir texto documental a prompt vacío + completion y trocear a un máximo de 126 tokens BPE (reservando BOS/EOS); registrar source_document_id, chunk_index, offsets y procedencia. No almacenar párrafos de miles de palabras como una sola fila: el entrenador truncaría casi todo el documento.
- Mantener la personalidad como una franja estable de aproximadamente 10–15% del corpus final, no dejarla desaparecer bajo el volumen documental. La proporción se mide tras el filtro de atractores y deduplicación.
- Congelar una versión inmutable con manifiesto, SHA-256, conteo de filas/tokens, distribución por dominio/tipo, licencia y reporte de rechazos.

## Fuentes prioritarias para evaluación e ingestión

La propuesta inicial queda en data/corpus_v4_source_plan.json. Priorizar:
1. Libros y prensa españoles de dominio público (PleIAs Spanish-PD-Books / Spanish-PD-Newspapers), troceados y filtrados por OCR/calidad.
2. Common Corpus, solo subconjuntos españoles y documentos con licencia individual verificable compatible (dominio público o licencias libres seleccionadas).
3. Ciencia y divulgación de repositorios open access, con licencia comprobada por documento.
4. Wikidata/Wikipedia y Stack Exchange, respetando sus licencias y atribución/ShareAlike donde corresponda.
5. Documentación técnica, código y manuales con licencias permisivas verificadas.
6. Personalidad: ampliar perfiles y diálogos variados con datos sintéticos claramente etiquetados o fuentes con permiso/licencia explícita; excluir datos personales reales y campos biográficos identificables.

**No se congela ni se autoriza entrenamiento todavía.** Esta PR instala la auditoría inicial y el control de admisión; la ingestión completa, el recuento final y la congelación se harán después de ejecutar los lotes y revisar sus manifiestos.
