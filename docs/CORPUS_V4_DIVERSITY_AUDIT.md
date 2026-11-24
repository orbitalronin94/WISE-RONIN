# Corpus v4 — auditoría de diversidad y admisión

## Estado

Auditoría inicial realizada sobre el artefacto acumulativo de la cadena v3.2 descargado desde GitHub Actions. Se inspeccionaron íntegramente los cinco deltas v3.2.001–v3.2.005 (JSONL), sus manifiestos y el índice de deduplicación incluido en el artefacto.

Además de los deltas, se descargó y examinó el JSONL exacto usado por el entrenamiento emparejado (279.613 filas, SHA-256 0f7db9d6c4264f68fd6a12a9fe5bfa708e4fc30c1df176de5a4f53081bfe79aa). La auditoría léxica global cubre ya el corpus efectivo completo. La distribución por fuente de los incrementos v3.2 sigue describiendo sus 81.825 filas nuevas.

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

La base exacta emparejada con los modelos contiene **279.613 filas** (197.788 de v3.1 + 81.825 deltas v3.2), de las cuales **178.090 son `raw_text_pretraining`**. La cifra de ~180.000 corresponde al bloque documental, no al total del corpus. Por tanto, el objetivo actual de 1,8 M de filas documentales equivale a **10,11× el bloque documental de partida**, pero no a 10× todas las filas de la base. Para llegar a 10× el total completo harían falta 2.796.130 filas finales; ese objetivo no debe confundirse con el alcance actual de v4.

## Distribución del corpus completo v3.2.005\n\n| Grupo | Filas | Proporción |\n|---|---:|---:|\n| raw_text_pretraining | 178.090 | 63,7% |\n| persona_profile + persona_dialogue + synthetic_persona_voice | 72.734 | 26,0% |\n| Otros SFT / QA / tareas | 28.789 | 10,3% |\n| **Total** | **279.613** | **100%** |\n\nEl corpus ya tiene un bloque de personalidad sustancial: 72.734 filas (26,0%). El filtro actual rechaza 6.845 de esas filas por atractores, dejando 65.889. Sin ampliación, al sumar 1,8 M de texto general el bloque caería a cerca del 3%. El plan fija una franja final del 10–15% de personalidad/conversación. El builder reserva 100.000 fragmentos para m-personas y 60.000 para diálogo SFT español (el primer benchmark de ingesta produjo 80.000 y 60.287 respectivamente, por lo que se reajustó el reparto). Junto a las filas conservadas deja aproximadamente 225.889 filas de personalidad/diálogo sobre un total previsto de 2,22 M (≈10,2%). No intenta conservar artificialmente el 26% original: eso exigiría cientos de miles de ejemplos adicionales y aumentaría el riesgo de sobrepeso sintético.\n\n## Diagnóstico de atractores de LUMEN

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

## Carga de entrenamiento para corpus grande

El loader de manifiestos ya no materializa todos los objetos JSONL en RAM. Construye índices laterales compactos de offsets de 64 bits por shard, verifica cada SHA-256 mientras recorre los ficheros y permite muestreo aleatorio perezoso de filas. El trainer accede a las filas seleccionadas por lote; no crea una lista Python de millones de diccionarios. Los índices se guardan en `.corpus_index/` y pueden reconstruirse sin alterar los shards congelados.

El token-cache heredado no se acepta para este modo indexado: pertenece a un corpus/hash concreto y su representación empaquetada cargaría el conjunto completo en memoria. Para v4 se usa tokenización por lote en el muestreo aleatorio. Antes de declarar entrenable el artefacto se ejecutarán pruebas de lectura aleatoria, verificación de hashes y rechazo de shards corruptos.

## Objetivo de corpus v4

- **Objetivo de esta tanda:** al menos 1,8 M de filas documentales nuevas, 10,11× las 178.090 filas documentales de la base emparejada.
- **No equivale a 10× el corpus completo:** la base tiene 279.613 filas; 10× el total exigiría 2.796.130 filas finales. El informe de congelación debe mostrar ambos multiplicadores y no confundirlos.
- **Aspiracional:** 100× el bloque documental (17,81 M nuevas) solo si la ingestión produce suficientes documentos únicos, útiles y con derechos verificables.
- El entrenador actual usa max_len=128 y espera prompt/completion, mientras los deltas fuente v3.2 serializan text. El adaptador debe convertir texto documental a prompt vacío + completion y trocear a un máximo de 126 tokens BPE (reservando BOS/EOS); registrar source_document_id, chunk_index, offsets y procedencia. No almacenar párrafos de miles de palabras como una sola fila: el entrenador truncaría casi todo el documento.
- Mantener la personalidad como una franja estable de aproximadamente 10–15% del corpus final, no dejarla desaparecer bajo el volumen documental. La proporción se mide tras el filtro de atractores y deduplicación.
- Congelar una versión inmutable con manifiesto, SHA-256, conteo de filas/tokens, distribución por dominio/tipo, licencia y reporte de rechazos.

## Fuentes prioritarias para evaluación e ingestión

La propuesta inicial queda en data/corpus_v4_source_plan.json. Priorizar:
1. Libros y prensa españoles de dominio público (PleIAs Spanish-PD-Books / Spanish-PD-Newspapers), troceados y filtrados por OCR/calidad.
2. Wikipedia española (dump 20231101.es, 13 shards, 3,49 GB), con CC BY-SA 3.0 + GFDL; se conserva URL de artículo, atribución y obligaciones ShareAlike.
3. Common Corpus, solo subconjuntos españoles y documentos con licencia individual verificable compatible (dominio público o licencias libres seleccionadas).
4. Ciencia y divulgación de repositorios open access, con licencia comprobada por documento.
5. Wikidata CC0 y Stack Exchange CC BY-SA, como fuentes separadas y con sus respectivas condiciones.
6. Documentación técnica, código y manuales con licencias permisivas verificadas.
7. SQAC (18.817 QA; 15.036 train) queda en espera: los metadatos declaran CC BY-SA 4.0, pero el cuerpo del README declara CC BY 4.0. No se incorpora hasta resolver la discrepancia de licencia.
8. SIQA_es y openbookqa-es permanecen excluidos por ser fuentes del holdout; EsBBQ solo ofrece test. AQuAS y RagQuAS quedan excluidos por CC BY-NC-SA no comercial.
9. Personalidad: ampliar perfiles y diálogos variados con datos sintéticos claramente etiquetados o fuentes con permiso/licencia explícita; excluir datos personales reales y campos biográficos identificables.

**No se congela ni se autoriza entrenamiento todavía.** Esta PR instala la auditoría inicial y el control de admisión; la ingestión completa, el recuento final y la congelación se harán después de ejecutar los lotes y revisar sus manifiestos.
