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

## Distribución del corpus completo v3.2.005\n\n| Grupo | Filas | Proporción |\n|---|---:|---:|\n| raw_text_pretraining | 178.090 | 63,7% |\n| persona_profile + persona_dialogue + synthetic_persona_voice | 72.734 | 26,0% |\n| Otros SFT / QA / tareas | 28.789 | 10,3% |\n| **Total** | **279.613** | **100%** |\n\nEl corpus ya tiene un bloque de personalidad sustancial: 72.734 filas (26,0%). Tras la auditoría de atractores y licencias, la base conserva aproximadamente 57.870 filas de personalidad/diálogo. La cuota nueva se recalibró a 180.000 fragmentos: 118.565 de m-personas y 61.435 de SFT-General-Spanish. Con 251.230 filas base admitidas y 1,8 M de filas documentales, la proyección es de unas 237.870 filas de personalidad/diálogo sobre 2.231.230 filas totales (≈10,66%). Es una previsión, no un resultado: el porcentaje definitivo se calcula después de los filtros y la deduplicación global, y debe quedar entre el 10% y el 15%.\n\n## Diagnóstico de atractores de LUMEN

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

El token-cache heredado no se acepta para este modo indexado: pertenece a un corpus/hash concreto y su representación empaquetada cargaría el conjunto completo en memoria. Para v4 se usa tokenización por lote en el muestreo aleatorio. Antes de declarar entrenable el artefacto se ejecutarán pruebas de lectura aleatoria, verificación de hashes y rechazo de shards corruptos. El artefacto congelado incluye también una copia de `tokenizer.json`; el trainer debe invocarse con `--data <directorio-v4> --tokenizer <directorio-v4>/tokenizer.json`, sin `--token-cache`.

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


## Reauditoría documental del artefacto emparejado (2026-11-24)

Esta sección sustituye las estimaciones preliminares anteriores sobre el alcance de la lista de atractores. Se ha vuelto a recorrer el JSONL exacto de 279.613 filas con la lista ampliada y la regla de agrupación documental usada por el merge.

- Filas con coincidencia directa en cuerpo o metadatos: **20.590**.
- Identificadores documentales afectados: **19.168**.
- Filas rechazadas al extender el rechazo a todos los fragmentos hermanos del documento: **28.383**.
- Filas base restantes tras el rechazo documental: **251.230**.
- Del bloque raw_text_pretraining se rechazan 2.217 filas; permanecen 175.873.
- Del bloque de personalidad se rechazan 14.864 filas; permanecen 57.870.
- Se rechazan las 9.091 filas del bloque contextual_legal_qa al aparecer el atractor en el texto.

**Corrección de identidad documental:** source_doi identifica en csic_es y tdx_es la colección/dataset, no cada documento. No se usa para agrupar rechazos. La clave prioriza source_document_id, identificadores/URLs documentales, source_id y source_row; en m-personas se agrupan los turnos por el ID de persona padre. Se añadió una prueba de regresión para impedir que un DOI compartido expulse documentos no relacionados.

### Ajuste de la cuota de personalidad

Con el rechazo documental completo, el candidato inicial de 160.000 filas dejaría el bloque de personalidad por debajo del mínimo del 10% después de la deduplicación y la mezcla. La cuota se elevó a **180.000 filas** (114.545 m-personas + 65.455 SFT-General Spanish). La proporción real solo se podrá confirmar después del merge, los rechazos y la deduplicación global.

Este cálculo no autoriza el entrenamiento. La compilación documental de 1,8 M, el merge final y la auditoría independiente de los shards siguen siendo requisitos.


## Política estricta de admisión aplicada a la construcción v4 (25-11-2026)

Para esta compilación se endurece la admisión: solo entran registros con licencia explícita exactamente identificada como dominio público, Apache-2.0 o CC-BY-4.0. Se rechazan licencias ausentes, no comerciales, internas, contradictorias o con obligaciones adicionales pendientes de revisión.

- **Documental admitido:** PleIAs/Spanish-PD-Books, revisión fijada, con filtro de idioma, OCR/calidad, PII, atractores y deduplicación.
- **En espera:** prensa histórica PleIAs, hasta acreditar dominio público por publicación/documento; Wikipedia española, hasta revisar la compatibilidad CC BY-SA 3.0/GFDL y sus obligaciones.
- **Personalidad:** BSC-LT/m-personas (Apache-2.0) y SFT-General-Spanish-50K (CC-BY-4.0), train únicamente, con procedencia por registro.
- **Base v3.2.005:** se conservan solo filas con licencia exacta admitida; los registros «Apache-2.0 (internal)» y «CC-BY-4.0 (internal)» se rechazan. Los registros heredados conservan su identificador estable de origen; no se inventan revisiones de fuente que no estén documentadas.
- **Fuera:** SQAC mientras persista la discrepancia de licencia; AQuAS/RagQuAS por licencia NC; SIQA_es/openbookqa-es/EsBBQ por holdout o split no admisible.

El filtro de licencia se aplica antes de tokenización, deduplicación y escritura. El manifiesto debe contar rechazos por licencia. Esta versión sigue siendo candidata y no autoriza entrenamiento.

### Recuento de licencias heredadas verificado en el JSONL real

Se inspeccionaron las 279.613 filas del fichero de entrenamiento, no solo el manifiesto:

| Licencia declarada | Filas | Decisión |
|---|---:|---|
| CC-BY-4.0 | 191.805 | Admitir |
| Apache-2.0 | 83.966 | Admitir |
| Apache-2.0 (internal) | 3.781 | Rechazar |
| CC-BY-4.0 (internal) | 61 | Rechazar |

El gate estricto retira **3.842 filas** con etiquetas internas no equivalentes a una licencia pública estándar. Las 275.771 filas restantes siguen sujetas a los filtros de atractores por documento, idioma, PII, longitud y deduplicación; esta tabla no equivale a afirmar que todas superan los demás controles.

## Integridad de la congelación y trazabilidad de candidatos (25-11-2026)

La congelación no debe tomar silenciosamente un artefacto antiguo cuando existe una compilación más reciente en curso o fallida. El workflow se endurece para:

- Seleccionar el último run de cada builder, no el último run exitoso histórico.
- Detener la congelación si ese último run no ha terminado correctamente; no retroceder a un candidato antiguo.
- Registrar tanto el ID del run como su commit de origen en el manifiesto final.
- Verificar que los commits candidatos pertenecen a la historia de la rama y que los scripts de construcción y la denylist no han cambiado desde la compilación.
- Mantener separados los artefactos candidatos (siempre `training_authorized=false`) del corpus final congelado.

La modificación del workflow está en la rama de la PR #97. La compilación OpenCulture estricta actual es [45561584743](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45561584743), basada en el commit `44db282`; sigue construyendo shards y aún no ha publicado artefactos. La compilación de personalidad recalibrada es [45561825977](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45561825977), con cuota de 180.000 filas (118.565 m-personas + 61.435 SFT-General-Spanish); también sigue construyendo y todavía no hay manifiesto candidato que verificar. El artefacto anterior de 165.000 filas queda descartado.

**Estado de esta auditoría:** corpus todavía no entrenable. La cifra de filas candidatas, por sí sola, no cambia este estado. Solo el artefacto final con hashes verificados, objetivos cumplidos, proporción de personalidad dentro del 10–15%, procedencia/licencias comprobadas, ausencia de atractores/PII/holdout y smoke test del cargador real puede recibir autorización.


## Reescalado de personalidad para la expansión documental x10 (25-11-2026)

La cuota anterior de 165.000 filas era un mínimo ajustado al límite inferior del 10%, no una reserva con margen. El cálculo debe incluir las 57.870 filas de personalidad que sobreviven en la base tras la auditoría de atractores; no se debe calcular la cuota como si la base careciera de personalidad. Con 251.230 filas base admitidas, 1.800.000 filas documentales nuevas y 165.000 filas nuevas de personalidad, la estimación previa era aproximadamente 10,06%, antes de los rechazos finales por licencia y deduplicación global.

Para no trabajar pegados al umbral, la cuota candidata se eleva a **180.000 filas nuevas** manteniendo el reparto de fuentes: **114.545 m-personas (Apache-2.0)** y **65.455 SFT-General-Spanish (CC-BY-4.0)**. Bajo los recuentos auditados, esto proyecta aproximadamente **10,66%** de personalidad/diálogo en el corpus final. Es una proyección, no un resultado: el manifiesto de merge debe calcular la proporción real después de todos los filtros y deduplicación.

El gate del merge sube a 175.000 filas mínimas de personalidad/diálogo y conserva la condición obligatoria de share entre 10% y 15%. Si la compilación documental supera 1,8 M o los rechazos alteran la mezcla, el gate debe fallar y la cuota se recalculará; nunca se autoriza por cumplir solo el recuento bruto del builder.

## Control de congelación actualizado (25-11-2026)

La ejecución antigua [45559065309](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45559065309), iniciada con la configuración previa, quedó cancelada y no produjo artefactos. La nueva ejecución [45562267530](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45562267530) falló deliberadamente en el control inicial: detectó que el último candidato OpenCulture seguía en curso y se negó a reutilizar un artefacto anterior. No descargó candidatos, no hizo merge y no autorizó entrenamiento.

Se corrigió además el orden final: el smoke test del cargador de producción debe pasar mientras el manifiesto sigue con `training_authorized=false`; solo después se cambia a `true`. Esta corrección está en el workflow de la rama. La próxima congelación se lanzará cuando ambos builders actuales hayan terminado y sus manifiestos se hayan auditado.

### Validación de código

El workflow de comprobaciones generales [45561843473](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45561843473), sobre el código de construcción y merge de la revisión 2f325bd, terminó correctamente: **159 pruebas pasaron en 5,72 s**. Esta validación cubre el conjunto de tests del repositorio; no sustituye la verificación de los artefactos de los dos builders ni el smoke test del corpus final.


### Rebalanceo por capacidad observada (25-11-2026)

La ejecución 45561825977 alcanzó 175.980 filas: m-personas completó 114.545/114.545; SFT-General-Spanish produjo 61.435/65.455. El manifiesto registró 72.273 chunks exactos duplicados y falló correctamente por quedar 4.020 filas por debajo del objetivo. No se rebaja el objetivo total: se recalibra el reparto a 118.565 m-personas y 61.435 SFT-General-Spanish, usando la capacidad observada del split train y ampliando la cuota de m-personas. La nueva ejecución es [45563015593](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45563015593); todavía debe terminar y publicar un manifiesto verificado.
