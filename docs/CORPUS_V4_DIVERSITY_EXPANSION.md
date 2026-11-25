# WISE-RONIN corpus v4 — expansión diversa y controlada

**Estado:** auditoría v3.2.005 realizada; pipeline de admisión preparado; v4 todavía no construido.  
**Entrenamiento autorizado:** no.  
**Base congelada:** artefacto de entrada del run 45398272055, spanish-v3-2.005-train.jsonl.

## 1. Auditoría material de v3.2.005

Se leyó el JSONL real del artefacto paired-v32-inputs-45398272055, no solo sus manifiestos.

- Filas: **279.613** (197.788 base + 81.825 incrementales).
- SHA-256 del train: 0f7db9d6c4264f68fd6a12a9fe5bfa708e4fc30c1df176de5a4f53081bfe79aa.
- JSONL descomprimido: **444.769.335 bytes**.
- 279.613/279.613 filas con language=es.
- Longitud media aproximada: 695 caracteres por fila; mediana 232.
- Personalidad: **72.734 filas (26,01%)**:
  - persona_profile: 35.997.
  - persona_dialogue: 35.487.
  - synthetic_persona_voice: 1.250.
- raw_text_pretraining: 178.090 filas (63,7%); la mayor parte del corpus es texto documental.

### Distribución por fuente

| Fuente | Filas |
|---|---:|
| tdx_es | 107.139 |
| csic_es | 70.951 |
| Kukedlc/Personas_Spanish-40k | 35.997 |
| BSC-LT/m-personas | 35.487 |
| somosnlp/instruct-legal-refugiados-es | 11.018 |
| OysterCoreAI/SFT-General-Spanish-50K | 6.892 |
| gplsi/uji_parallel_va_es | 5.573 |
| BSC-LT/ALIA-2606-SFT | 4.510 |
| StrataSynth/stratasynth-personas-spain | 1.250 |
| somosnlp/LingComp_QA | 741 |
| somosnlp/lenguaje-claro-dataset | 55 |

TDX y CSIC suman 178.090 filas, el 63,7% del conjunto. La siguiente versión debe reducir esta concentración mediante dominios y fuentes nuevas, no solo añadir más documentos parecidos.

## 2. Atractores actuales de LUMEN: veto de admisión

La primera lista se deriva de las salidas generadas por LUMEN en el holdout libre de 100 preguntas del paso 417.500:

- contexto documental
- democracia
- música
- deformación
- latinoamericana
- estados miembros
- unión europea

El escaneo de v3.2.005 encuentra 16.070 filas con al menos uno de esos términos (5,75%). Los principales focos son:
- 9.564 filas de la fuente legal contextual, principalmente por la plantilla Contexto documental.
- 2.512 perfiles de Personas_Spanish-40k.
- 2.336 diálogos de m-personas.
- 835 filas documentales TDX y 566 CSIC.

**No se eliminan retroactivamente filas del corpus v3.2 congelado.** El veto es una puerta de admisión para cada nuevo documento o fragmento que se proponga para v4. El filtro se aplica al texto que realmente verá el modelo, no a los metadatos externos. Usa coincidencia exacta de palabra/frase con normalización Unicode y límites de palabra, evitando falsos positivos por subcadenas (democrático no coincide con democracia).

La lista es deliberadamente LUMEN-específica. No se han bloqueado palabras genéricas como caso, personas, sido o enfoque. Se versionará si los holdouts posteriores revelan nuevos atractores.

Tras simular el veto sobre v3.2.005, quedan 263.543 filas y 67.875 filas de personalidad: **25,75%**. El peso de personalidad se mantiene sin tener que eliminar ni rebajar esa partición.

## 3. Objetivos de escala

El tamaño de referencia es 279.613 filas. La multiplicación se mide por filas utilizables después de todos los filtros, no por filas anunciadas en las fichas de datasets.

| Objetivo | Filas finales | Filas nuevas necesarias | Personalidad adicional para mantener 25% |
|---|---:|---:|---:|
| Base actual | 279.613 | — | — |
| **10× mínimo** | **2.796.130** | **2.516.517** | **626.299** |
| **100× aspiracional** | **27.961.300** | **27.681.687** | **6.917.591** |

Reglas:
- Mínimo de personalidad: **20%** del corpus final.
- Objetivo preferido: **25%**, cercano al 26,01% actual.
- La personalidad significa perfiles ficticios no psicométricos, voces diferenciadas, diálogos multi-turno, consistencia de carácter, preferencias no sensibles, desacuerdo razonado, humor, concisión, reconocimiento de errores y petición de aclaraciones.
- No incorporar etiquetas Big Five, diagnósticos, inferencias demográficas sensibles ni convertir correlaciones en estereotipos.
- La personalidad no se aumentará con miles de variaciones de la misma plantilla: diversidad de voces, registros, profesiones, edades ficticias y contextos sin convertirlos en estadísticas reales.

## 4. Inventario de bloques existentes y discrepancia de linaje

El encadenado v3.2.001–v3.2.005 contiene cinco deltas acumulativos:

| Delta | Filas | Bytes |
|---|---:|---:|
| v3.2.001 | 32.040 | 104.856.354 |
| v3.2.002 | 3.447 | 12.598.425 |
| v3.2.003 | 35.997 | 29.022.331 |
| v3.2.004 | 9.091 | 35.885.349 |
| v3.2.005 | 1.250 | 3.859.558 |

Solo el primer delta alcanza aproximadamente 100 MiB; los otros cuatro son incrementos menores. También existe un artefacto de la rama v3.1 candidata de 3.150.120.091 bytes. Su manifiesto de build registra **16.445.885 filas y 2.014.093.292 tokens**: parte de una base v3.0 de 9.933.007 filas y añade 6.550.199 fragmentos de ALIA Cultural Heritage y 2.700 filas SLR. Eso equivale a unas **58,8×** las filas de v3.2.005. **No se debe mezclar directamente con v3.2.005:** el manifiesto de v3.2 declara una base v3.1 candidata de 197.788 filas, por lo que hay dos linajes incompatibles que deben reconciliarse por SHA, manifiestos y procedencia.

El pipeline v4 escribe bloques JSONL de máximo 100 MiB sin comprimir, con número de filas, tamaño y SHA-256 por bloque. Un documento individual que supere el límite debe segmentarse por documento de origen, conservando source_document_id, offsets y hash del original.

## 5. Fuentes candidatas para la expansión

Orden de investigación, no autorización automática:

1. **PleIAs Common Corpus — español y licencias por documento.** Potencial de escala suficiente para el objetivo 100×, pero solo con allowlist de licencia a nivel de documento, lengua verificada, procedencia y exclusión de material sin derechos resueltos. No ingerir el agregado a ciegas.
2. **TDX Thesis Spanish Corpus.** La ficha de fuentes declara 9.790 documentos y 248,7 millones de tokens. Ya hay 107.139 filas TDX en v3.2; comparar IDs y hashes antes de ampliar, retirar bibliografía/boilerplate y evitar duplicar fragmentos.
3. **ESQAD.** Archivo de unos 676 MB; solo partición train claramente licenciada, separando benchmark, test, subconjuntos de evaluación humana y datos generados sin procedencia suficiente.
4. **Spanish-PD-Books y PleIAs BDH-Books.** Alto potencial de volumen. Verificar dominio público por obra y edición, OCR, atribución y deduplicación cruzada entre ambos catálogos.
5. **ALIA Spanish Cultural & Heritage.** Gran cobertura cultural e histórica. La licencia declarada CC BY-SA 4.0 exige una decisión de compatibilidad y trazabilidad antes de mezclarlo con el conjunto principal; mantener aislado mientras se resuelve.
6. **GovEasy y documentación pública española.** Menor volumen pero alta diversidad práctica; exigir URL oficial, fecha, vigencia y retirada de procedimientos obsoletos.
7. **OpenAlex Philosophy y fuentes académicas CC0/CC BY.** Solo metadatos/abstracts que tengan derechos claros; no reconstruir ni copiar artículos protegidos.
8. **Personalidad y diálogo.** Reauditar el remanente español de OysterCoreAI/SFT-General-Spanish-50K; buscar corpus de diálogo multi-turno y voces españolas con licencia explícita; ampliar perfiles sintéticos no sensibles con estilos realmente distintos. m-personas y Personas_Spanish-40k ya aportan 71.484 filas y no deben reimportarse sin deduplicación.

### Fuentes excluidas o separadas

- No incorporar CC-BY-NC, CC-BY-NC-SA ni fuentes con derechos upstream no resueltos al corpus principal.
- MS MARCO ES es retrieval/ranking: no aplanar sus triples como SFT generativo.
- No convertir NER, etiquetas clínicas ni anotaciones de extracción en diálogo inventado.
- No usar particiones test/validation/benchmark.
- No admitir datasets médicos con riesgo de reidentificación ni registros clínicos sin garantías comprobadas.
- No inventar pares pregunta-respuesta para documentos de pretraining: conservarlos como texto documental con su procedencia.

## 6. Política obligatoria de admisión v4

Cada nueva fuente debe:
1. Fijar revisión/commit inmutable y guardar SHA-256.
2. Verificar licencia del dataset y derechos upstream; allowlist inicial: Apache-2.0, MIT, CC BY 4.0, CC0 y dominio público por documento verificado.
3. Separar train de validation/test/benchmark.
4. Confirmar español por registro, no por nombre del repositorio.
5. Exigir source, source_revision, license, language, task y, para fragmentos, ID/hash/URL del documento de origen.
6. Ejecutar el veto de atractores, PII, texto vacío, OCR roto, boilerplate, spam y respuestas desalineadas.
7. Deduplicar exacto y aproximado contra v3.2 y todas las fuentes nuevas.
8. Inspeccionar muestra estratificada de al menos 100 registros por fuente (todos si hay menos), además de extremos de longitud.
9. Mantener el porcentaje de personalidad dentro del objetivo 20–25% y medir la diversidad de estilos, no solo el número de filas.
10. Escribir bloques de hasta 100 MiB, manifiestos por bloque y manifiesto acumulativo del corpus.
11. Congelar el corpus y el cache; LUMEN y Transformer reciben exactamente los mismos datos, tokenizer y orden de evaluación.
12. No entrenar hasta que todos los gates y el holdout congelado estén verificados.

## 7. Herramienta reproducible

scripts/audit_corpus_v4_gate.py audita JSONL, cuenta fuentes/tareas/licencias, aplica el veto, puede escribir un JSONL limpio y dividir aceptados en bloques de 100 MiB con hashes. La opción --require-provenance activa la comprobación estricta de procedencia/licencia para nuevas admisiones.

Prueba local:
- 3 tests de coincidencia exacta, falsos positivos por subcadenas y clasificación de personalidad.
- El informe de auditoría conserva el SHA del archivo de entrada.
- El script nunca modifica el corpus base y siempre emite training_authorized=false.

**Siguiente hito:** reconciliar el artefacto v3.1 candidato de 3,15 GB, auditar las fuentes de mayor escala y construir el primer lote v4. No iniciar entrenamiento con una versión parcial.


## 8. Auditoría previa del candidato grande v3.1

El artefacto se construyó en el run 45155150980 y conserva el estado BUILT_NEEDS_FINAL_AUDIT, con training_authorized=false. El informe de su base v3.0 (run 45142717502) impone gates aún abiertos:

- 19.740 near-duplicates confirmados con Jaccard de trigramas >=0,88 y ratio de longitud >=0,80.
- 157.874 consultas superaron el límite de 1.000 candidatos; no se revisaron completamente.
- 41 alertas heurísticas de posible marcador sensible: 13 en literatura y 28 en m-personas. No son incidentes confirmados; deben revisarse.
- 608 documentos literarios de train tienen revisiones y derechos en el manifiesto upstream, pero los fragmentos v3 no conservan ID/hash de documento; hay que restaurar esa relación.
- 63 completions largas repetidas frente a v2; cero duplicados exactos de pares y cero coincidencias exactas con validation/holdout.
- La categoría de personalidad verifiable_personality contiene 50.000 registros, pero su manifiesto indica 40 tarjetas factuales y siete URLs institucionales: no son 50.000 hechos independientes.
- Las 119.026 filas de synthetic_curricula derivan de 24×10 casos semilla revisados; no deben contarse como 119.026 piezas de conocimiento factual independiente. La muestra de m_personas y verifiable_personality también repite un prefijo largo de identidad estable en los prompts; v4 debe conservar la identidad canónica en una partición pequeña y no repetir ese prefijo en cada ejemplo si no aporta señal nueva.
- La construcción v3.1 añadió 6.550.199 fragmentos ALIA con licencia declarada CC BY-SA 4.0; antes de mezclarlos en el corpus principal hay que resolver compatibilidad, atribución y obligaciones de compartir igual. La construcción retiró 75.016 filas marcadas y omitió 11.570 duplicados ALIA. Además, el builder cortó esos documentos en fragmentos de 180 tokens, mientras que train/dataset.py limita cada secuencia a max_len=128 (127 tokens más EOS al truncar). Por tanto, los 2.014 millones de tokens del manifiesto son conteo previo al truncado, no tokens efectivos garantizados para entrenamiento. V4 debe volver a segmentar el texto documental en bloques de como máximo 127 tokens de contenido o cambiar y validar explícitamente la longitud de contexto.

La base v3.0 por sí sola ya representa aproximadamente 35,5× las filas de v3.2.005, pero no está autorizada: debe superar los gates de near-duplicates, desbordamiento de candidatos, alertas, trazabilidad literaria y derechos. El candidato completo de 16,45 M filas es una vía potencial para acercarse a 100×, pero no debe tratarse como corpus listo para entrenar.

El cache actual v3.2.005 contiene 22.240.374 tokens BPE efectivos (incluido EOS, sin padding), max_len=128 y tokenizer SHA d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe. El manifiesto v3.1 declara 2.014.093.292 tokens antes del límite de secuencia: una razón bruta de 90,6×, no una comparación efectiva todavía. El cache candidato debe construirse con max_len=128 para conocer la cifra comparable; el corte de 180 tokens hace probable una reducción importante.

El workflow de auditoría del candidato v3.1 debe ejecutar el veto de atractores sobre el archivo completo, medir la proporción real de personalidad y generar un informe reproducible antes de crear bloques limpios.


## 9. Mantener el peso de personalidad sin duplicar filas

El cargador actual usa muestreo uniforme de filas (rng.choices en train/dataset.py). Si se incorpora un corpus documental de 10–16 millones de filas como un único JSONL, la probabilidad de ver personalidad cae en proporción a su porcentaje bruto, aunque v3.2.005 tenga hoy un 26,01%.

Por eso v4 debe conservar particiones etiquetadas y no depender solo de un JSONL plano:
- Mantener un conjunto único de perfiles/diálogos/voz/personality-QA separado del pretraining documental.
- Medir la cuota por tokens efectivos después de max_len, no por filas ni por tokens previos al truncado.
- Objetivo de entrenamiento: 20% mínimo y 25% preferido de tokens de personalidad/voz/identidad, sujeto a calidad y tamaño de muestra.
- Añadir diversidad única de personalidad; no inflar el porcentaje replicando miles de veces las mismas filas.
- Aplicar un límite de repetición por ejemplo y por fuente; si la cuota no puede cubrirse con suficiente variedad, ampliar primero el pool en vez de forzar el sampler.
- Mantener identidad canónica en una partición pequeña y explícita; no repetir el mismo prefijo de identidad en decenas de miles de ejemplos si no aporta señal nueva.

La auditoría del candidato grande debe calcular la proporción real de tareas de personalidad y, después, construir un sampler por tarea/token. No se debe declarar que conserva el peso de personalidad hasta medir esa cuota efectiva.
