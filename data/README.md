# Corpus español v1 — procedencia, splits y hashes

## Estado actual

**Este es el corpus congelado que utilizan los entrenamientos emparejados de Lumen-Ronin y Transformer.** No debe confundirse con los antiguos planes de staging de fuentes ni con los datasets de ramas exploratorias.

El artefacto preparado y reutilizado por las continuaciones es [ronin-spanish-prepared-16000-45021399600](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45021399600). La evaluación se realiza sobre un holdout independiente que no participa en el entrenamiento ni en la actualización de glucosa, dolor o placer.

## Recuento de filas

| Conjunto | Filas | Uso |
|---|---:|---|
| Train | 70.105 | Actualización de parámetros |
| Validation | 1.413 | Validación separada |
| Holdout | 2.431 | Evaluación final congelada |
| Train + validation deduplicados | 71.518 | Base de entrenamiento/validación |
| Registros raw antes de deduplicar | 71.753 | Entrada al builder |
| Holdout + train + validation | 73.949 | Total de filas en los tres splits |

El holdout se compone de 1.931 preguntas de SIQA-es y 500 de OpenBookQA-es. Son fuentes de evaluación separadas del pool train/validation.

## Fuentes de entrenamiento

| Fuente | Filas de origen en el manifiesto | Licencia declarada |
|---|---:|---|
| `somosnlp/instruct-legal-refugiados-es` | 9.288 | Apache-2.0 |
| `OysterCoreAI/SFT-General-Spanish-50K` | 25.000 | CC-BY-4.0 |
| `gplsi/uji_parallel_va_es` | 19.939 | CC-BY-4.0 (solo columna española) |
| `BSC-LT/ALIA-2606-SFT` | 17.291 | CC-BY-4.0 a nivel de tarjeta, con allowlist por componente |

La fuente ALIA se filtra por componente. No se aplica su licencia general de forma indiscriminada a todos sus subconjuntos. El manifiesto excluye expresamente componentes NC, ND, de licencia desconocida o incompatible.

## Fuentes de evaluación

| Fuente | Filas | Licencia declarada |
|---|---:|---|
| `BSC-LT/SIQA_es` | 1.931 | CC-BY-4.0 |
| `BSC-LT/openbookqa-es` | 500 | CC-BY-4.0 |

El holdout no se usa para entrenamiento, ajuste de órganos, selección de hiperparámetros ni alimentación de señales internas. La evaluación emparejada carga ambos checkpoints y puntúa las mismas preguntas.

## Reproducibilidad

- Seed de construcción: `42`.
- Idioma: español (`es`).
- Normalización Unicode NFC, decodificación de entidades HTML y normalización de NUL/espacios.
- Identificación de idioma y deduplicación exacta por prompt/completion normalizados.
- Filtrado de componentes por licencia en ALIA.
- Manifiesto con procedencia, recuentos, política de licencias y SHA-256 por split.
- Corpus y tokenizer se descargan como artefactos congelados en los workflows de continuación.
- Token cache de contexto 128 preparado antes del entrenamiento; el workflow de entrenamiento no vuelve a tokenizar el corpus.

### Hashes del manifiesto

| Split | SHA-256 |
|---|---|
| Train | `d0dc9e12dc1b72b01eeafd42e204c59543e2196fc516ff1d67b1be9f78632b3f` |
| Validation | `2efacb8236c1041534199709791e63469ce36f64335aef1b60c8ff0c57a986b6` |
| Holdout | `e21ae7e292d603f1b8073d8114572c92fb0c6448bfaed9066413ddce09e5a359` |

El ZIP del artefacto preparado para la continuación de 16k tiene SHA-256:

`6595087077b476fa7557831b2da08fc01d9e6f54a4a23e1a0dd5737d17304e58`

## Naturaleza y límites del corpus

Este conjunto es principalmente un corpus español de instrucciones, QA, texto paralelo y SFT. **No es un corpus general de preentrenamiento web.** El manifiesto incluye tareas como QA legal, SFT general, texto paralelo, seguimiento de instrucciones y QA contextual.

Las etiquetas de licencia son las declaradas por las tarjetas y los componentes admitidos; no sustituyen una revisión legal de los materiales subyacentes. La presencia de un ejemplo en un dataset no convierte automáticamente su contenido en un hecho verificado. Los ejemplos jurídicos permanecen etiquetados por dominio y no constituyen asesoramiento legal.

## Workflow y artefacto

- Preparación y congelación: [run 45021399600](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45021399600).
- Artefacto compartido: `ronin-spanish-prepared-16000-45021399600`.
- Holdout más reciente completado: [step 61.000, run 45044892658](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45044892658).
- Continuación activa: [61.000 → 66.000, run 45061486739](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45061486739).

**Regla:** no se cambia el corpus, tokenizer ni holdout a mitad de una comparación emparejada. Cualquier nueva versión debe tener manifiesto, hashes y una evaluación que permita distinguir el efecto del modelo del efecto de los datos.


---

## Ampliación v2: corpus común para Lumen y Transformer

**Estado: PREPARACIÓN PENDIENTE — entrenamientos congelados.**

Se ha decidido ampliar el corpus de texto español con tres recursos preprocesados, manteniendo un único conjunto de entrenamiento para las dos arquitecturas. La ampliación no se considera terminada ni se atribuyen todavía recuentos nuevos: los tamaños publicados por los proveedores no equivalen a tokens efectivamente ingeridos por nuestra canalización.

| Fuente | Contenido | Licencia declarada | Tratamiento |
|---|---|---|---|
| [CSIC Spanish Corpus](https://zenodo.org/records/7313126) | Artículos y revistas científicas en español; 30.929 documentos y ~146,8 M tokens declarados | CC BY 4.0 | Candidato de inclusión; conservar atribución y metadatos |
| [TDX Spanish Thesis Corpus](https://zenodo.org/records/7313149) | Tesis académicas en español; 9.790 documentos y ~248,7 M tokens declarados | CC BY 4.0 | Candidato de inclusión; conservar atribución y metadatos |
| [CoWeSe — Spanish Biomedical Crawled Corpus](https://doi.org/10.5281/zenodo.5513237) | Corpus biomédico español preprocesado, procedente de un rastreo de más de 3.000 URL | CC BY 4.0 en el registro del recurso | Revisión de derechos de los textos subyacentes antes de incorporarlo |

**Advertencia de licencia sobre CoWeSe:** el registro Zenodo declara CC BY 4.0 para el recurso, pero la descripción de sus autores aclara que no son titulares de los textos recolectados. Por ello, no se tratará la licencia del paquete como prueba automática de que cada documento subyacente puede redistribuirse o utilizarse sin restricciones. CoWeSe permanecerá separado hasta completar una revisión de procedencia/derechos; no se mezclará silenciosamente.

### Contrato de integración

1. Descargar cada fuente desde su registro oficial y fijar versión, URL, fecha, tamaño y SHA-256.
2. Mantener el texto continuo como texto de preentrenamiento; no convertirlo artificialmente en pares de instrucciones y respuestas.
3. Normalizar, filtrar documentos vacíos/no españoles, eliminar duplicados exactos y detectar solapamientos con el train v1.
4. Preservar por documento la fuente, identificador disponible, licencia, URL/DOI y estadísticas de limpieza.
5. Construir secuencias compatibles con el contexto 128 y el tokenizador compartido actual. No cambiar el vocabulario BPE de 4.096 IDs sin un plan explícito de migración de embeddings/checkpoints.
6. Mantener la validación (1.413 registros) y el holdout (2.431 preguntas) byte por byte; no introducirlos en entrenamiento ni utilizarlos como material de deduplicación que altere la evaluación.
7. Publicar un manifiesto v2 con recuentos reales, tokens medidos por nuestro tokenizador, exclusiones, hashes y auditoría de solapamiento.
8. Hacer que Lumen y Transformer descarguen exactamente el mismo artefacto v2, tokenizer y token cache. No habrá corpus específico por modelo.

### Regla de pausa

No iniciar nuevos entrenamientos, benchmarks de entrenamiento ni continuaciones de checkpoints hasta que:

- la ingestión y limpieza estén completadas;
- el manifiesto y los hashes sean reproducibles;
- las secuencias y la caché hayan pasado validaciones de integridad;
- se confirme que validación y holdout permanecen intactos;
- ambos workflows de entrenamiento apunten al mismo artefacto versionado.

El corpus v1 y sus hashes anteriores se conservan como referencia histórica. La versión v2 será una nueva versión de datos, no una reescritura retrospectiva de los resultados 0→66k.


### Constructor reproducible y artefact congelado

El constructor está en `scripts/build_spanish_corpus_v2.py` y la ejecución automatizada en [build-spanish-corpus-v2.yml](../.github/workflows/build-spanish-corpus-v2.yml).

- Run de construcción: [45065384110](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45065384110).
- Fuentes ingeridas: CSIC y TDX, descargadas de Zenodo y verificadas contra los MD5 publicados.
- Tokenizador: se reutiliza el tokenizador BPE compartido del artefacto preparado 16k; no se reentrena ni cambia el vocabulario.
- Formato: JSONL compatible con el cargador actual. Los textos largos se dividen en ventanas de hasta 126 tokens BPE para dejar espacio a BOS/EOS en el contexto de 128.
- Evaluación: validación y holdout se copian byte por byte desde v1 y se verifican por SHA-256.
- Salida prevista: artefacto Actions `wise-ronin-spanish-corpus-v2-RUN_ID`, con `train.jsonl`, `validation.jsonl`, `holdout_multiple_choice.jsonl`, `manifest.json` y `tokenizer.json`.
- Retención prevista: 90 días. El manifiesto registra hashes, procedencia, DOI, licencia y recuentos reales.

**No declarar v2 listo para entrenar hasta que el run termine en verde y exista el artefacto descargable.** El manifiesto de construcción y los logs son la autoridad para los recuentos; las cifras de los registros originales son solo referencias de proveedor.
