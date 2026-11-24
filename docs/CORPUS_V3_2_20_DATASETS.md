# Curación de 20 fuentes para el corpus español v3.2

**Estado: registro de curación; no es un corpus construido ni autorizado para entrenamiento.**

La selección se ha diseñado para ampliar de verdad la biblioteca y evitar que el corpus se convierta en una colección de QA repetitivas. Separamos cinco carriles: conocimiento documental, conversación/persona, instrucción general, conocimiento especializado y tareas supervisadas.

## Distribución propuesta

| Carril | Fuentes | Objetivo |
|---|---:|---|
| Personalidad y conversación | 5 | Identidad conversacional consistente, estilos, perfiles y diálogo condicionado |
| Instrucción general y QA | 5 | Explicación, educación, lenguaje claro, trámites y QA contextual |
| Preentrenamiento documental | 5 | Literatura, humanidades, tesis, documentos multidominio |
| Recuperación | 1 | Aprender consultas, relevancia y contraste positivo/negativo |
| Ciencia y salud | 4 | Texto biomédico y extracción especializada |

## Las 20 fuentes seleccionadas

| # | Fuente | Licencia declarada | Tamaño publicado | Tratamiento |
|---:|---|---|---:|---|
| 1 | [SFT-General-Spanish-50K](https://huggingface.co/datasets/OysterCoreAI/SFT-General-Spanish-50K) | CC BY 4.0 | 50.413 | Conversación/instrucción; auditoría de calidad sintética |
| 2 | [mPersonas](https://huggingface.co/datasets/BSC-LT/m-personas) | Apache 2.0 | 510k multilingües | Filtrar español y preservar turnos/persona |
| 3 | [Personas_Spanish-40k](https://huggingface.co/datasets/Kukedlc/Personas_Spanish-40k) | Apache 2.0 | 37.476 | Perfiles; sesgo sectorial cultural/teatral |
| 4 | [StrataSynth Personas Spain](https://huggingface.co/datasets/StrataSynth/stratasynth-personas-spain) | CC BY 4.0 | 1.250 | Perfiles sintéticos y estilo; excluir atributos psicométricos sensibles del conditioning |
| 5 | [Big Five Personality Traits](https://huggingface.co/datasets/ola-owo/big-five-personality-traits) | CC BY 4.0 | Split español por contar | Descripciones de rasgos; revisar procedencia y duplicados |
| 6 | [LingComp_QA](https://huggingface.co/datasets/somosnlp/LingComp_QA) | Apache 2.0 | 803 | QA educativa; 741 ya entraron en v3.1 |
| 7 | [Lenguaje claro](https://huggingface.co/datasets/somosnlp/lenguaje-claro-dataset) | Apache 2.0 | 4.094 | Reformulación administrativa; 55 ya entraron en v3.1 |
| 8 | [Instruct Legal Refugiados](https://huggingface.co/datasets/somosnlp/instruct-legal-refugiados-es) | Apache 2.0 | 9.430 | QA contextual con jurisdicción y fecha |
| 9 | [GovEasy](https://data.goveasy.eu/es) | CC BY 4.0 | 438 trámites | Procedimientos administrativos |
| 10 | [ESQAD](https://zenodo.org/records/16351733) | CC BY 4.0 | ZIP 676,1 MB | QA educativa; separar evaluación y piloto calificado |
| 11 | [TDX Thesis Spanish Corpus](https://zenodo.org/records/7313149) | CC BY 4.0 | 248,7 M tokens | Preentrenamiento académico |
| 12 | [PleIAs BDH-Books](https://huggingface.co/datasets/PleIAs/BDH-Books) | Dominio público por obra | ~139.932 títulos | OCR, derechos por obra y deduplicación |
| 13 | [Spanish-PD-Books](https://huggingface.co/datasets/PleIAs/Spanish-PD-Books) | Dominio público por obra | ~302.640 textos | Literatura; alto riesgo de solapamiento con BDH |
| 14 | [PleIAs Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) | Filtrado por licencia documental | >10B tokens españoles reportados | Solo español y documentos con derechos allowlisted |
| 15 | [MS MARCO ES 500k](https://huggingface.co/datasets/dariolopez/ms-marco-es-500k) | Apache 2.0 | 500.000 tríos | Recuperación/ranking; no aplanar a QA generativa |
| 16 | [Enfermedades Wiki marzo 2024](https://huggingface.co/datasets/Alvaro8gb/enfermedades-wiki-marzo-2024) | MIT | ~1,42 M tokens en inventario SMC | Texto biomédico; atribución y procedencia por artículo |
| 17 | [PharmaCoNER](https://huggingface.co/datasets/PlanTL-GOB-ES/pharmaconer) | CC BY 4.0 | 15.868 frases anotadas | Extracción de entidades; solo train |
| 18 | [MEDDOCAN](https://huggingface.co/datasets/bigbio/meddocan) | CC BY 4.0 | 364.462 tokens en inventario SMC | Desidentificación; solo train y revisión de privacidad |
| 19 | [DisTEMIST](https://huggingface.co/datasets/bigbio/distemist) | CC BY 4.0 | 550.203 tokens en inventario SMC | Menciones de enfermedades; solo train |
| 20 | [BARR2](https://huggingface.co/datasets/somosnlp/SMC) | CC BY 4.0 | 1.732.432 tokens en inventario SMC | Texto biomédico de fondo; confirmar origen canónico |

> Las cifras de inventarios agregados no son recuentos de filas comparables. Se recalcularán sobre los archivos descargados y las revisiones fijadas.

## Personalidad: no volver a meterla en cuarentena por una palabra clave

La v3.1 apartó 50.121 registros por coincidencias de metadatos de personalidad/identidad/conducta. Eso fue una decisión conservadora de cribado, no una evaluación de que fueran malos. La v3.2 debe hacer una auditoría semántica y separarlos en:

- **Perfil/persona:** biografía ficticia, intereses, objetivos y contexto.
- **Estilo lingüístico:** longitud, registro, humor, franqueza, formalidad y variación dialectal.
- **Conducta conversacional:** admitir incertidumbre, corregir errores, no adular, responder a premisas falsas, mantener consistencia.
- **Diálogo multi-turno:** conservar el historial y medir contradicciones de identidad.
- **Atributos sensibles o psicométricos:** no usarlos como etiquetas de salud mental ni como predictores de conducta real. Mantenerlos fuera del conditioning cuando no sean imprescindibles.

Los perfiles sintéticos no son estadísticas poblacionales ni personas reales. Se utilizarán para entrenar consistencia narrativa y variación de voz, no para enseñar estereotipos.

## Reglas de ingestión

1. Fijar SHA/revisión inmutable y guardar licencia, autoría, URL y fecha de consulta.
2. Filtrar español de forma explícita; no confiar solo en el tag de idioma.
3. Descargar y revisar el esquema real y los splits antes de convertir.
4. Mantener cinco tipos de registro: `pretraining_document`, `sft_chat`, `persona_profile`, `retrieval_pairwise` y `supervised_extraction`.
5. No convertir texto largo en preguntas/respuestas inventadas. Segmentar documentos en continuaciones naturales y conservar sus metadatos.
6. No mezclar tareas NER, ranking y QA como si fueran ejemplos equivalentes. Mantener una tarea explícita y su objetivo.
7. No entrenar con test/validation ni con holdouts del proyecto.
8. Deduplicar exacto y aproximado contra v3.1 y entre fuentes; medir solapamiento de libros, Common Corpus y colecciones biomédicas.
9. Revisar privacidad y calidad; muestrear 100 ejemplos por fuente como mínimo, y todos los registros en fuentes pequeñas.
10. Construir cache con el tokenizer congelado de 4.096 IDs y contexto 128. No truncar silenciosamente: documentar exclusiones y diseñar segmentación cuando proceda.

## Gate

Este registro no autoriza entrenamiento. Antes de construir el artefacto v3.2 hay que resolver los campos de licencia y revisión por documento, fijar revisiones, contar filas/tokens españoles reales, revisar muestras y comprobar que el cache coincide con el JSONL. LUMEN y Transformer recibirán exactamente el mismo artefacto y los mismos hashes.

La lista ejecutable y sus reglas se mantienen en `data/corpus_v3_2_sources.json`.
