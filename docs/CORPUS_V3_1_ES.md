# Corpus español v3.1 — admisión conservadora

**Estado: candidato depurado; todavía no autorizado para entrenamiento.**

La v3.1 se construye en una rama aislada para el ciclo posterior a 400k. No modifica los datos de la campaña emparejada en curso. LUMEN y Transformer deberán recibir exactamente el mismo JSONL, tokenizer, cache y evaluación protegida.

## 1. Auditoría del corpus original

Se auditó directamente el artefacto congelado `ronin-v2-plus-personality-250k-45149684913`:

- 250.000 filas leídas.
- 0 filas vacías.
- 250.000 textos normalizados únicos; 0 duplicados exactos excedentes.
- 50.121 filas localizadas por metadatos de personalidad/identidad/conducta: 50.000 del conjunto `verifiable_personality` y 121 de otras tareas.
- 26 filas con patrón de email y 9.121 con secuencias numéricas parecidas a teléfonos. Son indicadores conservadores, no una confirmación de PII; puede haber falsos positivos.
- La auditoría produjo muestra estratificada y extracción de revisión. Los metadatos de las exportaciones se redactan recursivamente.

La coincidencia de palabras clave no certifica calidad semántica. Por eso **no se incorporan automáticamente las 50.121 filas de personalidad/identidad**. Quedan fuera del entrenamiento v3.1 y en cuarentena lógica para revisión posterior. También se excluyen del candidato las filas con marcadores de posible dato personal. El artefacto original permanece intacto y recuperable.

### Resultado de la primera construcción conservadora validada

La ejecución [45348291485](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45348291485) superó los pasos de construcción, copia del holdout y validación del token cache:

- 196.992 filas base retenidas tras filtros; 53.008 en cuarentena.
- 796 filas nuevas admitidas: 741 de LingComp_QA y 55 de lenguaje claro.
- 197.788 filas totales en el candidato.
- Tokenizer congelado: SHA-256 `d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe`.
- JSONL candidato: SHA-256 `93e8c997535ca2bdc5215937e3ada6bc21e39a89d3131daa71c93b605582cd13`.

La cifra de cuarentena no es la suma simple de los contadores de motivos porque algunas filas presentan más de un marcador. Las repeticiones de prompt del corpus base se conservan: en los fragmentos de preentrenamiento es normal que un mismo prefijo tenga continuaciones distintas. La comprobación de respuestas conflictivas se aplica solo a las nuevas fuentes QA.

El holdout congelado se copió y comparó byte a byte; el workflow también validó que el número de filas del cache coincide con el JSONL, que el hash de datos y tokenizer coincide con el manifiesto y que todas las secuencias cacheadas respetan el contexto 128.

## 2. Fuentes nuevas admitidas técnicamente

Se admiten solo fuentes en castellano con licencia Apache-2.0 declarada y estructura generativa directa:

| Fuente | Contenido | Admisión |
|---|---|---|
| [SomosNLP LingComp_QA](https://huggingface.co/datasets/somosnlp/LingComp_QA) | QA educativa de lingüística computacional | Admitida al candidato tras validación de campos, licencia, duplicados y longitud |
| [SomosNLP lenguaje-claro-dataset](https://huggingface.co/datasets/somosnlp/lenguaje-claro-dataset) | Reformulación de textos administrativos en castellano claro | Admitida al candidato tras validación de idioma, licencia, duplicados y longitud |

Las revisiones exactas de Hugging Face se fijan en el manifiesto. El constructor no confía en el número publicado: registra filas leídas, añadidas y descartadas por fuente en cada ejecución.

Criterios adicionales:
- prompt y completion deben ser cadenas reales, no valores nulos convertidos a texto;
- se descartan campos vacíos, demasiado cortos, pregunta y respuesta idénticas, marcadores de posible dato personal y conflictos de respuestas para el mismo prompt;
- no se truncan silenciosamente ejemplos que excedan el contexto 128: se excluyen;
- se deduplica contra el corpus retenido y entre las fuentes;
- se conserva `source_id`, licencia y SHA de revisión por fila.

La admisión técnica no equivale a certificar que cada respuesta sea factual. Se adjunta una muestra por fuente para spot-check humano.

## 3. Fuentes excluidas o reservadas

| Fuente | Decisión |
|---|---|
| IIC AQuAS / RagQuAS | Excluidas: CC BY-NC-SA 4.0, restricción no comercial |
| Projecte Aina MentorES | Cuarentena: condición ShareAlike y discrepancia de licencia por resolver |
| BSC-LT IFEval_es | Solo evaluación; no entrenar |
| BSC-LT SIQA_es y openbookqa-es | Evaluación/holdout protegido; nunca reintroducir en entrenamiento |
| MS MARCO ES 500k | Excluido: ranking query/positive/negative, no SFT generativo directo |
| ESQAD | Reservado: aunque declara CC BY 4.0, requiere inspección del piloto docente/estudiantil y separación de evaluación antes de admitirlo |

## 4. Construcción y contrato técnico

El builder:
1. valida el hash del JSONL base, tokenizer y token cache original;
2. retiene las filas base que superan los filtros conservadores y sus token IDs originales;
3. deja fuera las filas en cuarentena sin alterar el artefacto fuente;
4. añade únicamente nuevas filas admitidas de las dos fuentes Apache-2.0;
5. tokeniza solo las nuevas filas con el tokenizer BPE congelado de 4.096 IDs;
6. genera un cache `ronin_token_cache_v1` ligado a SHA-256 del JSONL, tokenizer, longitud máxima y número de filas;
7. copia el tokenizer y el holdout congelado, y verifica que el holdout copiado sea idéntico.

El manifiesto registra el número exacto de filas retenidas, añadidas y puestas en cuarentena, motivos de exclusión, revisiones de origen y hashes.

## 5. Gate de autorización

`training_authorized=false` se mantiene hasta:
- revisar semánticamente el conjunto de personalidad en cuarentena, fila por fila o mediante una política de muestreo con criterios explícitos;
- inspeccionar la muestra estratificada de las dos fuentes nuevas y verificar respuestas;
- revisar los falsos positivos de marcadores numéricos antes de cualquier re-admisión;
- comprobar near-duplicates y conflictos de respuesta;
- verificar byte a byte tokenizer, validation y holdout;
- validar hashes y correspondencia JSONL/cache;
- aprobar un único artefacto compartido por LUMEN y Transformer.

No se lanza entrenamiento con esta rama ni se mezcla v3.1 en la campaña que termina en 400k. El artefacto es un candidato conservador y reproducible, no una afirmación de calidad humana completa.
