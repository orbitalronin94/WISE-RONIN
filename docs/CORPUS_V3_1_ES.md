# Corpus español v3.1 — auditoría y admisión

**Estado: candidato en construcción. No autorizado para entrenamiento.**

La v3.1 se prepara en una rama aislada para el siguiente ciclo posterior a 400k. No modifica el corpus v2 ni los artefactos usados por la campaña emparejada en curso. LUMEN y Transformer recibirán exactamente el mismo fichero, tokenizer y token cache cuando el gate de admisión se cierre.

## 1. Auditoría del conjunto de personalidad ya usado

El punto de partida es el artefacto congelado ronin-v2-plus-personality-250k-45149684913, cuyo JSONL de entrenamiento contiene 250.000 filas. La auditoría lee ese fichero original, no una reconstrucción aproximada.

El workflow genera:

- recuentos por fuente, tarea y esquema de campos;
- filas con metadatos que sugieren pertenencia a identidad/personality/conducta, como ayuda de localización, no como clasificación semántica;
- duplicados exactos normalizados, textos vacíos y distribución de longitudes;
- indicadores de posibles emails, teléfonos y DNI;
- muestra determinista estratificada por fuente y una extracción de candidatos para revisión humana.

Los ficheros de revisión redactan patrones evidentes de email, URL, teléfono y DNI. La redacción es una salvaguarda auxiliar y no sustituye una auditoría de privacidad. Ningún ejemplo se elimina ni se reescribe automáticamente en el corpus original.

**La revisión manual debe clasificar cada ejemplo de personalidad** en: conservar, corregir, retirar o mover a evaluación. Debe comprobar identidad del asistente, límites y capacidades, incertidumbre, corrección de errores, resistencia a premisas falsas, estabilidad entre turnos, diversidad de formulaciones y ausencia de biografía ficticia. Las respuestas factuales requieren verificación contra fuentes, no solo fluidez.

## 2. Fuentes nuevas integradas en formato nativo

El builder incorpora dos conjuntos pequeños, en castellano y con licencia Apache-2.0 declarada en sus tarjetas. Ambos se convierten al esquema del proyecto (prompt, completion, task, language, source, source_id, license, source_revision).

| Fuente | Tamaño publicado | Contenido | Admisión inicial |
|---|---:|---|---|
| [SomosNLP LingComp_QA](https://huggingface.co/datasets/somosnlp/LingComp_QA) | 803 filas train | Preguntas y respuestas educativas de lingüística computacional | Incluido en candidato; deduplicación y muestreo humano pendientes |
| [SomosNLP lenguaje-claro-dataset](https://huggingface.co/datasets/somosnlp/lenguaje-claro-dataset) | 4.090 filas train | Reformulación de textos administrativos en castellano claro; es-ES | Incluido en candidato; control factual y muestreo humano pendientes |

Las revisiones exactas de Hugging Face se resuelven al iniciar la construcción, se fijan durante la descarga y se guardan en el manifiesto. El builder comprueba que la licencia declarada siga siendo Apache-2.0, valida los campos, filtra idioma cuando existe etiqueta, elimina duplicados exactos respecto al corpus base y entre fuentes, y conserva la procedencia por fila.

**No se afirma que todas las respuestas sean verdad por el mero hecho de pertenecer a un dataset publicado.** La incorporación inicial es una admisión técnica al candidato; la admisión definitiva requiere spot-check de respuestas y auditoría de near-duplicates.

## 3. Fuentes examinadas y no incorporadas

| Fuente | Motivo |
|---|---|
| [IIC AQuAS](https://huggingface.co/datasets/IIC/AQuAS) y [RagQuAS](https://huggingface.co/datasets/IIC/RagQuAS) | CC BY-NC-SA 4.0: restricción no comercial incompatible con la política de admisión permisiva del proyecto |
| [Projecte Aina MentorES](https://huggingface.co/datasets/projecte-aina/MentorES) | La tarjeta muestra CC BY-SA 4.0; requiere resolver la condición ShareAlike antes de mezclarlo |
| [BSC-LT IFEval_es](https://huggingface.co/datasets/BSC-LT/IFEval_es) | Conjunto de evaluación; su tarjeta indica expresamente que no debe utilizarse para entrenar |
| [BSC-LT SIQA_es](https://huggingface.co/datasets/BSC-LT/SIQA_es) | Holdout protegido del proyecto; nunca se reintroduce en entrenamiento |
| [BSC-LT openbookqa-es](https://huggingface.co/datasets/BSC-LT/openbookqa-es) | Holdout protegido del proyecto; nunca se reintroduce en entrenamiento |
| [MS MARCO ES 500k](https://huggingface.co/datasets/dariolopez/ms-marco-es-500k) | Dataset de ranking query/positive/negative, no pares generativos prompt/completion; no convertir negativos en respuestas ni degradar la tarea para aparentar compatibilidad |
| [ESQAD](https://zenodo.org/records/16351733) | CC BY 4.0 y prometedor: EVAU manual y un piloto de 923 pares valorados por docentes/estudiantes. Se reserva para una segunda admisión tras inspeccionar estructura, puntuaciones y separación entre evaluación y entrenamiento; no se ingiere el ZIP completo a ciegas |

La licencia de una tarjeta es una señal de admisión, no una auditoría legal de todos los textos subyacentes. La procedencia y las condiciones de cada fuente se preservan.

## 4. Construcción y contrato técnico

El workflow descarga el artefacto base de 250k, conserva todas sus filas y sus tokens ya calculados, añade únicamente las filas nuevas y tokeniza solo esas adiciones con el tokenizer BPE congelado de 4.096 IDs.

Produce:

- spanish-v3-1-candidate.jsonl;
- spanish-v3-1-token-cache.pt en formato ronin_token_cache_v1;
- copia exacta del tokenizer;
- copia byte por byte del holdout y, si está presente, de validation;
- manifiesto con revisiones, licencias, recuentos, descartes y SHA-256;
- muestra de las filas nuevas para revisión humana;
- informe del dataset de personalidad original.

El cache se acepta únicamente si sus hashes, longitud máxima, número de filas y tokenizer coinciden con el manifiesto. El builder no reentrena tokenizer, no cambia el contexto 128 y no altera los splits protegidos.

## 5. Gate para autorizar entrenamiento

El manifiesto del artefacto mantiene training_authorized=false. Solo se cambiará tras completar y documentar:

1. revisión humana del subconjunto original de personalidad, con decisiones por fila;
2. muestreo factual de las fuentes añadidas, especialmente lenguaje claro y QA técnico;
3. near-duplicate detection entre fuentes y corpus base;
4. revisión de privacidad y marcadores sensibles;
5. verificación byte a byte de validation y holdout;
6. verificación de hashes de corpus, tokenizer y token cache;
7. aprobación de un único artefacto compartido por LUMEN y Transformer.

La v3.1 no se mezcla en la campaña que termina en 400k. El candidato queda aislado hasta superar todos los controles.
