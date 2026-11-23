# Lumen-Ronin Spanish Corpus v3.1 — plan de admisión y ampliación

**Rama:** `corpus/lumen-ronin-spanish-v3-1`  
**Base inmutable:** Lumen-Ronin Spanish Corpus v3.0.0, train SHA-256 `87595c3217c3b8166eadaf24309cfe4753a6da4e7c1d7c91bfd5322cae114811`  
**Estado:** preparación de v3.1; todavía no hay artefacto ni autorización de entrenamiento.

## Objetivo

Crear v3.1 como una nueva versión reproducible, sin modificar v3.0.0. La ampliación cubrirá humanidades y ciencias humanas —filosofía, historia global, patrimonio, derecho, ética y ciencias sociales— además de razonamiento matemático, lógica y medicina. Solo se incorporarán registros de entrenamiento cuya licencia, split, procedencia y calidad hayan sido verificados.

La versión final conservará byte por byte los archivos de validation, holdout y tokenizer de v2/v3. No se introducirán datos de evaluación externos en train sin confirmar primero que no proceden de sus splits de evaluación.

## Auditoría previa de v3.0.0: resultados y acciones

Informe de auditoría: [run 45142717502](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45142717502), artefacto [14805384712](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45142717502/artifacts/14805384712).

- 9.933.007 filas train; SHA-256 confirmado.
- Validation (1.413), holdout MC (2.431) y tokenizer (vocabulario 4.096) idénticos a v2.
- Cero duplicados exactos de pares prompt/completion frente a v2; cero duplicados exactos internos entre nuevas incorporaciones; cero coincidencias exactas con validation/holdout.
- 63 completions repetidas frente a v2 (longitud >=40): revisar y retirar solo las redundancias no justificadas.
- 19.740 near-duplicates detectados por Jaccard de trigramas >=0,88 y ratio de longitud >=0,80. El detector no es exhaustivo: no tratar el resultado como 19.740 duplicados confirmados automáticamente. Revisar por fuente, eliminar solapamientos textuales claros y documentar excepciones legítimas.
- 157.874 consultas superaron el límite de candidatos: el área no examinada queda pendiente; ampliar la búsqueda con indexación/particionado para no darla por aprobada.
- 41 alertas de posible marcador sensible: 13 en literatura y 28 en M-Personas. Son alertas heurísticas, no incidentes confirmados; revisar el texto completo y excluir lo que resulte impropio.
- Literatura: restaurar la relación entre cada fragmento, documento de origen, revisión, licencia y hash del texto original. La v3.0.0 solo conserva índices de segmento.
- M-Personas: revisar muestras y sesgos demográficos/ocupacionales; conservar la licencia y revisión exactas.
- Personalidad y currículos sintéticos: conservar atribución y linaje; no presentarlos como hechos independientes ni corpus empírico.
- ALIA self-identity heredado: verificar la licencia upstream y mantener el aviso de adaptación antes de reutilizar.

**Decisión:** v3.0.0 permanece congelado y con `training_authorized=false`. v3.1 reconstruirá train desde la base congelada, aplicará remediación y volverá a ejecutar controles globales.

## Catálogo de nuevas fuentes

### Admisión técnica prioritaria

| Fuente | Tema y formato de origen | Licencia declarada | Tratamiento v3.1 | Estado |
|---|---|---|---|---|
| [MGSM8KInstruct Parallel](https://huggingface.co/datasets/Mathoctopus/GSM8KInstruct_Parallel) | Matemáticas multilingües; `prompt`, `chosen`, `reject`; 73.6k filas globales | Apache-2.0 | Extraer solo ejemplos españoles de train; `prompt` → `prompt`, `chosen` → `completion`; descartar rechazos y splits de evaluación | Candidato prioritario; comprobar filas y procedencia por idioma |
| [SLR-Bench-Spanish](https://huggingface.co/datasets/AIML-TUDA/SLR-Bench-Spanish) | Razonamiento lógico inductivo; 18.053 filas train en config v1-All; `prompt`, `ground-truth rule`, programa de validación y metadatos curriculares | CC BY 4.0 | Convertir prompt/regla a SFT; guardar programa y nivel en metadatos; train solamente; mantener validation/test fuera | Candidato prioritario; paper arXiv:2506.15787 |
| [CasiMedicos / Antidote](https://github.com/ixa-ehu/antidote-casimedicos) | Medicina clínica MIR y argumentación; 1.561 preguntas comentadas, subconjunto clínico de 622, y CasiMedicos-Arg multilingüe | CC BY 4.0 | Importar solo filas españolas de train; pregunta/caso como prompt y explicación médica como completion; preservar anotaciones argumentativas como metadatos | Candidato prioritario; revisión de datos y licencia por archivo |

### Humanidades y ciencias humanas

| Fuente | Tema y escala publicada | Licencia declarada | Tratamiento v3.1 | Estado |
|---|---|---|---|---|
| [ALIA Spanish Cultural & Heritage](https://huggingface.co/datasets/SINAI/ALIA-es-cultural-heritage) | Patrimonio, historia, archivos, cultura material e inmaterial; 236.314 registros, 100 fuentes, 939M tokens declarados | CC BY-SA 4.0 | Selección estratificada y limitada por fuente; raw-text pretraining; preservar source_id y licencia de cada fuente; estudiar obligaciones ShareAlike del conjunto derivado | Candidato condicionado; no importar el corpus completo |
| [OpenAlex Philosophy Corpus](https://huggingface.co/datasets/CristianPelayo/openalex-philosophy) | Metadatos y abstracts académicos de filosofía; snapshot CC0, corpus multilingüe | CC0 para metadatos OpenAlex; verificar por separado cualquier texto/abstract reconstruido | Admitir solo registros españoles de alta confianza y abstracts presentes en los metadatos; no descargar ni copiar artículos protegidos | Candidato condicionado; deduplicar por OpenAlex work ID |
| [Corpus RLD](https://zenodo.org/records/17931030) | 1.940 sentencias del Supremo y Constitucional (2012–2018), lenguaje y autoridad lingüística en jurisprudencia | CC BY 4.0 | Importar texto legal y metadatos de procedencia; separar sentencias y anotaciones; revisar datos personales y restricciones de documentos fuente | Candidato prioritario; dataset publicado por UCM |
| [HiST-LLM](https://zenodo.org/records/14671248) / Seshat | Historia global; 36.000 puntos de datos, 600 sociedades y más de 2.700 referencias | Licencia exacta del snapshot por verificar; Seshat general declara CC BY-NC-SA | No importar hasta resolver licencia del artefacto concreto y derechos de los datos subyacentes; evitar cualquier split de benchmark | En espera de derechos |
| [UniMoral](https://aclanthology.org/2025.acl-long.294/) | Dilemas morales y razonamiento intercultural en seis idiomas, incluido español | Verificar licencia del dataset y split antes de usar | Seleccionar únicamente train español con anotaciones y contexto cultural; no convertir perfiles de anotadores en etiquetas identitarias del modelo | Candidato condicionado; paper ACL 2025 |
| [MCSQ](https://www.upf.edu/web/mcsq) | Ciencias sociales: encuestas ESS, EVS, SHARE y WageIndicator; 306 cuestionarios, 766k segmentos, traducción española y catalana | CC BY-NC-SA 4.0 | No incorporar al corpus de entrenamiento compartido mientras no se resuelva la restricción no comercial y ShareAlike | Excluido por ahora |
| [IMPACT-es](https://digitisation.eu/knowledge/language-resources/impact-es-2/) | Historia del español: más de 100 libros, ~8M palabras y léxico diacrónico | CC BY-NC-SA | No incorporar hasta definir compatibilidad de licencia; excelente candidato para una partición separada de investigación no comercial | Excluido por ahora |
| [Seshat Global History Databank](https://github.com/datasets/seshat) | Historia comparada, organización política y social, religiones y crisis | CC BY-NC-SA según aviso de los responsables | No importar por restricción NC y ambigüedades en snapshots/derivados | Excluido por ahora |

### Fuentes que no deben confundirse con corpus admitidos

- [ALIA Cultural Heritage Synthetic Instructions](https://huggingface.co/datasets/SINAI/ALIA-es-cultural-heritage-synthetic-instructions): 748.480 instrucciones sintéticas. Mantener fuera de la admisión inicial hasta revisar generador, trazabilidad de cada respuesta, contaminación con la base y licencia derivada. No tratarlas como anotaciones humanas.
- [Filosofía clásica en español](https://huggingface.co/datasets/ecastera/filosofia-es): 7,13k filas; la ficha pública no basta para establecer licencia y trazabilidad obra por obra. No admitir hasta revisar autores, ediciones, transcripciones y derechos.
- [LEM-Ethics](https://huggingface.co/datasets/lthn/LEM-Ethics): proyecto en desarrollo con datos generados por modelos; no tiene todavía evidencia independiente suficiente para entrar en esta versión.
- El benchmark HiST-LLM y los splits de evaluación de SLR/MGSM/CasiMedicos no se incorporarán a train.

## Esquema de conversión

Todos los ejemplos se normalizarán al formato JSONL del proyecto:

- `prompt`: entrada completa en español, incluyendo caso/contexto/instrucción cuando corresponda.
- `completion`: respuesta de referencia, explicación o fragmento documental.
- `task`: tarea explícita (p. ej. `math_reasoning`, `inductive_logic`, `clinical_reasoning`, `humanities_raw_text`, `legal_analysis`).
- `language`: `es`.
- `source`, `source_id`, `source_revision`, `license`: procedencia obligatoria.
- `source_document_id`, `source_document_sha256`, `source_url`: obligatorios para fragmentos de documentos.
- Metadatos de dominio (nivel curricular, categoría temática, tipo de documento, anotación argumentativa) se conservan sin concatenarlos innecesariamente a la respuesta.

No se inventarán pares pregunta/respuesta para corpus de texto continuo. Se usarán `prompt=""` y `task="raw_text_pretraining"` solo para fuentes cuyo uso como pretraining esté permitido.

## Política de muestreo y equilibrio

- La base v3.0.0 se conserva como punto de partida, pero se corrigen las filas que no superen la auditoría.
- No dejar que ALIA Cultural Heritage, literatura o un único dominio dominen la ampliación por volumen.
- Muestreo estratificado por fuente, tema, periodo histórico, región y dificultad cuando esos metadatos existan.
- Priorizar ejemplos españoles. Mantener el catalán solo en una partición identificada y autorizada, no mezclarlo silenciosamente con `language=es`.
- Registrar filas antes/después, bytes, tokens BPE reales con tokenizer v2, longitud, distribución por fuente/tarea y SHA-256.
- No alterar tokenizer, validation ni holdout.

## Gates para declarar v3.1 utilizable

1. Revisar manualmente las 41 alertas y las 63 completions repetidas de v3.0.0.
2. Resolver la muestra de near-duplicates; reejecutar búsqueda con índice particionado y sin overflow de candidatos.
3. Restaurar trazabilidad documental de literatura y revisar sesgos de M-Personas.
4. Verificar licencia, revisión, split, atribución y derechos de cada fuente nueva.
5. Revisar una muestra por fuente (incluidos extremos de longitud y temas sensibles); filtrar respuestas vacías, truncadas, alucinadas o mal alineadas.
6. Deduplicar exacta y aproximadamente contra train v2/v3 y entre fuentes nuevas.
7. Confirmar cero contaminación exacta con validation/holdout; conservar esos archivos byte-identical.
8. Medir token count completo, pesos por fuente y tamaño comprimido/extraído.
9. Publicar manifiesto reproducible con hashes y `training_authorized=true` solo si todos los gates aplicables pasan.

**Estado actual:** catálogo y plan documentados; v3.1 no está construido todavía. No ejecutar entrenamiento con esta rama hasta que exista artefacto final y manifiesto de autorización.
