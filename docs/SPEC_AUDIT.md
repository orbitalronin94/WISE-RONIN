# Auditoría de cobertura — RONIN-LUMEN V1

**Objeto:** contrastar el documento maestro de 16 secciones con los archivos de la PR #13 y separar requisitos incorporados, pruebas reales y decisiones todavía abiertas.

**Estado:** especificación transcrita; especificación técnica todavía no congelable al 100% por contradicciones internas y decisiones sin resolver. Esta auditoría no cambia el documento maestro ni autoriza implementar órganos.

## 1. Cobertura de las 16 secciones

| Sección | Requisito | Estado documental | Evidencia / trabajo pendiente |
|---|---|---|---|
| 1. Contexto y filosofía | Identidad, historia V2, voz, confirmado/rumor, KOAN y holdout | Incorporado | README y documento maestro; la proporción lingüística de las fuentes requiere decisión |
| 2. Restricciones duras | Tamaño, CPU, tiempos, vocabulario, datos, holdout y reproducibilidad | Incorporado | Hay una contradicción numérica entre parámetros y tamaño del fichero; ver §3 |
| 3. Principios | Sorpresa, homeostasis, sparsity, escalas de memoria, rutas, energía y sueño | Incorporado | No hay tests de arquitectura porque M2 no está autorizado |
| 4. Arquitectura | Ocho órganos, fórmulas, dimensiones e hiperparámetros | Incorporado | Varias definiciones matemáticas y de estado requieren precisión; ver §3 |
| 5. Estructura | Módulos, separación de responsabilidades y decisiones V2→V1 | Parcial | Estructura M1 existe; módulos de datos son placeholders que ahora se están preparando; conflicto GELU/SwiGLU |
| 6. Procedimiento | Inventario, archivo, M1, órganos por separado, integración y datos | Parcial | V2 archivada; inventario aún sin líneas/imports completos; smoke local hecho; Actions de M1 no acreditado |
| 7. Dataset | Mezcla 25k, fuentes, filtros, formato, licencias y holdout | Parcial | Fuentes y cuotas preparadas para staging; falta ejecución real, 1k código con licencias y 5k corpus propio |
| 8. Entrenamiento | Bloques 500, checkpoints, curriculum, ORPO, logs y comandos | Incorporado como requisito | Workflow de entrenamiento sigue placeholder; no implementar hasta autorización |
| 9. GitHub Actions | Runner CPU, 5h30, checkpoint y artifacts | Incorporado como requisito | El workflow de preparación de datos se añadió; aún no se ha ejecutado |
| 10. Evaluación | Holdout, sampling, seis métricas y detector de degradación | Incorporado como requisito | Holdout conservado; auditoría de contaminación pendiente; 55 prompts únicos en 500 filas |
| 11. Riesgos | Homeostasis, Top-K, sueño, KAN, datos, holdout y checkpoint | Incorporado | Requiere pruebas cuando se autorice implementación |
| 12. Milestones | M1–M5 y criterios | Incorporado | M1 local parcial; no avanzar a órganos |
| 13. Checklist | Criterios para declarar listo | Incorporado | No debe marcarse completo mientras falten dataset, auditoría y CI |
| 14. Estilo | Python, comentarios, dependencias, logging y config | Incorporado | Verificar al implementar |
| 15. Prohibiciones | No atención, no RoPE, no GPU, no holdout train, etc. | Incorporado | La política se mantiene |
| 16. Primera acción | Inventario, archivo, M1 y parar | Incorporado | La autorización actual amplía la tarea a preparar especificación y fuentes; no autoriza órganos ni entrenamiento |

## 2. Estado comprobable de los artefactos

- `DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md` y `SPEC_RONIN_LUMEN_V1.md` son actualmente el mismo contenido y comparten SHA de blob. No hay divergencia entre copias.
- El documento contiene las 16 secciones y el anexo de trazabilidad.
- `README.md` reivindica la combinación de diseño como hipótesis experimental, no como rendimiento probado ni invención individual de todos los componentes.
- `archive/v2/INVENTARIO.md` registra 54 archivos, tamaños y blob SHAs; ya incorpora recuentos de líneas e imports estáticos de los 12 scripts Python y 14 workflows. Los seis JSONL grandes siguen pendientes de materialización completa; el NPZ es binario.
- El holdout preservado tiene 500 registros y 55 prompts distintos. No se ha certificado ausencia de contaminación.
- `data/train_25k.jsonl` no existe todavía.
- `scripts/download_sources.py` ya prepara muestreo determinista, deduplicación exacta entre fuentes, hash de cada fichero, revisión inmutable resuelta para cada dataset y manifiesto de procedencia. El filtro de prefijo se aplica por fuente durante el staging; el builder final tendrá que validar el conjunto global.
- `.github/workflows/prepare-data.yml` prepara un artifact de candidatos, no un dataset final.
- No se ha descargado ni validado aún ningún corpus de fuentes en un runner.

## 3. Decisiones técnicas que no deben resolverse en silencio

### 3.1 Tamaño frente a parámetros

La especificación exige 3–4M parámetros objetivo y 6M máximo, además de 20–30 MB en float16/int8. Solo los pesos de 3–4M parámetros ocupan aproximadamente 6–8 MB en fp16 o 3–4 MB en int8, antes de metadatos y tokenizer. Hay que definir qué incluye el objetivo de 20–30 MB: pesos, tokenizer, runtime, estado persistente u otros artefactos.

### 3.2 GELU frente a SwiGLU

- §4.7 define MLP 256→512→256 con GELU.
- §5.3 ordena reescribir SwiGLU y mantenerlo solo en la MLP Head.
  
No son la misma implementación. Falta elegir una.

### 3.3 Sueño frente a poda del vocabulario

- La poda del vocabulario está fijada cada 1.000 pasos.
- El diagrama de arquitectura dice “cada 500 pasos: Sueño + Poda”.
  
Hay que aclarar si el sueño ocurre cada 500 y la poda cada 1.000, o si ambos ocurren juntos.

### 3.4 Signo de la homeostasis

La fórmula prescrita es `b_i += eta * (m_i - target)`. Si `b_i` se suma como bias convencional a la activación, una neurona sobreactiva recibe más bias y puede activarse todavía más. No se cambia la fórmula sin autorización; hay que especificar si el bias se resta, si se aplica con signo contrario o si la convención de activación es distinta.

### 3.5 Estado Delta

- No está claro si existe una matriz S independiente por cada una de las cuatro capas Delta o una única matriz compartida.
- `decay(s)=sigmoid(a*s+b)` no define inicialización de a/b ni qué comportamiento inicial se pretende.
- `lr(s)=clamp(s,0.1,1.0)` puede saturarse con normas L2 grandes; falta definir escala/normalización.
- No se especifica si S se guarda en fp16 o fp32 durante actualización ni cómo se estabiliza la multiplicación recurrente.

### 3.6 Memoria

- La MLP 256→64→256 no especifica cómo se indexan, recuperan y actualizan los recuerdos.
- La matriz lenta 128×256 no especifica función de similitud, colisiones, reemplazo ni cómo se identifica que un patrón k reapareció tres veces.
- Falta definir si el estado elástico y el delta sobreviven a los límites de secuencia y cómo se serializan en checkpoints.

### 3.7 Top-K y KAN

- “Capas bajas/altas” no está ligado inequívocamente a las cuatro capas Delta, al MLP o a ambos.
- La definición 256→32 splines con 8 puntos de control no determina la parametrización exacta ni el recuento de parámetros reproducible.
- El router define p_op/p_lang, pero falta especificar cómo se combinan logits de cabezas con salidas de forma distinta.

### 3.8 ORPO y datos de preferencias

ORPO se exige en fase 3, pero la receta de datos no define pares preferido/rechazado ni el procedimiento para crearlos. Además, ORPO no aparece entre las dependencias permitidas. Hay que especificar implementación y corpus antes de fase 3.

### 3.9 Dataset: tabla de cuotas frente a 80/20

La tabla §7.2 suma 25.000 ejemplos externos. El texto inmediatamente posterior exige 80% de fuentes probadas y 20% de corpus propio. El muestreador actual escala las cuotas externas a 80% (20.000 filas) y reserva 5.000 para corpus propio. La tabla y esta interpretación deben reconciliarse formalmente antes de congelar el dataset.

### 3.10 Confirmado, rumor y calidad de fuentes

El campo booleano `confirmed` no permite representar claramente “fuente con licencia/procedencia conocida” frente a “afirmación factual verificada”. Un dataset educativo o una respuesta generada no convierte automáticamente cada afirmación en hecho confirmado. La etiqueta actual del muestreador es conservadora y marca como no confirmadas las fuentes no verificadas individualmente.

### 3.11 Idioma

Las fuentes públicas de la receta son principalmente inglesas. No existe cuota explícita de español ni dataset público español designado. El 20% propio no puede asumirse suficiente sin medir idioma, longitud y distribución de tipos.

### 3.12 The Stack / Python-Edu

Python-Edu deriva de The Stack V2. La etiqueta ODC-By del corpus no elimina las licencias originales de cada fichero, atribución ni términos de retirada. Se bloquean 1.000 filas de código hasta poder asociar contenido, repositorio, ruta y licencia permitida de forma verificable.

### 3.13 Holdout y métricas

- 500 filas con solo 55 prompts únicos requieren análisis de variantes y duplicados antes de certificar diversidad.
- El detector habla de 100 respuestas, mientras el umbral KOAN se formula sobre 500; debe definirse el denominador.
- La perplexity de V2 y LUMEN no es directamente comparable si usan tokenizadores distintos. Se necesita métrica común o evaluación adicional.
- La auditoría de overlap train/holdout debe ser bloqueante y no puede “reparar” el holdout.

### 3.14 Checkpoint y reproducibilidad

La lista actual incluye pesos, optimizer, scheduler, paso, seed, config, homeostasis y memoria lenta. Para reanudación exacta faltan por definir RNG states, estado del sampler/stream, versión y hash del dataset, tokenizer/vocabulario vivo y cualquier estado adaptativo persistente.

## 4. Bibliotecas: staging, no descarga completa

La especificación menciona colecciones de escala muy superior a la capacidad de un runner gratuito. No se intentará descargar FineWeb-Edu, DCLM o SmolLM-Corpus completos. El plan es streaming y muestreo de las cuotas requeridas.

Fuentes públicas habilitadas para staging: TinyStories, BabyLM 2026 Strict, FineWeb-Edu sample-10BT, DCLM-Baseline, Cosmopedia v2, Orca-Math y TinyGSM. Las licencias y condiciones están detalladas en `data/README.md`.

Pendientes/bloqueadas:
- Python-Edu/The Stack: auditoría por fichero.
- GSM8K: solo evaluación.
- Corpus original del propietario: 5.000 registros todavía no suministrados en el repo.
- Licencias share-alike/atribución: mantener notices y procedencia en el artifact y en cualquier redistribución.

El workflow de preparación se ha creado, pero aún no se ha ejecutado ni existe un artifact. Requiere una ejecución de Actions desde una rama donde GitHub tenga disponible el workflow. No se afirmará que los datasets están descargados hasta que exista un artifact con manifiesto, filas y hashes.

## 5. Gate de cierre

No se declara “spec completa y bibliotecas descargadas” hasta que:

1. Se decidan o se marquen explícitamente como no resueltas las contradicciones §3.
2. El inventario V2 incluya recuento de líneas e imports para los 54 archivos.
3. El workflow de preparación se ejecute en runner y genere el artifact con hashes.
4. El muestreo alcance exactamente las cuotas sin filtros incumplidos.
5. Se resuelva la cuota de 1.000 ejemplos de código.
6. Se incorpore el corpus propio de 5.000 registros o se cambie formalmente el reparto.
7. Se ejecute la auditoría de solapamiento contra el holdout sin modificarlo.
8. Se documente la mezcla lingüística y las atribuciones finales.

**No autoriza:** implementación de órganos, M2, entrenamiento, modificación del holdout ni merge de la PR.


## 6. Actual implementation update — 2026-11-21

Following the owner's explicit instruction to integrate the architecture, the branch now contains first-pass executable implementations in:
- `model/embedding.py`, `energy.py`, `delta_state.py`, `homeostasis.py`, `memory.py`, `heads.py`, and `lumen.py`;
- `train/loss.py`, `curriculum.py`, `dataset.py`, `sleep.py`, `quant.py`, and `train.py`;
- `scripts/train_tokenizer.py` and `data/build_dataset.py`.

The model has four recurrent Delta layers, tied input/output embeddings, one energy allocator, four homeostasis/Top-K stages, elastic/slow memory, routed KAN/MLP heads and a JEPA predictor. A first static count is **2,281,841 parameters**. This meets the hard 6M ceiling but not the 3–4M target band. The 20–30 MB model-storage requirement is also not demonstrated by this parameter count.

### Implementation status and explicit limitations

| Area | Implemented now | Not yet complete / validation gate |
|---|---|---|
| Embedding Vivo | Usage accounting; 1,000-forward cadence; up-to-5% dead-token candidate proposal | Does not rewrite tokenizer IDs or replace fragments; proposals are intentionally non-destructive until atomic tokenizer/embedding/optimizer remap exists |
| Energy | Softmax allocation to delta/memory/head; entropy metric | Need verify entropy behavior and budget allocation across training |
| Delta | Four per-layer 256×256 recurrent states; surprise threshold; learned decay; energy-scaled write; reset policy | Exact formula differs from hand sketch in normalization/clamping and initialization; stability and sequence semantics need tests |
| Homeostasis | EMA activation, subtractive adaptive threshold, Top-K | Uses negative-feedback sign to avoid the positive-feedback issue in the literal pseudocode; owner must confirm this interpretation |
| Memory | Elastic MLP; slow bank; similarity-based repeated-pattern consolidation | Slot collision/replacement, 512-token decay and exact 3-in-1000 window semantics are simplified or pending |
| Heads | Router and interpolated spline KAN + GELU MLP | KAN is a compact spline head, not a validated general arithmetic solver; router may collapse without supervised task labels |
| Sleep | 50 offline steps on slow-memory targets; only memory MLP optimized; reset fast states | Skips when no consolidated memories exist; needs non-degradation tests; sleep optimizer is recreated per sleep |
| JEPA | Predictor trained against mean embedding of next eight tokens | Needs numerical tests and ablation; target masking/padding behavior requires review |
| Repetition | Differentiable probability of a same-token triple | This is an auxiliary proxy, not the original argmax diagnostic; generation-time repetition still needs eval metrics |
| Curriculum | Explicit phase windows; phase 2 stratified 70/30 sampler; fail-closed missing labels | Phase 1/3 labels and data quality must be audited; ORPO is explicitly rejected until pairs/objective are implemented |
| Checkpoint | Model, optimizer, scheduler, global step, phase, config, corpus hash, Torch/Python/local sampler RNG | Delta is reset at every batch boundary; exact-resume equivalence test remains required; no multi-worker sampler state exists |
| Quantization | Fake-int8 activation simulation hooks from step 5,000 on KAN and memory | Not int8 weight quantization, serialization or size reduction |
| BPE | Corpus-only ByteLevel BPE training script with reserved PAD/UNK/BOS/EOS/SEP | Cannot run until final 25k dataset exists; verify vocab exactly 4096 |
| Data | Deterministic source staging, final assembly script, holdout exact-hash gate | Source staging run in progress; 1k code rows need per-file license audit; 5k owner-curated rows absent |
| Actions | CPU architecture/unit/smoke workflow and separate source-staging workflow | Latest runner jobs must pass; source workflow must produce and verify artifact |

### Important implementation corrections

- Delta state is a registered buffer and is reset at batch boundaries. Checkpoints therefore represent the declared reset policy, not a hidden live sequence state.
- Pruning is a proposal only. Zeroing rows while the fixed BPE tokenizer can still emit those IDs would silently corrupt the model, so destructive pruning is blocked pending atomic vocabulary remapping.
- The trainer refuses ORPO rather than pretending a `pass` is preference optimization.
- Fake-int8 is clearly named activation simulation and does not claim model compression.
- The loss includes a differentiable triple-repeat probability; the non-differentiable argmax statistic from the manual draft would not train the logits.
- The trainer uses a fixed CPU device and a 500-step maximum block. The explicit dummy mode is limited to 10 steps and is not production data.

### Runner and corpus status

The CPU workflow is now running compile, parameter-count, unit-test and 10-step dummy-smoke gates. Earlier attempts failed because invoking `python train/train.py` placed the `train/` directory, not repository root, on `sys.path`; the workflow was corrected to invoke `python -m train.train`. Do not count those failed attempts as validation.

The source-library workflow has passed dependency installation and downloader syntax validation and is currently sampling public datasets. No artifact is considered available until the workflow completes and the uploaded manifest/files are inspected. The final corpus remains blocked by missing owner data and the unresolved per-file code-license allocation.

**The architecture is now implemented in first-pass code, not validated as a learning system. M2 integration work is underway by explicit owner instruction; training on real data, performance claims, holdout evaluation and merge remain gated.**
