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
