# RONIN-LUMEN V1 — procedencia y preparación de datos

## Estado

**No existe todavía un dataset final de entrenamiento.** El workflow `.github/workflows/prepare-data.yml` prepara un artefacto temporal de candidatos externos; no crea `train_25k.jsonl`, no modifica el holdout y no inicia entrenamiento.

La receta de la especificación contiene una tensión que no se debe resolver en silencio:

- La tabla de §7.2 asigna los 25.000 registros a seis familias externas.
- El párrafo anti-atractor exige que el 80% sea corpus probado y el 20% sea corpus propio.
- Para respetar el 80/20, el muestreador reserva 20.000 filas externas y deja 5.000 para el corpus curado por el propietario.
- No se fabricará ese 20% propio. Hasta recibirlo, el dataset final permanece bloqueado.

La asignación externa se escala proporcionalmente desde la tabla original. Las cuotas actualmente implementadas suman 19.000 filas; las 1.000 restantes corresponderían a Python-Edu/The Stack y están bloqueadas por auditoría de licencias por archivo.

## Fuentes externas y condiciones

| Fuente | Dataset / subset | Cuota externa | Licencia declarada | Estado |
|---|---|---:|---|---|
| TinyStories | `roneneldan/TinyStories` | 4.000 | CDLA-Sharing-1.0 | Muestreo permitido sujeto a obligaciones share-alike |
| BabyLM | `BabyLM-community/BabyLM-2026-Strict` | 4.000 | MIT | Muestreo habilitado |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu`, `sample-10BT` | 2.000 | ODC-By-1.0 | Muestreo habilitado; atribución obligatoria |
| DCLM | `mlfoundations/dclm-baseline-1.0` | 2.000 | CC-BY-4.0 | Muestreo habilitado; atribución obligatoria |
| Cosmopedia v2 | `HuggingFaceTB/smollm-corpus`, `cosmopedia-v2` | 4.000 | ODC-By-1.0 (tarjeta del corpus) | Muestreo de `text` solamente; excluir `prompt` y `seed_data` |
| Orca-Math | `microsoft/orca-math-word-problems-200k` | 1.500 | MIT | Muestreo habilitado; validar soluciones |
| TinyGSM | `TinyGSM/TinyGSM` | 1.500 | MIT (tarjeta del dataset) | Muestreo habilitado; validar código/soluciones |
| Python-Edu / The Stack V2 | `HuggingFaceTB/smollm-corpus`, `python-edu` | 1.000 | ODC-By a nivel de corpus; licencias originales por fichero | **Bloqueado** hasta resolver atribución y licencia individual |
| GSM8K | `openai/gsm8k` | 0 | MIT | **Solo benchmark/evaluación**, nunca entrenamiento |

Las licencias anteriores son las declaradas en las tarjetas oficiales consultadas. No equivalen a una opinión legal ni eliminan las condiciones de las obras subyacentes. En particular:

- ODC-By requiere atribución de la base de datos.
- CDLA-Sharing incorpora obligaciones de compartir bajo sus condiciones.
- The Stack exige respetar las licencias originales de cada repositorio y mantener la información de procedencia. No se incluirá código solo porque la colección agregada tenga una etiqueta abierta.
- El corpus FineWeb-Edu deriva de Common Crawl y está sujeto también a sus términos.
- Los textos sintéticos no se marcarán como hechos verificados por el mero hecho de proceder de un dataset educativo.

## Idioma y calidad: limitación visible

Las fuentes públicas seleccionadas son mayoritariamente inglesas. La especificación no define una cuota lingüística explícita ni una fuente española concreta. El corpus propio de 5.000 filas será decisivo para la voz española, pero no se asumirá que compensa por sí solo el desequilibrio lingüístico. Debe revisarse la composición real antes de congelar el dataset.

La columna `confirmed` es una etiqueta de procedencia del pipeline, no una garantía epistemológica de que cada afirmación sea verdadera. En la primera preparación se marca como `false` todo texto que no haya sido verificado individualmente; FineWeb-Edu y DCLM no se tratan como fuentes de hechos confirmados.

## Reproducibilidad

- Seed congelada: `42`.
- Lectura mediante Hugging Face Datasets en streaming.
- Resolución de la revisión actual de cada dataset a un commit SHA inmutable, registrado en el manifiesto.
- Deduplicación exacta por SHA-256 del texto normalizado, compartida entre fuentes.
- Filtro de prefijo de cinco tokens por fuente durante el staging: si un grupo supera diez registros, se conserva uno. El builder final aún debe repetir la auditoría de prefijos sobre la mezcla completa.
- Cada fichero JSONL tiene su SHA-256 en `manifest.json`.
- Los candidatos se almacenan como artefacto de GitHub Actions, no en Git.
- El muestreador falla si no alcanza la cuota; no reduce cuotas silenciosamente.
- El holdout `holdout_500.jsonl` no se lee ni se modifica durante esta preparación.

Workflow disponible en GitHub Actions: **Prepare RONIN-LUMEN source libraries**. El artefacto esperado se llama `lumen-source-candidates-seed-42`. Ejecución completada: [run 44887380018](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44887380018). Se generó el artifact `lumen-source-candidates-seed-42` (artifact ID `14700274265`, ZIP SHA-256 `eab0eb4f513238d6f76f02c5ac760959f43ba6404cf5b722fbe3f44da17d6bfa`). Se verificaron los siete recuentos de filas y los siete SHA-256 internos contra `data/source_manifest_seed42.json`. El artifact contiene 19.000 candidatos externos, no el dataset final de 25.000.

## Pendientes antes de crear `train_25k.jsonl`

1. Conseguir y auditar los 1.000 ejemplos Python-Edu/The Stack o autorizar una sustitución compatible.
2. Incorporar las 5.000 filas originales curadas por el propietario.
3. Resolver explícitamente la tensión entre la tabla de cuotas de §7.2 y el reparto 80/20.
4. Revisar la cuota de español y el balance por idioma.
5. Validar matemáticas, código, deduplicación semántica y contaminación del holdout.
6. Registrar licencias y atribuciones finales por fuente en el manifiesto.

**No se entrena con este staging. No se marca el dataset como listo hasta cerrar todos los puntos anteriores.**
