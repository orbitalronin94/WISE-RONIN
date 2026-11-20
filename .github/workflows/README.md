# GitHub Actions — WISE-RONIN

Índice de automatizaciones y sus responsabilidades. Los workflows publican artefactos de ejecución cuando corresponde; los checkpoints no se versionan como commits ordinarios.

| Workflow | Archivo | Función |
|---|---|---|
| First 10k SFT | [wise-ronin-first-10k.yml](wise-ronin-first-10k.yml) | Entrena desde cero durante 10.000 actualizaciones y evalúa greedy. |
| Expand dataset | [wise-ronin-expand-dataset.yml](wise-ronin-expand-dataset.yml) | Genera y valida el corpus determinista ampliado. |
| Dataset audit | [wise-ronin-dataset-audit.yml](wise-ronin-dataset-audit.yml) | Audita duplicados y distribución del corpus heredado; no entrena. |
| Balanced 80k | [wise-ronin-balanced-80k.yml](wise-ronin-balanced-80k.yml) | Descarga checkpoint 10k, construye currículo 80k, entrena dos épocas y evalúa sampling. |
| Balanced 80k validation | [wise-ronin-balanced-80k-validation.yml](wise-ronin-balanced-80k-validation.yml) | Valida builder, categorías, unicidad y compilación en PR. |

## Reglas operativas

- Mantener `holdout_wise_ronin_500.jsonl` fuera del entrenamiento y del tokenizer.
- No activar ORPO sin pares de preferencia etiquetados.
- No eliminar ni sobrescribir las filas fuente no seleccionadas.
- Mantener las rutas de scripts/datos sincronizadas entre workflows y documentación.
- Revisar los informes y generaciones, no solo el exit code del workflow.
