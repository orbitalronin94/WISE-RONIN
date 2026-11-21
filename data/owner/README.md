# Datos aportados por el propietario: requisitos antes de congelar el corpus

El constructor del dataset se detiene deliberadamente hasta disponer de estos dos ficheros:

- `curated_5000.jsonl`: exactamente 5.000 registros originales y curados manualmente.
- `python_edu_licensed.jsonl`: exactamente 1.000 registros de código, cada uno vinculado a un fichero de origen inmutable y con licencia auditada individualmente.

No se generará relleno sintético para completar ninguna cuota. No copiar el holdout histórico a estos ficheros.

## Registro curado por el propietario

Cada línea debe ser un objeto JSON con:

- `prompt`: entrada o contexto no vacío.
- `completion`: texto objetivo no vacío.
- `origin`: exactamente `owner_curated`.
- `license`: licencia real o términos autorizados explícitamente por el propietario.
- `rights_confirmed`: booleano `true`, solo después de confirmar que el registro puede usarse para entrenamiento e incluirse en el artefacto del repositorio.
- `language`: código ISO opcional (por ejemplo `es`, `en`); si se omite, la auditoría asigna una etiqueta heurística aproximada.
- `type`: categoría de tarea opcional.

## Registro de código con licencia auditada individualmente

Cada línea debe incluir `prompt` y `completion` no vacíos, además de todos estos campos:

- `source_repo`: propietario/repositorio canónico.
- `source_path`: ruta exacta del fichero original.
- `source_commit`: SHA inmutable del commit de Git, no el nombre de una rama.
- `source_hash`: SHA-256 del fichero de código original.
- `license`: licencia identificada para ese fichero exacto en ese commit.
- `license_url`: URL canónica del texto de licencia o de la evidencia de licencia del repositorio.
- `license_file_path`: ruta del fichero de licencia/aviso en el repositorio de origen.
- `license_file_sha256`: SHA-256 de ese fichero de evidencia.
- `attribution`: texto de atribución que debe acompañar a la redistribución.
- `license_audited`: booleano `true`, solo después de que una persona haya comprobado la licencia a nivel de fichero.
- `rights_confirmed`: booleano `true`, solo después de confirmar que se permiten el entrenamiento y la redistribución previstos.
- `type`: normalmente `codigo`.
- `language`: código ISO opcional.

La etiqueta de licencia de un dataset no basta para el código procedente de The Stack. Se rechaza cualquier fila sin evidencia de licencia por fichero o sin las marcas explícitas de auditoría. Un booleano es una declaración de revisión, no una verificación legal automática.

## Balance lingüístico

Las fuentes externas actuales son mayoritariamente inglesas. El constructor informa de una distribución heurística de idiomas y del porcentaje estimado de español, pero no inventa un umbral. Una vez acordada la proporción deseada, ejecutar el constructor con `--min-spanish-share` para imponerla. Para obtener recuentos fiables de producción, etiquetar manualmente los registros ambiguos o multilingües en vez de tratar la heurística como verdad de referencia.

## Gates para construir el corpus

Desde la raíz del repositorio, una vez presentes ambos ficheros:

```bash
python -m data.build_dataset \
  --sources artifacts/lumen-source-candidates \
  --owner data/owner \
  --holdout data/holdout_500.jsonl \
  --output data/train_25k.jsonl
```

El constructor verifica hashes/revisiones de las fuentes, cuotas exactas, procedencia por fichero, duplicados de texto, prefijos de cinco tokens repetidos, integridad de las 500 filas del holdout, coincidencias exactas y subcadenas largas de prompts. Escribe `train_25k.manifest.json` con recuentos, hashes de fuentes, estimaciones lingüísticas y SHA-256 de la salida.

La auditoría de coincidencia léxica es conservadora, pero no demuestra ausencia de fuga semántica. El manifiesto final debe revisarse manualmente antes de entrenar el tokenizer. El entrenamiento del tokenizer y el del modelo son operaciones separadas.
