# Registro de experimentos

Este documento conserva los hitos medidos y separa los resultados confirmados de los experimentos en curso. Los enlaces de Actions son la fuente primaria de logs y artefactos.

## 1. SFT inicial — 10.000 pasos

- Run: [44709299157](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44709299157)
- Estado: completado correctamente.
- Checkpoint final: paso 10.000.
- Dataset de entrada: tres JSONL heredados.
- Filas declaradas en origen: 147.627.
- Filas después de deduplicación exacta: 88.348.
- Parámetros efectivos: 4.267.520.
- Vocabulario BPE efectivo: 1.297.
- Batch: 4; secuencia máxima: 512; LR máximo: 3e-4.
- Tiempo de entrenamiento: 1.562,47 segundos.
- Loss final registrada: 0,16234.

La loss descendió de forma pronunciada. Esto prueba optimización del objetivo de entrenamiento, no competencia factual ni generalización.

### Holdout greedy

El evaluador heurístico marcó 35/500 (7%). Sin embargo:
- Los 500 registros contienen solo 55 preguntas distintas.
- Se observó repetición elevada de respuestas y plantillas.
- Hubo errores factuales claros, incluso en operaciones sencillas.
- La puntuación por subcadenas puede generar falsos positivos y falsos negativos.

Conclusión: el checkpoint sirve como línea base reproducible, no como modelo validado.

## 2. Corpus ampliado

- PR: [#3](https://github.com/orbitalronin94/WISE-RONIN/pull/3)
- Generador: `generate_wise_ronin_expanded.py`
- Corpus: `data/expanded/wise_ronin_expanded.jsonl`
- Filas: 147.627.
- Prompts únicos: 147.627.
- Respuestas literales únicas: 147.627.
- Anclas factuales: 402.
- Grupos independientes `fact_id`: 10.341.
- Operaciones aritméticas base: 9.939.

Las variantes multiplican formulaciones, no conocimiento factual independiente. El corpus se genera desde un banco integrado y aritmética exacta; no es una enciclopedia externa auditada.

## 3. Currículo equilibrado — 80.000 filas

- PR: [#4](https://github.com/orbitalronin94/WISE-RONIN/pull/4)
- Builder: `build_balanced_80k.py`
- Validación: [run 44721788327](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44721788327)
- Filas: 80.000 exactas.
- Prompts únicos: 80.000.
- Respuestas únicas: 80.000.
- Filas fuente utilizadas: 66.280.
- Filas fuente reservadas/no seleccionadas: 81.347.
- Variantes adicionales de geografía/general: 4.360, conservando el `fact_id`.
- Nuevas divisiones exactas: 9.360 filas.

Distribución objetivo:
- Geografía: 10.000.
- Química: 15.000.
- General: 15.000.
- Sumas: 10.000.
- Restas: 10.000.
- Multiplicaciones: 10.000.
- Divisiones: 10.000.

La ampliación de geografía y general no aumenta el número de hechos independientes; las operaciones de división añadidas sí representan nuevas tareas aritméticas exactas.

## 4. Continuación desde el checkpoint 10k

- Run: [44721903295](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44721903295)
- Configuración: dos épocas sobre 80.000 filas, batch 8, LR 1,5e-4.
- Actualizaciones adicionales previstas: 20.000.
- Paso global esperado: 30.000.
- Tokenizer: reutilizado desde el checkpoint original.
- ORPO: desactivado; no hay pares reales `chosen/rejected`.

**Estado observado al actualizar esta documentación:** el job había validado la descarga del checkpoint, la construcción del currículo, el balance exacto, la reserva y la compilación. El paso de entrenamiento seguía en ejecución; evaluación y subida de artefactos estaban pendientes. Consultar el enlace de Actions para el estado final.

## Próxima evaluación

Comparar checkpoint 10k y checkpoint 30k (si el segundo termina) en el mismo holdout y con idénticos parámetros:
1. Factualidad revisada semánticamente, separada de coincidencia literal.
2. Diversidad de respuestas y diversidad por pregunta.
3. Frecuencia de aperturas, coletillas y motivos repetidos.
4. Diferencia entre greedy y sampling.
5. Errores y regresiones por categoría.

No comparar solo la loss ni tratar 500 registros repetidos como 500 preguntas independientes.
