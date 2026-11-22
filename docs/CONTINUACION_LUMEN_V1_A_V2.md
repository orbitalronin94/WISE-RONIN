# Continuación de Lumen: paso 1.000 → 2.000 con corpus español v2

## Parent exacto

- Workflow de origen: [45068173591](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45068173591).
- Checkpoint: `checkpoints/lumen/step_1000.pt`.
- Arquitectura: `ronin_lumen_v3_3_neuroplasticity`.
- Se conserva el estado del modelo, AdamW, scheduler, RNG, trazas de elegibilidad, historial de diagnósticos y eventos de sueño.
- Corpus padre v1 SHA-256: `d0dc9e12dc1b72b01eeafd42e204c59543e2196fc516ff1d67b1be9f78632b3f`.

## Corpus v2

- Artefacto congelado del workflow [45065384110](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45065384110).
- 9.357.876 registros de entrenamiento: 70.105 de v1, 3.423.248 chunks CSIC y 5.864.523 chunks TDX.
- Tokenizer compartido sin cambios: vocabulario 4.096 y SHA-256 `d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe`.
- Validación (1.413) y holdout (2.431) se copian byte por byte desde v1.
- CoWeSe permanece excluido por incertidumbre sobre los derechos de los textos subyacentes.

## Muestra de entrenamiento para este piloto

El corpus v2 completo tiene más de nueve millones de filas, pero este tramo de 1.000 pasos solo consume 8.000 ejemplos de minibatch. Para evitar cargar y tokenizar 9,36 millones de filas en un runner CPU limitado, el workflow construye una muestra reservoir estratificada, determinista y de 200.000 filas:

- 20.000 filas de v1.
- 72.000 chunks CSIC.
- 108.000 chunks TDX.
- Semilla de muestreo 42.

La muestra se selecciona recorriendo el corpus v2 completo, verifica el SHA-256 publicado en el manifiesto y luego genera una caché BPE exacta. La muestra no reemplaza ni modifica el artefacto v2 congelado; es el subconjunto operativo de este piloto. Los lotes se extraen de esa muestra con reemplazo.

## Continuación

- Lumen: paso global 1.000 → 2.000; no se reinicializan pesos ni optimizador.
- Batch 8, contexto 128, semilla 42, todos los órganos activos.
- Se conserva el perfil CPU anterior.
- El scheduler agotado del tramo 0→1.000 se continúa mediante una transición coseno nueva entre los pasos 1.000 y 2.000, con warmup de 20 actualizaciones. Los momentos AdamW permanecen intactos.
- El cambio de corpus solo se permite explícitamente para este checkpoint exacto y exige el mismo tokenizer.
- Checkpoints cada 250 pasos y recuperación cada 100.
- La validación y el holdout permanecen fuera del entrenamiento. No se afirma mejora de generalización hasta evaluarlos.

Este es un tramo de continuación experimental de Lumen, no una comparación emparejada nueva contra Transformer.
