# Holdout emparejado de 500.000 pasos

**Fecha:** 24 de noviembre de 2026  
**Run:** [45454209241](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45454209241)  
**Artefacto de predicciones:** [14959695455](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45454209241/artifacts/14959695455)

## Protocolo

Se evaluaron los checkpoints LUMEN y Transformer de step 500.000 sobre el mismo holdout congelado de 2.431 preguntas. Ambos usaron el mismo tokenizer, contexto máximo 128 y el mismo método de puntuación: NLL condicional media de los tokens de cada opción de respuesta. No hubo actualizaciones durante la evaluación.

Los dos entrenamientos partieron de los checkpoints verificados de step 467.500 y completaron la continuación hasta 500.000 en bloques recuperables. El corpus fue v3.2.005 (279.613 filas), con el mismo hash de datos y tokenizer en ambos modelos.

## Resultados

| Modelo | Aciertos | Total | Accuracy | NLL media |
|---|---:|---:|---:|---:|
| LUMEN | 816 | 2.431 | 33,57 % | 5,5997249 |
| Transformer | 840 | 2.431 | 34,55 % | 5,3877226 |

Transformer obtiene 24 respuestas correctas más, una diferencia de 0,987 puntos porcentuales, y una NLL media 0,2120023 menor.

### Comparación emparejada

| Resultado pregunta a pregunta | Preguntas |
|---|---:|
| Ambos correctos | 535 |
| Solo LUMEN correcto | 281 |
| Solo Transformer correcto | 305 |
| Ambos incorrectos | 1.310 |
| **Total** | **2.431** |

El test exacto bilateral de McNemar, calculado sobre las 586 preguntas discordantes, da **p = 0,3421**. La diferencia observada no es estadísticamente concluyente en este conjunto. No se declara un ganador definitivo.

### Evolución respecto al check de step 467.500

| Modelo | Aciertos 467.500 | Aciertos 500.000 | Cambio | NLL 467.500 | NLL 500.000 |
|---|---:|---:|---:|---:|---:|
| LUMEN | 807 | 816 | +9 | 5,5894 | 5,5997 |
| Transformer | 822 | 840 | +18 | 5,4356 | 5,3877 |

El Transformer mejora tanto en aciertos como en NLL durante este tramo. LUMEN suma nueve aciertos, aunque su NLL media aumenta ligeramente. Estos cambios describen únicamente este holdout y no prueban una tendencia universal.

La exactitud de ambos modelos permanece cerca del nivel de azar aproximado del conjunto (31,62 % para su mezcla de preguntas de tres y cuatro opciones). Por ello, el resultado debe interpretarse como una comparación de modelos pequeños en una tarea limitada, no como demostración de competencia general.

## Checkpoints y trazabilidad

- Checkpoint LUMEN step 500.000: artefacto lumen-step-500000-v32-45454209241.
- Checkpoint Transformer step 500.000: artefacto transformer-step-500000-v32-45454209241.
- Holdout: artefacto ronin-v32-step500000-paired-holdout-45454209241, ID 14959695455.
- Run completo: [45454209241](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45454209241).

## Siguiente tramo: 500.000 → 750.000

La continuación mantiene ambos modelos emparejados y encadena **25 bloques de 10.000 pasos por arquitectura**. Se conservan los momentos del optimizador y el RNG; se inicia una nueva transición cosenoidal desde el estado efectivo del scheduler de step 500.000 hacia LR objetivo 3e-5 en step 750.000.

Se mantiene el corpus v3.2.005 autorizado, tokenizer compartido, seed 42, batch 8, contexto 128 y holdout congelado. LUMEN conserva su perfil check 0 con lazy-backward y surprise-gate; Transformer sigue siendo el baseline. Un nuevo holdout común se ejecutará al finalizar ambos modelos.

El corpus candidato v4 queda expresamente excluido: su auditoría independiente detectó un registro sin source_document_id y no autorizó su uso.
