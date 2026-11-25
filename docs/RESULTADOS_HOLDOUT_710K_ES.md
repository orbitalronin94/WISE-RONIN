# Holdout doble emparejado — step 710.000

**Fecha:** 25 de noviembre de 2026  
**Workflow:** [45558806917](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45558806917)  
**Artefacto de resultados:** [14996339403](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45558806917/artifacts/14996339403)

## Resumen ejecutivo

LUMEN queda por delante del Transformer en el holdout oficial de elección múltiple de 2.431 preguntas: **839 frente a 814 aciertos**, con una NLL media ligeramente menor. Es una inversión respecto al holdout de step 500.000. Sin embargo, la comparación pregunta a pregunta no es estadísticamente concluyente y el holdout abierto de 100 preguntas sigue mostrando repetición y respuestas poco coherentes en ambos modelos.

## 1. Protocolo oficial: elección múltiple

Se evaluaron los checkpoints exactos de step 710.000 sobre el mismo conjunto congelado de 2.431 preguntas. Se utilizó el tokenizer compartido, contexto máximo 128 y NLL condicional media de los tokens de cada opción. No hubo actualizaciones de pesos durante la evaluación.

| Modelo | Aciertos | Total | Accuracy | NLL media |
|---|---:|---:|---:|---:|
| **LUMEN** | **839** | 2.431 | **34,5125 %** | **5,4783408** |
| Transformer | 814 | 2.431 | 33,4842 % | 5,4867445 |
| Diferencia LUMEN − Transformer | +25 | — | +1,0283 pp | −0,0084036 |

Las dos métricas registradas favorecen a LUMEN, aunque la diferencia de NLL es pequeña.

### Análisis emparejado por pregunta

| Resultado | Preguntas |
|---|---:|
| Ambos correctos | 498 |
| Solo LUMEN correcto | 270 |
| Solo Transformer correcto | 258 |
| Ambos incorrectos | 1.230 |
| **Total** | **2.431** |

Entre las 528 preguntas discordantes, el test exacto bilateral de McNemar da **p ≈ 0,6322**. Por tanto, no hay evidencia estadística suficiente para afirmar que la ventaja de 25 aciertos refleje una diferencia estable más allá de la muestra evaluada.

### Evolución desde step 500.000

| Modelo | Aciertos 500k | Aciertos 710k | Cambio |
|---|---:|---:|---:|
| LUMEN | 816 | 839 | **+23** |
| Transformer | 840 | 814 | **−26** |

En 500k el Transformer aventajaba a LUMEN por 24 aciertos; en 710k LUMEN lo aventaja por 25. Es una inversión de 49 respuestas en la diferencia relativa. Debe interpretarse como evolución observada de estos checkpoints bajo esta campaña, no como prueba causal de que un mecanismo concreto haya producido el cambio.

## 2. Holdout abierto informal: 100 preguntas

Se ejecutó también scripts/freeform_identity_general_holdout.py sobre el conjunto fijo de 100 preguntas en español. El primer prompt se verificó literalmente como **«¿Quién eres?»**. Los dos modelos recibieron los mismos prompts, sin opciones ni respuestas de referencia.

Configuración: greedy argmax, sin penalización de repetición, máximo de 96 tokens nuevos y contexto máximo 128. Este test es cualitativo: no calcula accuracy ni compara contra una respuesta patrón.

| Indicador | LUMEN | Transformer |
|---|---:|---:|
| Finalización por EOS | 82/100 | 83/100 |
| Alcanzaron el máximo de 96 tokens | 18/100 | 17/100 |
| Media de tokens generados | 42,02 | 43,21 |
| Respuestas que contienen «la que la que» | 41/100 | — |
| Respuestas que contienen «Contexto documental» | 20/100 | — |
| Respuestas que contienen «en Python» | — | 22/100 |
| Respuestas que contienen «música» | — | 10/100 |

Los recuentos de frases indican presencia literal en una respuesta, no exclusividad ni evaluación semántica. Se incluyen como señales diagnósticas de recurrencia.

### Primera pregunta: «¿Quién eres?»

- **LUMEN:** «La relación entre la que la que la que la que…» — cae en repetición local.
- **Transformer:** «La diferencia de la estructura de la memoria es la de la música…» — deriva hacia un bucle temático y repetitivo.

Ninguna de las dos respuestas constituye una contestación satisfactoria de identidad. El holdout libre confirma que la ventaja en elección múltiple no equivale a fluidez conversacional resuelta.

## 3. Interpretación conjunta

1. **Elección múltiple:** ventaja observada de LUMEN en aciertos y NLL.
2. **Evidencia pareada:** la diferencia de aciertos no es estadísticamente concluyente (McNemar p≈0,632).
3. **Generación abierta:** ambos modelos siguen mostrando bucles; LUMEN conserva «la que» y «Contexto documental», mientras Transformer presenta recurrencias de «en Python» y «música».
4. **EOS:** ambos terminan por EOS en más de cuatro quintas partes de las respuestas, pero esto no garantiza que la respuesta sea coherente.
5. **Causalidad:** el resultado no aísla el efecto de cada órgano, mecanismo, scheduler o tramo de datos. Se comparan los checkpoints producidos por la campaña, no intervenciones independientes.

La conclusión prudente es que LUMEN ha recuperado la ventaja en este holdout oficial, pero no hay base para proclamar superioridad general ni para dar por resuelto el problema de generación libre.

## 4. Trazabilidad y artefactos

- Workflow de holdouts: [run 45558806917](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45558806917).
- Artefacto de evaluación: [14996339403](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45558806917/artifacts/14996339403).
- JSON oficial: paired_step710000_official_2431.json.
- JSON de generación libre: freeform-step710000/step710000_freeform_100.json.
- Informe de generación libre: freeform-step710000/step710000_freeform_100.md.
- Checkpoint LUMEN step 710.000: [artefacto 14996386123](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45480785478/artifacts/14996386123).
- Checkpoint Transformer step 710.000: [artefacto 14988880143](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45480785478/artifacts/14988880143).

La evaluación es reproducible a partir de esos checkpoints y del artefacto de resultados. El holdout oficial y el conjunto abierto se mantienen fuera del entrenamiento.
