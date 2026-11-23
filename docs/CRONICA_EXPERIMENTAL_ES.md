# Crónica experimental de Lumen-Ronin
>
> Del primer encendido al linaje evaluado en 133.200 pasos.
>
> **Corte documental:** 23 de noviembre de 2026. El último checkpoint emparejado evaluado es 133.200, sobre el corpus español v2.

## Cómo leer esta cronología

Esta crónica separa tres cosas que no deben confundirse:

- **Linaje principal:** checkpoints que alimentan la secuencia emparejada Lumen/Transformer.
- **Pilotos y ramas exploratorias:** pruebas que sirvieron para descubrir fallos, medir rendimiento o ensayar arquitectura, pero que no son padres del checkpoint actual.
- **Estado de ejecución:** un workflow puede fallar después de guardar un checkpoint válido; también puede terminar bien sin que eso convierta su resultado en una prueba de generalización.

Los holdouts de esta tabla son evaluaciones congeladas de 2.431 preguntas españolas. Accuracy y NLL se muestran por separado: no miden lo mismo. Los resultados son descriptivos; una diferencia de accuracy no implica por sí sola significación estadística.

## I. La primera chispa: pilotos y correcciones

### Piloto de 500 pasos — 21 de noviembre

[Run 44900623884](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44900623884) — piloto emparejado de tamaño comparable, Lumen y Transformer.

- Corpus piloto de 19.000 filas; seed 42; batch 8.
- Se alcanzó el checkpoint 500 en ambos modelos.
- El log reveló un problema estructural de arranque: a partir de los pasos 200–500 varias capas Delta dejaron de escribir y la ruta Homeostasis podía anular la representación en lugar de modularla.
- La telemetría mostraba que las capas superiores perdían actividad efectiva. El porcentaje de neuronas muertas no era suficiente para detectar el problema: el fallo estaba en la ruta de actualización y en la escritura, no simplemente en un contador de unidades.

**Qué dejó:** el piloto no se promocionó como demostración de aprendizaje. Sirvió para localizar el fallo de arranque y motivó la ruta residual de V3.1 y la migración del estado homeostático. Es prehistoria técnica, no un checkpoint padre del linaje actual.

### Arranque emparejado desde cero hasta 1.000

[Entrenamiento 0 → 1.000, run 45012493289](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45012493289) · [Holdout de step 1.000, run 45013766761](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45013766761)

La línea comparativa que se siguió después comenzó desde cero con ambos modelos, mismo corpus/tokenizer, seed 42, batch 8 y contexto 128. El primer holdout español dejó esta referencia:

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen | 759/2.431 | 31,22 % | **7,2570** |
| Transformer | 764/2.431 | **31,43 %** | 14,6228 |

El Transformer acertó cinco preguntas más, pero su NLL fue mucho peor. La primera lectura ya exigía separar capacidad de clasificación y calidad de las probabilidades.

## II. El primer recorrido largo

### 1.000 → 11.000 — primera continuación extensa

[Run 45014470774](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45014470774) · ejecución completada.

- Ambos modelos continuaron desde el par de checkpoints de 1.000.
- Se mantuvieron el corpus español congelado, tokenizer y protocolo común.
- Se alcanzó step 11.000 y se ejecutó el holdout emparejado.

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen | 741/2.431 | 30,48 % | **7,0225** |
| Transformer | 745/2.431 | **30,65 %** | 7,0573 |

La precisión quedó prácticamente empatada; Lumen pasó a tener una NLL ligeramente menor. La loss de entrenamiento descendía, pero el holdout no autorizaba a hablar de una ventaja clara.

### 11.000 → 16.000 — incorporación de los órganos afectivos

[Run 45021399600](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45021399600) · ejecución completada.

Se introdujo en la continuación de Lumen la regulación separada de glucosa, dolor y placer. Las señales se actualizan en los controles de loss y modulan el presupuesto interno de energía. No se utilizaron las preguntas del holdout para alimentar esos órganos.

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen | 786/2.431 | **32,33 %** | 7,1953 |
| Transformer | 761/2.431 | 31,30 % | **6,8550** |

Lumen ganó 25 respuestas en accuracy; Transformer conservó mejor NLL. La nueva regulación era una hipótesis de entrenamiento que debía seguir contrastándose, no una causa demostrada de la mejora.

### 16.000 → 21.000 — primera ventaja clara en clasificación

[Run 45025897082](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45025897082) · ejecución completada.

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen | 837/2.431 | **34,43 %** | 7,1043 |
| Transformer | 766/2.431 | 31,51 % | **6,7644** |

Lumen aventajó al baseline por 71 respuestas (+2,92 puntos porcentuales). Transformer mantuvo menor NLL. Fue una señal prometedora en clasificación, no una victoria universal en todas las métricas.

## III. La prueba de continuidad

### 21.000 → 26.000

[Run 45030262114](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45030262114) · ejecución completada.

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen | 823/2.431 | **33,85 %** | **6,6643** |
| Transformer | 773/2.431 | 31,80 % | 6,7229 |

Lumen conservó 50 aciertos de ventaja y, esta vez, también obtuvo una NLL ligeramente menor. El resultado fue favorable en las dos métricas, aunque el holdout seguía siendo una única muestra congelada.

### 26.000 → 31.000

[Run 45034338841](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45034338841) · ejecución completada.

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen | 797/2.431 | **32,78 %** | **6,5564** |
| Transformer | 783/2.431 | 32,21 % | 6,6534 |

La ventaja de Lumen se redujo a 14 respuestas, pero la NLL siguió siendo menor. Importa la trayectoria: no hubo una mejora monótona de accuracy en cada checkpoint, aunque la NLL de Lumen continuó descendiendo.

### 31.000 → 41.000 — checkpoint recuperado tras fallo de workflow

[Entrenamiento 45037602119](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45037602119) · [Holdout independiente a 41.000, run 45044070812](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45044070812).

El workflow, cuyo nombre declaraba una continuación 31.000 → 51.000, se marcó como fallido en la verificación final del scheduler. Sin embargo, ambos jobs habían guardado sus checkpoints de step 41.000. Se verificaron y evaluaron en una ejecución independiente. Por eso se conserva el checkpoint, pero no se presenta el workflow original como un éxito completo.

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen | 801/2.431 | 32,95 % | **6,4777** |
| Transformer | 814/2.431 | **33,48 %** | 6,6670 |

Aquí el Transformer pasó a liderar ligeramente en accuracy, mientras Lumen mantuvo mejor NLL. La diferencia de precisión no fue concluyente en el contraste emparejado.

### 41.000 → 61.000 — continuación larga y nueva referencia

[Run 45044892658](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45044892658) · ejecución completada.

Ambos modelos continuaron desde 41.000 hasta 61.000 con el mismo corpus español congelado, tokenizer, batch 8, contexto 128 y seed 42.

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen | 795/2.431 | 32,70 % | **6,1771** |
| Transformer | 824/2.431 | **33,90 %** | 6,4988 |

El Transformer aventajó a Lumen por 29 respuestas. Lumen, sin embargo, conservó una NLL media 0,3217 menor. Frente a 41.000, ambos mejoraron NLL; la de Lumen descendió con más intensidad. No hay base para ocultar que el baseline gana en accuracy en este checkpoint.

## IV. El cierre de la primera campaña: 61.000 → 66.000

[Run 45061486739](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45061486739) · completado y evaluado.

El entrenamiento parte de los artefactos verificados del run 45044892658:

- Lumen: checkpoint 61.000, con órganos ×1,50 y Delta con dos refinamientos paralelos.
- Transformer: checkpoint 61.000, arquitectura y configuración baseline sin cambios.
- Mismo corpus/tokenizer congelados, batch 8, contexto 128 y seed 42.
- Ambos recorren 5.000 pasos; el holdout emparejado de 2.431 preguntas se ejecuta cuando los dos terminen.

Ambos jobs y el holdout de 2.431 preguntas terminaron correctamente. Resultados a 66k: Lumen 792/2.431 (32,58 %), NLL 6,0756; Transformer 824/2.431 (33,90 %), NLL 6,4868. Este checkpoint se convirtió en el padre del experimento v2.

### Un duplicado que no pertenece al linaje promovido

El run [45054687034](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45054687034) aparece como otra continuación 41.000 → 61.000 en curso. No es el padre utilizado por el run 45061486739: este último declara y verifica explícitamente los artefactos 61.000 del run 45044892658. No deben mezclarse sus resultados ni sus checkpoints sin una auditoría de procedencia.

## V. Las ramas que enseñaron a medir

Los siguientes experimentos son parte de la historia de ingeniería, pero no son tramos adicionales del linaje 0 → 61.000:

| Familia | Runs representativos | Qué investigaron |
|---|---|---|
| Continuaciones exploratorias 2k–9k | [44936542271](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44936542271), [44940676355](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44940676355), [44950089771](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44950089771) | Continuidad, estabilidad, compilación y primeros bloques largos. Algunos intentos fallaron o fueron sustituidos; no deben sumarse como una sola cadena de checkpoints. |
| CPU y cuello de botella | [44941577031](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44941577031), [45003557749](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45003557749), [45003881342](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45003881342) | Separación de forward/backward, eager frente a compile, hilos, fastpaths y coste real de la recurrencia. |
| Sueño y replay | [45008983356](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45008983356), [45010378745](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45010378745) | Verificar si había registros consolidados y si el sueño podía ejecutarse con la memoria disponible. |
| Corpus español | [45007651092](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45007651092), [45008636659](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45008636659), [45021399600](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45021399600) | Construcción, deduplicación, filtrado de componentes por licencia, manifiesto y congelación de los splits. |
| Benchmark de órganos | [45053192228](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45053192228), [45054764955](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45054764955), [45062110496](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45062110496) | Hipótesis de dolor/placer y simplificación Delta. Son microbenchmarks de entrenamiento, no holdouts ni evidencia de generalización. |

## VII. Qué ha enseñado el recorrido

1. **La loss no basta.** En varios checkpoints la NLL y la accuracy cuentan historias diferentes.
2. **La continuidad importa.** Los checkpoints conservan pesos, buffers, RNG y estado del optimizador; no son simples exportaciones de pesos.
3. **La arquitectura debe sobrevivir al entrenamiento.** La ruta residual y la corrección del arranque surgieron de fallos observados, no de una analogía decorativa.
4. **La velocidad debe medirse en el trainer real.** Un microbenchmark aislado no garantiza que una optimización acelere la ejecución completa.
5. **Un fallo de workflow no invalida automáticamente un checkpoint, pero tampoco se debe ocultar.** Se verifica el archivo y se evalúa por separado.
6. **La comparación sigue abierta.** Lumen ha mostrado ventajas de clasificación en ciertos puntos y menor NLL en otros; el Transformer ha recuperado la ventaja de accuracy a 41k y 61k.

## Registro de procedencia

- Corpus español v1: 70.105 train / 1.413 validation / 2.431 holdout.
- Holdout fijo de evaluación: 1.931 preguntas SIQA-es y 500 OpenBookQA-es.
- Seed: 42.
- Batch: 8.
- Contexto: 128 tokens.
- La evaluación no actualiza pesos ni alimenta glucosa, dolor o placer.
- Los runs fallidos, cancelados y duplicados se conservan como historia operativa, pero no se promocionan silenciosamente como progreso.


## V. La nueva frontera: 66.000 → 73.500 sobre corpus español v2

[Run 45071952349](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45071952349) · ejecución completada con quality gate, preparación de corpus, ambos entrenamientos y holdout final correctos.

### Preparación común

- Corpus v2 congelado en [run 45065384110](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45065384110).
- 9.357.876 filas de entrenamiento completas; 1.413 de validación; 2.431 preguntas de holdout.
- Muestra determinista de 200.000 filas: 20.000 v1, 72.000 CSIC y 108.000 TDX.
- Tokenizer de 4.096 IDs, PAD=0 y EOS presente; SHA-256 `d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe`.
- CSIC y TDX bajo CC BY 4.0; CoWeSe excluido por incertidumbre sobre los derechos de los textos recolectados.
- Seed 42, batch 8, contexto 128 y caché de tokens compartida.

### Continuación emparejada

Ambos modelos partieron de los checkpoints de step 66.000 y añadieron 7.500 pasos. Lumen mantuvo `ronin_lumen_v3_2_parallel_delta`, todos los órganos activos, perfil ×1,50 y dos refinamientos Delta paralelos. El Transformer conservó su arquitectura baseline. Los dos usaron la misma muestra v2, tokenizer y protocolo de evaluación.

| Modelo | 66k | 73,5k | Cambio |
|---|---:|---:|---:|
| Lumen | 792/2.431 · 32,58 % | 791/2.431 · 32,54 % | −1 acierto |
| Transformer | 824/2.431 · 33,90 % | 792/2.431 · 32,58 % | −32 aciertos |

| Modelo | NLL 66k | NLL 73,5k | Cambio relativo |
|---|---:|---:|---:|
| Lumen | 6,0756 | **5,8320** | −4,01 % |
| Transformer | 6,4868 | 6,4189 | −1,05 % |

En accuracy, el Transformer pasó de +32 aciertos a +1. En NLL, Lumen amplió su ventaja de 0,4112 a 0,5869 puntos. La lectura prudente es que Lumen conservó su nivel de aciertos mientras mejoró con más intensidad la probabilidad asignada a las respuestas; el Transformer perdió 32 aciertos en este holdout. La transición de corpus y los 7.500 pasos ocurrieron juntos, así que este resultado no aísla causalmente el efecto del corpus v2.

### Artefactos promovidos como referencia

- [Run emparejado 66k→73,5k](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45071952349)
- [Holdout final](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45071952349/artifacts/14775305157)
- Checkpoint Lumen 73,5k: artefacto `ronin-lumen-step73500-v2-45071952349`.
- Checkpoint Transformer 73,5k: artefacto `ronin-transformer-step73500-v2-45071952349`.

Este par queda fijado como referencia para la siguiente continuación emparejada. La campaña siguiente prevista es step 73.500 → 133.500, +60.000 pasos por modelo, con corpus v2 y holdout común.

## VI. Qué ha enseñado el recorrido

1. **La loss no basta.** En varios checkpoints la NLL y la accuracy cuentan historias diferentes.
2. **La continuidad importa.** Los checkpoints conservan pesos, buffers, RNG y estado del optimizador; no son simples exportaciones de pesos.
3. **La arquitectura debe sobrevivir al entrenamiento.** La ruta residual y la corrección del arranque surgieron de fallos observados, no de una analogía decorativa.
4. **La velocidad debe medirse en el trainer real.** Un microbenchmark aislado no garantiza que una optimización acelere la ejecución completa.
5. **Un fallo de workflow no invalida automáticamente un checkpoint, pero tampoco se debe ocultar.** Se verifica el archivo y se evalúa por separado.
6. **La comparación sigue abierta.** Lumen conserva una NLL claramente menor a 73,5k, mientras la accuracy está prácticamente empatada; se necesitan más continuaciones y análisis por pregunta.


## VIII. La prueba decisiva de la campaña v2: step 133.200

[Continuación 73,5k→133,5k, run 45082812702](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45082812702) · [Holdout emparejado exacto, run 45123727239](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45123727239).

### Procedencia de los checkpoints

La continuación larga tuvo un desenlace operativo desigual: el Transformer completó step 133.500, mientras el job de Lumen fue interrumpido después de guardar su checkpoint válido de step 133.200. Para no comparar pasos distintos, el workflow de holdout recuperó ese checkpoint de Lumen y avanzó el Transformer desde su checkpoint 133.000 exactamente 200 pasos, preservando el estado de optimizador, scheduler y RNG.

El workflow verificó que ambos checkpoints estaban en **step 133.200** y compartían el mismo SHA de tokenizer. El holdout se ejecutó sobre las mismas 2.431 preguntas, con contexto 128. El job de evaluación terminó correctamente.

### Resultado del holdout

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen-Ronin | **810/2.431** | **33,32 %** | **5,7221** |
| Transformer baseline | 793/2.431 | 32,62 % | 5,8947 |

Lumen obtuvo 17 aciertos más (+0,70 puntos porcentuales) y una NLL 0,1726 menor. En relación con el holdout de 73.500:

| Modelo | Aciertos 73,5k | Aciertos 133,2k | Cambio |
|---|---:|---:|---:|
| Lumen | 791 | **810** | **+19** |
| Transformer | 792 | 793 | +1 |

| Modelo | NLL 73,5k | NLL 133,2k | Cambio |
|---|---:|---:|---:|
| Lumen | 5,8320 | **5,7221** | −0,1099 |
| Transformer | 6,4189 | 5,8947 | −0,5242 |

### La divergencia entre training loss y holdout

En los logs de entrenamiento del tramo 73,5k→133k, el Transformer mantuvo menor loss de minibatch en los intervalos observados y una pendiente de descenso más pronunciada. Sin embargo, en el holdout de step 133.200 Lumen quedó por delante tanto en accuracy como en NLL.

Esta divergencia es el resultado experimental central de este corte: **la menor training loss del Transformer no se tradujo en mejor resultado en este holdout**. Es compatible con la hipótesis de que la arquitectura de Lumen induce una dinámica de aprendizaje distinta, pero no identifica causalidad ni prueba superioridad general. Hace falta repetir con nuevas semillas, ampliar el análisis por categorías y conservar el mismo protocolo de evaluación.

### Artefactos y promoción

- [Run de entrenamiento 73,5k→133,5k](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45082812702)
- [Run de holdout emparejado 133,2k](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45123727239)
- [Artefacto del holdout y checkpoint Transformer sincronizado](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45123727239/artifacts/14794737935)
- Checkpoint Lumen 133,2k: latest_good.pt preservado en el artefacto del run 45082812702 (ronin-lumen-step133500-v2-45082812702).
- Checkpoint Transformer 133,2k: step_133200.pt incluido en el artefacto del run 45123727239.

**Promoción:** el par de checkpoints de step 133.200 queda fijado como referencia experimental actual de Lumen-Ronin frente al baseline Transformer. La promoción es documental y de referencia reproducible; no afirma una victoria universal ni sustituye futuras réplicas independientes.
