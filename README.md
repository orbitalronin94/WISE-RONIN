# Lumen-Ronin
### Una arquitectura pequeña. Una pregunta enorme: ¿y si aprender no fuera solo escalar?

> **LUMEN ES UN PERRO.** Un perro de Pavlov que aprende lento, pero aprende. Que saliva, que tiene hambre, que le duele, que le da placer. Y un perro, aunque sea más lento que un loro para repetir, es más listo que un loro para aprender.

Lumen-Ronin es un experimento de arquitectura recurrente, CPU-first y de pocos millones de parámetros. En lugar de limitarse a apilar bloques Transformer y aumentar cómputo, explora otra pregunta: **¿pueden un estado asociativo, la homeostasis, un presupuesto energético y varias escalas de memoria mejorar la manera en que aprende un modelo pequeño?**

No presentamos una metáfora como un descubrimiento. Cada órgano es código, cada hipótesis debe medirse y cada resultado debe sobrevivir a un holdout. La épica está en la pregunta y en el trabajo de responderla, no en exagerar las conclusiones.

El repositorio conserva el nombre técnico **WISE-RONIN**. El proyecto activo es **Lumen-Ronin**; un Transformer pequeño permanece como baseline de control.

---

## Índice

- [Estado actual](#estado-actual)
- [La tesis: el perro y el loro](#la-tesis-el-perro-y-el-loro)
- [Qué estamos construyendo](#qué-estamos-construyendo)
- [Arquitectura actual](#arquitectura-actual)
- [La campaña de entrenamiento: de cero a 133,2k](#la-campaña-de-entrenamiento-de-cero-a-1332k)
- [Resultados emparejados](#resultados-emparejados)
- [Corpus y evaluación](#corpus-y-evaluación)
- [Cómo se investiga](#cómo-se-investiga)
- [Documentación y código](#documentación-y-código)
- [Límites y compromisos](#límites-y-compromisos)

## Estado actual

<sup>Actualizado: 23 de noviembre de 2026 · rama `main`</sup>

<details>
<summary><strong>La campaña activa</strong></summary>

**Último checkpoint emparejado evaluado: step 133.200**, sobre el corpus español v2. El holdout común de 2.431 preguntas está completado.

- Lumen: **810/2.431 (33,32 %)**, NLL **5,7221**.
- Transformer: 793/2.431 (32,62 %), NLL 5,8947.
- Lumen aventaja al baseline por 17 aciertos y obtiene una NLL 0,1726 menor.
- Desde 73,5k, Lumen suma 19 aciertos netos; Transformer suma 1.
- En el tramo 73,5k→133k, Transformer mantuvo menor training loss. El holdout, sin embargo, favorece a Lumen en accuracy y NLL.

**Estado operativo:** step 133.200 queda promovido como referencia experimental emparejada. El checkpoint Lumen procede del run de continuación interrumpido tras guardar 133.200; el checkpoint Transformer se avanzó exactamente 200 pasos desde 133.000 para igualar el corte. Ambos checkpoints y el resultado del holdout están preservados en Actions.

</details>

## La tesis: el perro y el loro

El loro representa la fluidez: producir secuencias deprisa y parecer competente por lo que repite. El perro representa otra apuesta: un sistema cuyo aprendizaje está regulado por señales internas, consecuencias, recursos y memoria.

En Lumen, esa metáfora tiene correspondencias técnicas concretas:

- **Glucosa:** reserva de energía interna que responde a mejoras verificadas de la loss.
- **Dolor:** señal acotada de regresión; reduce de forma acotada el multiplicador energético interno.
- **Placer:** señal independiente de mejora; eleva de forma moderada el multiplicador energético interno.
- **Homeostasis:** regula la actividad efectiva de las unidades.
- **Delta:** escribe en un estado asociativo cuando la sorpresa lo justifica.
- **Memoria:** separa adaptación elástica y registros lentos.

No son emociones humanas, no implican experiencia subjetiva y no convierten al modelo en un organismo. Son mecanismos computacionales inspirados en preguntas estudiadas por la neurociencia.

## Qué estamos construyendo

No buscamos demostrar que un modelo pequeño vaya a derrotar a los grandes por decreto. Investigamos si un sistema pequeño puede aprender de manera más estable y útil cuando su arquitectura incorpora algo más que capacidad bruta.

La hipótesis central es de ingeniería: quizá no todo el progreso dependa de multiplicar parámetros y operaciones. Quizá también importe cómo se asigna el presupuesto, cuándo se escribe en memoria, qué actividad se regula y cómo se conserva lo aprendido.

> **No esperaremos al hardware del futuro para estudiar principios del aprendizaje.** No necesitamos neuronas vivas, un chip neuromórfico ni un ordenador cuántico para experimentar con homeostasis, sorpresa, presupuestos y memoria. Podemos construir aproximaciones con CPU, código y modelos pequeños.

La biología tardó millones de años en descubrir ciertos trucos. Nosotros podemos estudiarlos con Python. Y cualquiera puede intentarlo.

## Arquitectura actual

La arquitectura activa se identifica en los checkpoints como `ronin_lumen_v3_2_parallel_delta`.

| Subsistema | Función |
|---|---|
| Embedding vivo | Vocabulario de 4.096 IDs, dimensión 256 y seguimiento de uso |
| Estado Delta ×4 | Memoria asociativa recurrente de rango 64, guiada por sorpresa |
| Homeostasis ×4 | Regulación de actividad efectiva y Top-K (16/32) |
| Presupuesto energético | Distribuye energía entre Delta, memoria y salida |
| Glucosa | Reserva energética acotada alimentada por mejoras de loss |
| Dolor y placer | Reguladores independientes de regresión y mejora |
| Memoria elástica/lenta | MLP rápido y banco de 128 slots |
| Cabeza de salida | Router entre una rama KAN spline y una MLP |
| Embedding/output | Pesos compartidos |

**Configuración de ejecución del bloque 61k→66k:** cuatro hilos intra-op, dos inter-op, MKLDNN activado, AdamW `foreach=false`, afinidad CPU fijada, Delta pre-normalization desactivada y dos refinamientos paralelos. El valor base de refinamientos en `CONFIG` es seis; el perfil de ejecución lo sobrescribe a dos.

La extensión dual-state ZE está disponible en el código, pero no se activa en el entrenamiento actual. El sueño/replay también está implementado, pero el workflow 61k→66k usa `--no-sleep`. No atribuimos a un checkpoint mecanismos que no ejecutó.

La descripción completa, con responsabilidades, fórmulas, parámetros y distinción entre código disponible y código activo, está en [Arquitectura actual](docs/ARQUITECTURA_ACTUAL_ES.md).

## La campaña de entrenamiento: de cero a 133,2k

El linaje emparejado se inició desde cero y avanzó mediante continuaciones verificadas. Cada bloque conserva checkpoints y estado de optimizador; los holdouts comparan ambos modelos sobre las mismas 2.431 preguntas.

| Tramo | Run de entrenamiento | Estado |
|---|---|---|
| 0 → 1k | [45012493289](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45012493289) | Completado; holdout [45013766761](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45013766761) |
| 1k → 11k | [45014470774](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45014470774) | Completado |
| 11k → 16k | [45021399600](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45021399600) | Completado; órganos afectivos integrados |
| 16k → 21k | [45025897082](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45025897082) | Completado |
| 21k → 26k | [45030262114](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45030262114) | Completado |
| 26k → 31k | [45034338841](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45034338841) | Completado |
| 31k → 41k | [45037602119](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45037602119) | Checkpoints guardados; gate final del workflow falló; holdout separado |
| 41k → 61k | [45044892658](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45044892658) | Completado y evaluado |
| 61k → 66k | [45061486739](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45061486739) | Completado; holdout 66k |
| 66k → 73,5k (corpus v2) | [45071952349](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45071952349) | Completado; holdout v2 de 2.431 preguntas |
| 73,5k → 133,5k (corpus v2) | [45082812702](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45082812702) | Transformer completó 133,5k; Lumen guardó 133,2k antes de la cancelación del job |

El primer piloto de 500 pasos detectó un colapso de escrituras en las capas superiores. No se oculta ni se suma al linaje actual: fue una prueba de arranque que ayudó a corregir la ruta residual y la migración homeostática.

La crónica completa explica qué cambió en cada tramo, qué runs fallaron después de producir checkpoints válidos y qué ramas fueron únicamente benchmarks: [Crónica experimental, step 0 → actualidad](docs/CRONICA_EXPERIMENTAL_ES.md).

## Resultados emparejados

Último holdout completo: **step 133.200**, run [45123727239](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45123727239).

| Modelo | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| Lumen-Ronin | **810/2.431** | **33,32 %** | **5,7221** |
| Transformer baseline | 793/2.431 | 32,62 % | 5,8947 |

En 133,2k Lumen supera al Transformer por 17 respuestas (+0,70 puntos porcentuales) y obtiene una NLL 0,1726 menor. Frente al holdout de 73,5k, Lumen pasa de 791 a 810 aciertos y reduce su NLL de 5,8320 a 5,7221; Transformer pasa de 792 a 793 aciertos y reduce su NLL de 6,4189 a 5,8947. Son resultados descriptivos de un holdout fijo: la diferencia debe confirmarse con nuevas semillas y evaluaciones independientes.

El resultado merece atención porque las curvas de training loss del tramo 73,5k→133k favorecieron al Transformer, mientras el holdout de 133,2k favorece a Lumen en ambas métricas. Esto muestra que la training loss por sí sola no predice el orden de rendimiento en este test; no demuestra por sí solo superioridad general ni identifica qué mecanismo causó la diferencia.

La serie histórica completa está en la [crónica experimental](docs/CRONICA_EXPERIMENTAL_ES.md). Las generaciones libres se documentan aparte porque no son comparables directamente con el test de opción múltiple.

## Corpus y evaluación

**Corpus v3 experimental:** ya está construido y congelado en la rama `corpus/lumen-ronin-spanish-v3`. Añade 575.558 registros netos (9.933.007 train en total), incorpora personalidad y currículos en castellano condicionados a la identidad Lumen-Ronin y conserva sin cambios validation, holdout y tokenizer de v2. El artefacto y sus límites de autorización están documentados en [Corpus v3](docs/CORPUS_V3_ES.md). **No se ha entrenado con v3:** la deduplicación global y la auditoría de contaminación siguen pendientes.

**Corpus v3.1 en preparación:** se ha abierto una rama separada para ampliar humanidades, ciencias humanas, matemáticas, lógica y medicina. El catálogo incluye filosofía, historia global, patrimonio cultural, jurisprudencia española, ética intercultural y ciencias sociales; también define la remediación de los near-duplicates y de las alertas detectadas en v3.0.0. Licencias no comerciales o todavía no verificadas quedan expresamente fuera de la admisión automática. Consulta [plan de corpus v3.1](docs/CORPUS_V3_1_ES.md). No existe aún un artefacto v3.1 ni autorización de entrenamiento.

Los tramos 66k→73,5k y 73,5k→133,2k utilizaron el corpus español v2 congelado, preparado con seed 42. El manifiesto registra 9.357.876 filas de entrenamiento, 1.413 de validación y 2.431 de holdout. La muestra determinista compartida de entrenamiento contiene 200.000 filas (20k v1, 72k CSIC y 108k TDX); tokenizer de 4.096 IDs, PAD=0 y EOS presente. Hash del tokenizer: `d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe`. CSIC y TDX se incorporaron bajo CC BY 4.0; CoWeSe quedó excluido por derechos de los textos de origen.

| Split | Filas |
|---|---:|
| Entrenamiento | 70.105 |
| Validación | 1.413 |
| Holdout | 2.431 |

El holdout contiene 1.931 preguntas SIQA-es y 500 OpenBookQA-es. Está separado del entrenamiento y no alimenta glucosa, dolor ni placer. La preparación aplica deduplicación, limpieza, filtros por componente y una política explícita de licencias.

La procedencia, fuentes, manifiesto, hashes y estado de la ampliación están en [data/README.md](data/README.md). La ampliación se aplicará por igual a Lumen y Transformer; ningún modelo volverá a entrenar con v1 mientras esté pendiente la preparación y auditoría de v2.

## Cómo se investiga

El proyecto no acepta una mejora solo porque baje la loss o porque un workflow se llame `fast`.

1. **Mismo punto de partida:** comparar checkpoints, corpus, tokenizer, seed, batch y contexto.
2. **Medir por separado:** accuracy, NLL, loss, throughput, estabilidad y comportamiento generativo.
3. **Ablaciones explícitas:** cambiar un mecanismo cada vez y conservar una ruta de referencia.
4. **Preservar órganos:** no desactivar un órgano como atajo de rendimiento sin un experimento funcional separado.
5. **Verificar checkpoints:** pesos, buffers, RNG, optimizador, scheduler, hashes y paso real.
6. **Publicar los fallos:** distinguir un job fallido de un checkpoint inválido; no ocultar reintentos, cancelaciones ni duplicados.

Los microbenchmarks de 100 o 500 pasos sirven para filtrar hipótesis de entrenamiento y velocidad. No sustituyen al holdout ni prueban generalización.

## Documentación y código

### Lectura recomendada

- [Arquitectura actual](docs/ARQUITECTURA_ACTUAL_ES.md) — anatomía del modelo y configuración realmente activa.
- [Crónica experimental](docs/CRONICA_EXPERIMENTAL_ES.md) — linaje desde cero, cada continuación y sus holdouts.
- [Neurociencia y decisiones de arquitectura](docs/NEUROCIENCIA_Y_ARQUITECTURA_ES.md) — referencias, analogías y límites.
- [Glucosa, dolor y placer](docs/PAIN_PLEASURE_ORGANS_ES.md) — evolución de los reguladores afectivos.
- [Experimento de neuroplasticidad](docs/EXPERIMENTO_NEUROPLASTICIDAD_ES.md) — rama aislada, chunking adaptativo, trazas de elegibilidad y protocolo Lumen vs Transformer desde cero.
- [Documento maestro V1](DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md) — especificación de diseño.
- [Especificación V1](SPEC_RONIN_LUMEN_V1.md) — contrato arquitectónico.
- [Índice de documentación](docs/README.md).

### Código principal

- [Modelo integrado](model/lumen.py)
- [Estado Delta](model/delta_state.py)
- [Homeostasis](model/homeostasis.py)
- [Memoria](model/memory.py)
- [Presupuesto y glucosa](model/energy.py)
- [Dolor](model/pain.py)
- [Placer](model/pleasure.py)
- [Router y cabezas KAN/MLP](model/heads.py)
- [Pérdidas](train/loss.py)
- [Entrenador y checkpoints](train/train.py)
- [Baseline Transformer](train/transformer_baseline.py)

## Límites y compromisos

- Lumen no es un organismo ni una prueba de conciencia.
- Las analogías biológicas no demuestran equivalencia con procesos neuronales.
- Un holdout de 2.431 preguntas no representa toda la inteligencia ni todos los dominios.
- La NLL y la accuracy deben leerse por separado.
- Un benchmark corto no demuestra una mejora sostenida.
- El código opcional no se considera activo si el workflow no lo habilita.
- Los resultados negativos, fallos y retrocesos forman parte de la investigación.

**La promesa de Lumen no es que ya haya resuelto el aprendizaje. Es que se atreve a formular otra arquitectura, exponerla a controles y dejar que los resultados decidan.**