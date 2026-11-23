# Gate de benchmark por órgano neuroplástico

## Objetivo

La integración de los órganos de Lumen-Ronin se acepta únicamente cuando cada órgano supera una comparación emparejada de 100 actualizaciones frente al control. El objetivo es mejorar simultáneamente el coste de ejecución y la loss, no intercambiar una métrica por otra.

## Diseño obligatorio por órgano

Para cada órgano se ejecutan un control y una candidata desde el mismo checkpoint verificado. Deben coincidir corpus y hash, tokenizer y hash, secuencia de lotes, semilla, batch, contexto, optimizador, scheduler, número de hilos, afinidad, perfil CPU y estado inicial de todos los órganos. El control y la candidata deben consumir exactamente los mismos 100 minibatches.

La medición se divide en cinco ventanas consecutivas de 20 pasos. Se registran:
- tiempo de pared de los 100 pasos y pasos/segundo;
- tiempo de forward, backward y mantenimiento del órgano, si el perfil permite medirlos sin introducir overhead desigual;
- loss media de cada ventana, media global, desviación entre ventanas y loss final;
- normas de pesos/gradientes/estados, actividad efectiva, NaN/Inf y contadores propios del órgano;
- hashes de configuración, checkpoint, corpus, tokenizer y secuencia de minibatches.

## Regla de aceptación

Una candidata solo supera el gate si cumple todo lo siguiente frente al control emparejado:
1. tiempo total de los 100 pasos estrictamente menor;
2. loss media de las cinco ventanas estrictamente menor;
3. loss de la última ventana no empeora;
4. no aparecen NaN/Inf, explosión de normas, colapso de actividad ni pérdida de estado persistente;
5. las pruebas de propiedades del órgano y de equivalencia de las señales compartidas pasan.

Los resultados se informan con diferencias absolutas y porcentuales. No se redondean antes de decidir el gate. Una mejora de throughput con loss peor, o una loss menor con más tiempo, es un resultado no elegible.

## Repetición y búsqueda

Cada órgano se prueba de forma aislada primero. Se repiten perfiles/hiperparámetros candidatos de forma determinista, cambiando una hipótesis identificable cada vez. Cada intento conserva su JSON y logs aunque falle. El baseline se vuelve a medir cuando cambia el runner o el entorno, y nunca se compara una candidata con un control de otro checkpoint o de otra carga de datos.

Para evitar una búsqueda infinita, la campaña debe declarar un presupuesto máximo de variantes por órgano y permitir reanudarla. Si ninguna variante supera el gate dentro del presupuesto, el órgano queda implementado pero desactivado/no promovido; no se rebaja el criterio para forzar un ganador. El informe debe decir explícitamente «sin candidata elegible».

## Compartición de cálculos

Las señales compartidas se calculan una sola vez y cada estado conserva un único productor. En esta implementación, `SignalContext` transporta sorpresa e incertidumbre por forward; `Presupuesto.check_loss_signal` produce una sola señal de mejora/regresión que consumen glucosa, dolor y placer. La actividad efectiva sigue siendo propiedad de cada órgano homeostático y la elegibilidad sigue siendo estado local de cada Delta: no se duplican en el contexto si no hay varios consumidores que necesiten la misma lectura. Las actualizaciones específicas siguen siendo independientes. Cualquier ampliación del contexto debe conservar productor, forma, dtype y contador de paso para detectar datos obsoletos.

Se incluyen pruebas que verifican:
- una única producción de cada señal por paso;
- igualdad numérica entre el cálculo de referencia y el cálculo compartido;
- ausencia de mutaciones por consumidores;
- comportamiento correcto ante señales ausentes, NaN/Inf y estados reiniciados;
- coste de construcción/consumo del contexto contabilizado en el tiempo total.

## Campaña por familias

Se exige un benchmark individual de 100 pasos para cada una de las doce familias descritas en `docs/CEREBRO_A_CODIGO_HIPOTESIS_ES.md`: elegibilidad/modulación, tagging/captura, ELM, one-shot por incertidumbre, recompensa distal, diversidad/Top-K, escalado homeostático, umbral intrínseco, poda, sueño/renormalización, replay secuencial comprimido y LTD protegida.

ELM se mide como probe offline sobre representaciones congeladas: sus métricas de tiempo y calidad se informan aparte y no se presentan falsamente como 100 pasos de entrenamiento autoregresivo. Los mecanismos periódicos (sueño, replay, poda y LTD) deben incluir su coste amortizado dentro del bloque medido; no se permite excluir su mantenimiento del tiempo total.

Tras superar los gates individuales se ejecuta una prueba integrada de 100 pasos para detectar interferencias y doble cómputo. El piloto integrado de 1.000 pasos y cualquier holdout son etapas posteriores, no sustitutos de estos gates.

## Integridad experimental

El benchmark de 100 pasos es un filtro de ingeniería, no evidencia de generalización. No se usa el holdout para ajustar parámetros. Los resultados se conservan incluso si no hay ganador. No se promueve ni se modifica el checkpoint de referencia durante esta campaña.


## Torneo de 18 cambios: comparación directa contra main

Workflow: `.github/workflows/benchmark-neuroplasticity-18.yml`.

Cada candidato dispone de un runner independiente. Dentro de ese runner se ejecutan control y candidato en orden alterno según el identificador, para equilibrar el calentamiento de caché del sistema. Ambos parten del mismo checkpoint promovido de paso 155.700, mismo corpus v2 + personalidad de 250k, tokenizer/cache, semilla 42, estado RNG, batch 8 y contexto 128. El perfil CPU del control se lee directamente del fichero versionado en el `main` congelado (`80a3354cd462d0969915ca745a46babd59729f86`); la candidata hereda ese perfil y añade un solo delta funcional sobre la base de señales compartidas de esta PR. No se usa holdout.

La matriz inicial contiene:

1. `01_shared_signal_core`: productor compartido de señales en perfil neutro.
2. `02_pain_response`: respuesta del órgano dolor.
3. `03_pleasure_response`: respuesta del órgano placer.
4. `04_pain_budget_gain`: modulación del presupuesto por dolor.
5. `05_pleasure_budget_gain`: modulación del presupuesto por placer.
6. `06_ltp_threshold`: umbral de escritura/LTP.
7. `07_disinhibition_gain`: ganancia de desinhibición.
8. `08_modulator_gain`: ganancia del modulador de tercer factor.
9. `09_eligibility_window`: retención temporal de elegibilidad.
10. `10_attention_gate`: ganancia atencional dependiente de sorpresa.
11. `11_tagging_capture`: tagging/captura.
12. `12_intrinsic_threshold`: umbral intrínseco adaptativo.
13. `13_simulated_pruning`: máscara de poda no destructiva.
14. `14_tag_protected_ltd`: LTD condicionada por recompensa negativa y tags.
15. `15_one_shot_memory`: escritura one-shot acotada en memoria.
16. `16_homeostatic_scaling`: escalado homeostático multiplicativo por canal.
17. `17_prioritized_replay`: replay priorizado/comprimido; se fuerza el ciclo de sueño dentro del bloque para medirlo.
18. `18_sleep_downscaling`: downscaling durante sueño; se fuerza el ciclo dentro del bloque para medirlo.

El gate de promoción de esta campaña exige simultáneamente: loss media de 100 pasos menor, pasos/segundo mayor, última ventana de 20 pasos no peor, dead máximo no mayor y ausencia de NaN/Inf. Los empates no pasan. La velocidad se calcula a partir de la suma de los tiempos `profile_1` del trainer (100 pasos), excluyendo arranque, carga de corpus y carga del modelo; el tiempo de sueño queda incluido en el tiempo de paso. Cada informe conserva los SHA exactos de `main` y de la candidata. Cada candidato conserva `main-control.log`, `candidate.log`, perfiles y `comparison.json` en su artefacto.

### Aprobados y pendientes

El workflow agrega automáticamente `AGGREGATE.md` y `approved_candidates.json`. Incluso si falta un artefacto, los 18 candidatos quedan contabilizados: los que no tengan resultado se anotan como pendientes por ejecución ausente. Solo los aprobados se incluyen en el perfil combinado para el siguiente entrenamiento. Los demás permanecen pendientes con su motivo y logs; no se borran ni se etiquetan como descartados. Si ninguno pasa, no se activa ningún candidato. Si pasan uno o más y no hay conflictos de configuración, el mismo workflow lanza una continuación de 1.000 pasos desde 155.700 con exclusivamente esos cambios aprobados; si no pasa ninguno, esa etapa queda omitida.

**Dependencia compartida:** el productor común de señales forma parte de todos los candidatos de esta PR; el candidato 01 mide su coste/beneficio en neutro. Por tanto, la aprobación de un órgano acredita el paquete «base compartida + delta del órgano», no una atribución causal aislada del delta respecto a la base.

ELM sigue fuera de este torneo de entrenamiento: es un probe analítico offline y no modifica el camino autoregresivo. Se evaluará con un protocolo auxiliar propio, sin fingir que puede bajar la loss del trainer.
