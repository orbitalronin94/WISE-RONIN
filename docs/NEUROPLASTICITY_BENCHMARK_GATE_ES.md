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

Las señales comunes se calculan una sola vez por paso y se exponen mediante un contexto inmutable: error/sorpresa, actividad efectiva por unidad, elegibilidad, incertidumbre, señal de mejora/regresión y presupuesto disponible. Los órganos consumen esas señales; no mantienen copias ni vuelven a calcularlas. Las actualizaciones específicas siguen siendo independientes y cada señal incluye productor, forma, dtype y contador de paso para detectar datos obsoletos.

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
