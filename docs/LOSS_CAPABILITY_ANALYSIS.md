# Auditoría: coste computacional y capacidades según la loss

**Estado: investigación abierta; no se autoriza inferir capacidad a partir de una loss aislada.**

Esta auditoría separa dos preguntas que deben estudiarse juntas:

1. ¿Qué operaciones cuestan tiempo y pueden fusionarse o eliminarse sin cambiar la función?
2. ¿Qué capacidades se conservan, aparecen o se degradan cuando la loss mejora?

## Evidencia histórica inicial

### Holdout congelado de 2.431 preguntas

Los dos informes usan el mismo holdout español v2+personality:

| Checkpoint LUMEN | Aciertos | Accuracy | NLL media |
|---|---:|---:|---:|
| step 219.700 | 831/2.431 | 34,18 % | 5,7441 |
| step 400.000 | 801/2.431 | 32,95 % | 5,5620 |

Entre ambos checkpoints:
- La NLL media mejora 0,1821 (aprox. 3,17 %).
- La exactitud top-1 cae 30 respuestas (−1,23 puntos porcentuales).

Esto es una señal importante: **mejor NLL no equivale automáticamente a más respuestas correctas**. El modelo puede mejorar la probabilidad media asignada a las respuestas evaluadas y, al mismo tiempo, cambiar el orden de sus opciones de forma que acierte menos veces. No llamamos a esto calibración sin medir ECE/Brier y curvas de confianza.

Comparación emparejada en step 400.000:
- LUMEN: 801/2.431 (32,95 %), NLL 5,5620.
- Transformer: 821/2.431 (33,77 %), NLL 5,3209.
- LUMEN queda 20 aciertos por detrás y con NLL 0,2410 mayor.

En step 219.700/219.750, LUMEN había obtenido 831 aciertos frente a 805 del Transformer, aunque su NLL también era mayor (5,7441 frente a 5,5147). El cambio de ventaja top-1 entre checkpoints merece análisis por categoría y por pregunta, no una conclusión global basada solo en la métrica agregada.

Fuentes:
- Holdout 219.700/219.750: run 45257284512.
- Holdout step 400.000: run 45379135782.
- Ambos evaluaron las mismas 2.431 preguntas congeladas. La comparación entre checkpoints no demuestra causalidad: hay que comprobar lineage, corpus, tokenizer, scheduler y perfil de entrenamiento antes de atribuir el cambio a un mecanismo concreto.

### La loss por minibatch es muy ruidosa

Logs de entrenamiento de LUMEN muestran ejemplos en los que la loss de un único minibatch cae y vuelve a subir rápidamente:
- step 239.000: loss 1,13084; el promedio de la ventana usado por glucosa era 2,900237.
- step 239.200: loss 3,45369; ventana 2,961913.
- step 239.300: loss 4,33311; ventana 3,055522.
- step 329.700: loss 1,07981; ventana 2,672499.

Por tanto, el mínimo de un paso no es un umbral fiable para declarar convergencia ni para decidir que el modelo ya no necesita actualizar pesos. El trainer ya calcula una media de ventana para la señal de glucosa/dolor/placer, pero el surprise gate actual y los eventos `time_to_target` se basan en la loss escalar del minibatch. Esa diferencia semántica debe ser visible y medida.

## Qué significa realmente la loss actual

`train/loss.py` combina:
- token cross-entropy;
- JEPA latente (peso 0,3);
- diversidad de activaciones (peso 0,01);
- router (peso 0,01);
- repetición de triples (peso 0,05);
- complejidad opcional.

La loss total no es idéntica a la token loss. Los umbrales de gate y los hitos deben indicar expresamente qué señal usan. En particular, una loss total baja puede deberse parcialmente a términos auxiliares; no basta para afirmar que ha mejorado la predicción del siguiente token.

## Hipótesis que se van a contrastar

1. **Fluidez frente a exactitud:** al bajar NLL, ¿mejoran continuidad y selección probabilística pero no necesariamente la respuesta top-1?
2. **Memorización frente a transferencia:** ¿las mejoras se concentran en ejemplos cercanos al corpus o se extienden a categorías y fuentes no vistas?
3. **Estabilidad frente a mínimos locales:** ¿las mejoras de ventana sobreviven a varios bloques, cambios de semilla y evaluación sin estado mutable?
4. **Especialización:** ¿qué categorías mejoran o empeoran en cada tramo de loss (razonamiento, factualidad, matemáticas, instrucciones, estilo, EOS, no repetición)?
5. **Órganos:** ¿qué señales acompañan a una mejora sostenida: escritura Delta, homeostasis efectiva, memoria lenta, sueño, glucosa, dolor y placer?
6. **Confianza:** ¿mejora la NLL a costa de concentrar masa en opciones equivocadas? Medir ECE, Brier, margen top-1/top-2 y NLL por categoría.
7. **Eficiencia:** ¿se puede conservar cada fortaleza con menos backward, menos sincronizaciones Python/PyTorch y operaciones matemáticas compartidas?

## Instrumentación necesaria

Cada evaluación de checkpoint debe producir una fila con:
- step, SHA de checkpoint, corpus, tokenizer, perfil y seed;
- loss total y cada componente por ventanas (media, mediana, p90 y desviación);
- token NLL y accuracy top-1 en holdout fijo;
- exactitud y NLL por categoría, más matriz de cambios por pregunta;
- EOS, longitud, repetición, diversidad, consistencia y formato;
- ECE/Brier/margen si el evaluador expone probabilidades;
- actividad/uso de Delta, homeostasis, memoria, escritura y sueño;
- coste por paso: forward, backward, optimizer, órganos, Python overhead y sincronizaciones.

No mezclar el holdout de 2.431 preguntas con probes de generación libre ni con loss de entrenamiento. Cada familia de evaluación conserva su propia métrica y su versión congelada.

## Prioridades de auditoría de código

- Evitar `.item()`/sincronizaciones host dentro del camino caliente salvo donde sean imprescindibles.
- Mover contadores que solo se consultan cada N pasos fuera del forward de cada batch.
- Reutilizar cálculos matemáticos comunes solo con paridad de salida y gradientes.
- Vectorizar escrituras de memoria elástica y evitar bucles por muestra cuando sea equivalente.
- Medir coste/beneficio de AdaptiveChunking y de cada refinamiento Delta.
- Verificar si máscaras, proyecciones y normalizaciones se pueden compartir entre órganos/capas.
- Separar tiempo de preparación de corpus/tokenización del tiempo real de entrenamiento.


## Primer mapa de coste (medido + hipótesis de código)

En logs de entrenamiento LUMEN, una ventana de 100 pasos en el entorno CPU registró:
- forward: 4,467 s (18,8 %);
- backward: 15,792 s (66,5 %);
- overhead Python/otros: 3,483 s (14,7 %);
- total: 23,742 s.

El backward sigue siendo el bloque dominante. No obstante, casi 3,5 segundos por cada 100 pasos fuera de forward/backward justifican una auditoría seria de Python, logging, conversiones escalares y escrituras de estado.

Candidatos para perfilar en aislamiento, todavía no optimizaciones aprobadas:

1. **Memoria elástica:** escritura por bucle sobre muestras del batch; `registrar_nueva` usa `nonzero`, `.item()` y escrituras escalares. La actualización de decaimiento también consulta `tokens_since_decay.item()` en el camino de entrenamiento. Vectorizar y agrupar escrituras puede reducir overhead, pero debe preservar política de reemplazo y estado.
2. **Chunking adaptativo:** cada uno de los cuatro Delta órganos obtiene un tamaño mediante `surprise.mean().item()` y luego ejecuta pooling, interpolación, similitud y normalización. Es una ruta con coste real y una extracción escalar por órgano. Medir su contribución frente a una ruta identidad y perfiles de chunking fijo.
3. **Contadores sin consumidor frecuente:** `VivoEmbedding.forward` hace `bincount` en cada batch, aunque las propuestas de poda se consultan cada 1.000 pasos y no se aplican al tokenizer. `Homeostasis.m` también se actualiza cada batch pese a estar descrito como diagnóstico, mientras el control efectivo usa `effective_m`. Confirmar todos los consumidores y tests antes de reducir frecuencia o retirar trabajo.
4. **Sincronizaciones y diagnósticos:** Delta materializa normas con `.item()` aun cuando la colección de diagnósticos puede estar desactivada; la comprobación de modulación usa una rama Python basada en un escalar tensorial; `SignalContext.from_surprise` ejecuta una comprobación de finitud completa por forward. En CPU son conversiones/operaciones Python; en aceleradores podrían forzar sincronización host-device.
5. **Delta recurrente:** cada órgano calcula Gram de claves y consulta-clave con matrices B×T×T, y actualiza pesos recurrentes en cada refinamiento porque el decaimiento sí cambia entre refinamientos. No se debe declarar redundante ese bucle sin demostrar equivalencia; sí medir el coste por refinamiento (perfil actual de entrenamiento: dos).
6. **Loss:** cross-entropy y la pérdida de repetición calculan normalizaciones logarítmicas sobre logits. Existe un test de equivalencia de la pérdida de repetición, pero falta un benchmark que demuestre que compartir normalización con token CE mejora CPU sin aumentar memoria ni ralentizar kernels.
7. **Cabezas:** KAN y MLP se calculan siempre antes de mezclarse con el router. Aunque el router favorezca una rama, saltarse la otra cambia gradientes y puede cambiar comportamiento; medir primero utilización del router y coste por rama.

### Interpretación del gate actual

El gate `--surprise-gate-loss 1.8` decide con la loss total de un minibatch aislado. En los logs hay minibatches con loss 1,13 mientras la ventana de 100 pasos está cerca de 2,9. Así que el gate puede confundir un batch fácil con estabilidad del aprendizaje. Antes de reactivarlo en un bloque largo, comparar:
- gate por loss total instantánea;
- gate por media móvil de loss total;
- gate por token loss media;
- umbral con histéresis y mínimo de actualizaciones completas.

La comparación debe registrar cuántos backward/updates se evitan, qué categorías pierden capacidad y qué ocurre en holdout. No cambiar el comportamiento hasta medirlo.

## Criterio de aceptación

Ninguna optimización se activa en main por una mejora de throughput aislada. Debe:
1. pasar tests de equivalencia de outputs, estados y gradientes cuando corresponda;
2. mejorar throughput mediano en benchmarks repetidos y emparejados;
3. no degradar materialmente la loss por ventanas;
4. conservar o mejorar holdout y probes por categoría;
5. documentar cualquier cambio deliberado de comportamiento.

**Principio:** bajar la loss es una señal de aprendizaje; identificar qué capacidad mejora exige medir esa capacidad. La arquitectura debe reforzar fortalezas observadas, no perseguir un escalar a ciegas.
