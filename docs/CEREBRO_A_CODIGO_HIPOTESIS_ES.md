# Cerebro a Código: hipótesis falsables para Lumen-Ronin

**Estado: agenda experimental; no son mecanismos promovidos ni resultados del modelo.**  
**Regla de admisión:** referencia comprobable, implementación ejecutable, pruebas de equivalencia cuando corresponda, ablación emparejada y holdout intacto. Una analogía biológica no constituye evidencia de mejora en Lumen.

## Correcciones bibliográficas y de alcance

Las seis entradas propuestas contienen cinco líneas conceptuales: R-STDP y el modelo de Izhikevich describen esencialmente la misma familia de aprendizaje de tres factores; se mantienen ambas referencias porque una es una revisión teórica y la otra un modelo de red spiking.

- Frémaux y Gerstner (2016), no «Gerstner, Frémaux y Urban»: [Neuromodulated Spike-Timing-Dependent Plasticity, and Theory of Three-Factor Learning Rules](https://doi.org/10.3389/fncir.2015.00085), *Frontiers in Neural Circuits* 9:85. DOI verificado.
- Redondo et al. (2010): [Synaptic tagging and capture](https://doi.org/10.1523/JNEUROSCI.3140-09.2010), *Journal of Neuroscience* 30(14):4981–4989. DOI verificado.
- Huang, Zhu y Siew (2006): [Extreme learning machine: Theory and applications](https://doi.org/10.1016/j.neucom.2005.12.126), *Neurocomputing* 70:489–501. DOI verificado.
- Lee, O'Doherty y Shimojo (2015): [Neural computations mediating one-shot learning in the human brain](https://doi.org/10.1371/journal.pbio.1002137), *PLOS Biology* 13(4):e1002137. DOI verificado.
- Izhikevich (2007): [Solving the Distal Reward Problem through Linkage of STDP and Dopamine Signaling](https://doi.org/10.1093/cercor/bhl152), *Cerebral Cortex* 17(10):2443–2452. DOI verificado.
- Nicola y Clopath (2019), no Savtchenko et al.: [A diversity of interneurons and Hebbian plasticity facilitate rapid compressible learning in the hippocampus](https://doi.org/10.1038/s41593-019-0415-2), *Nature Neuroscience* 22:1168–1181. **Corrección importante:** el DOI propuesto como `...0415-5` no coincide; el DOI de la publicación localizada termina en `0415-2`.

## Inventario del código actual

Inspección del código en la rama de integración:

- `model/delta_state.py` ya conserva `eligibility_trace` y `eligibility_modulator` como buffers persistentes; la traza se actualiza sin gradiente y el modulador se limita a [-1, 1] y se escala.
- El modulador previo se aplica al estado Delta de la siguiente pasada. El trainer guarda la política de traza persistente y registra `pleasure_minus_pain` como señal retardada.
- `model/homeostasis.py` ya mide actividad efectiva después de ReLU/Top-K, actualiza una EMA por unidad y ajusta sesgos con una meta `m_target`.
- El trainer ya dispone de replay/sueño, pero su presencia en el código no demuestra que se haya ejecutado en todos los experimentos. Los flags del workflow mandan.
- No se ha encontrado una implementación ELM ni una política hipocampal one-shot gobernada por incertidumbre en el camino de entrenamiento inspeccionado.

Esto significa que R-STDP y homeostasis no son ideas que haya que añadir desde cero: la investigación debe probar si la señal, la escala temporal y el acoplamiento actuales están bien definidos y aportan valor.

## 1. Regla de tres factores: elegibilidad × modulador

**Referencia:** Frémaux & Gerstner (2016), DOI 10.3389/fncir.2015.00085.  
**Modelo computacional mínimo:**

`e[t+1] = lambda_e * e[t] + h[t]`  
`delta_w[t] = eta * M[t] * e[t]`

`h` representa coincidencia/actividad local; `M` es un modulador global retardado, que puede codificar mejora, regresión, sorpresa o recompensa según una hipótesis explícita.

**Conexión existente:** Delta ya acumula una traza y recibe un modulador retardado relacionado con placer menos dolor. No afirmar que esto sea STDP biológico: no hay spikes pre/post ni kernel temporal STDP explícito en el modelo rate-based actual.

**Experimento falsable:**
1. Comparar señal actual `pleasure - pain` con modulador cero, signo invertido y una señal de mejora normalizada.
2. Mantener idénticos checkpoint, lotes, seed, tokens y holdout.
3. Medir NLL/accuracy, norma y saturación de traza, cambios de estado, estabilidad y coste CPU.
4. Probar retardos artificiales y decaimientos distintos; el modulador tardío debe perder influencia cuando la traza se extingue.
5. Verificar que M=0 implica delta de actualización exactamente cero en el mecanismo aislado.

**Criterio de rechazo:** si el mecanismo no mejora holdout o empeora estabilidad/tiempo frente a M=0, no se promueve.

## 2. Synaptic tagging & capture: tag local y recurso global

**Referencia:** Redondo et al. (2010), DOI 10.1523/JNEUROSCI.3140-09.2010.  
**Abstracción experimental:** un evento local deja un tag con decaimiento; un evento global habilita recursos de consolidación; solo la coincidencia dentro de ventana permite captura.

`tag[t+1] = lambda_tag * tag[t] + local_event[t]`  
`capture[t] = (tag[t] > threshold) AND global_resource[t]`

**Conexión propuesta, aún no demostrada:** el replay/sueño podría proporcionar una señal global de consolidación. Pero el replay actual no es automáticamente PRP ni reproduce la biología de síntesis proteica. Primero hay que medir qué parámetros se consolidan y si existe una selección local persistente.

**Experimento falsable:** etiquetar actualizaciones candidatas, generar replay fuerte y débil en ventanas controladas, y comparar con replay sin tags y tags sin replay. Medir retención después de entrenamiento adicional y transferencia a holdout, además de interferencia sobre tareas anteriores.

**Criterio de rechazo:** si el replay con tags no mejora retención frente a replay simple o aumenta interferencia, no se añade complejidad.

## 3. Extreme Learning Machine (ELM): proyección fija y lectura analítica

**Referencia:** Huang, Zhu & Siew (2006), DOI 10.1016/j.neucom.2005.12.126.  
**Mecanismo:** proyección oculta aleatoria fija `H = g(X W + b)`; ajuste analítico de salida `beta = pinv(H) T`.

**Límite de inferencia:** ELM demuestra una alternativa de entrenamiento para redes feed-forward de una capa oculta bajo sus condiciones; no demuestra que Delta, STDP o Lumen puedan sustituir backprop global en un modelo autoregresivo. Tampoco es una justificación teórica directa de la arquitectura Lumen.

**Experimento separado, no invasivo:** añadir un pequeño probe ELM sobre representaciones congeladas de Lumen y Transformer. Entrenar solo el readout analítico y comparar regresión/clasificación de probes con un readout lineal ridge. No modificar pesos del modelo ni el pipeline de lenguaje.

**Criterio de rechazo:** si el probe no supera ridge o no aporta una medición útil, queda como análisis auxiliar, no como órgano.

## 4. One-shot condicionado por incertidumbre

**Referencia:** Lee, O'Doherty & Shimojo (2015), DOI 10.1371/journal.pbio.1002137.  
**Idea traducible:** una estimación de incertidumbre causal puede modular la tasa de aprendizaje, con régimen incremental y régimen rápido.

`alpha[u] = alpha_base + gate(u) * (alpha_fast - alpha_base)`

El paper estudia una tarea de aprendizaje humano y evidencia de participación hipocampal/prefrontal; no proporciona un algoritmo universal ni autoriza a fijar `alpha=1` para gradientes de un LLM.

**Experimento seguro:** no tocar el optimizador global. Crear un banco asociativo pequeño, con tasa local limitada, y comparar tasa fija frente a tasa adaptada por incertidumbre calibrada. Evaluar retención, interferencia, calibración de incertidumbre y casos adversariales.

**Criterio de rechazo:** si la incertidumbre alta coincide con ejemplos ruidosos/adversariales o causa interferencia, desactivar el modo rápido.

## 5. Izhikevich: recompensa distal en redes spiking

**Referencia:** Izhikevich (2007), DOI 10.1093/cercor/bhl152.  
**Aportación específica:** un modelo de red de neuronas spiking combina STDP y dopamina para resolver el crédito de recompensas que llegan segundos después.

**Relación con la entrada 1:** es una formulación concreta de la familia de tres factores, no un mecanismo independiente que deba duplicarse en Lumen.

**Experimento posible:** usarlo como referencia matemática para validar retardos y decaimiento de la traza actual. No afirmar equivalencia neuronal sin dinámica de spikes. Si se implementa una versión spiking, debe vivir en un experimento separado y compararse con la versión rate-based.

## 6. Interneuronas, diversidad y aprendizaje comprimido

**Referencia corregida:** Nicola & Clopath (2019), DOI 10.1038/s41593-019-0415-2.  
**Hallazgo relevante:** el trabajo modela redes de interneuronas que organizan secuencias, compresión y replay de ensamblajes en el hipocampo. No establece una regla general de «90 % inhibidas» ni que un 10–20 % de unidades muertas sea el óptimo de una red de lenguaje.

**Conexión existente:** Lumen ya aplica Top-K y homeostasis sobre la actividad efectiva post-gate. La métrica relevante no es solo el porcentaje global de unidades inactivas: hay que medir distribución de uso, concentración, persistencia temporal, diversidad entre contextos y recuperación de unidades.

**Experimento falsable:** barrido de `m_target` y `k_top` en ramas aisladas. Informar:
- fracción de unidades sin actividad efectiva sostenida;
- entropía y Gini de utilización;
- proporción de unidades que se reactivan en ventanas posteriores;
- NLL/accuracy y pérdida de conocimiento previo;
- coste de entrenamiento.

**Criterio de rechazo:** no fijar 10–20 % como objetivo a priori. Seleccionar solo si un rango mejora métricas de generalización sin colapso ni concentración patológica.

## Orden de trabajo recomendado

1. **Auditar y medir lo que ya existe:** elegibilidad, señal moduladora, homeostasis y replay; añadir métricas y pruebas aisladas.
2. **Experimento R-STDP:** ablación del modulador y barrido de decaimiento/retardo.
3. **Experimento tagging/replay:** tags locales × replay global, con control factorial.
4. **Barrido homeostático:** sparsity efectiva, diversidad y reactivación, sin imponer porcentaje biológico.
5. **One-shot:** solo en memoria auxiliar, con incertidumbre calibrada y límites de actualización.
6. **ELM:** probe offline sobre representaciones congeladas; no altera el trainer.

## Especificación mínima de pruebas

Los tests de mecanismo deben comprobar propiedades, no solo que el código termina:

- modulador cero ⇒ actualización elegible cero;
- traza decae monótonamente para `0 < lambda < 1`;
- tag sin recurso global ⇒ no hay captura;
- recurso global sin tag ⇒ no hay captura;
- incertidumbre baja ⇒ tasa base; alta ⇒ tasa rápida acotada;
- máscara inhibida ⇒ actualización local cero;
- salida ELM reproducible con seed fijo y readout analítico sin modificar la proyección.

Los teoremas Lean deben formalizar estas propiedades algebraicas pequeñas con tipos y supuestos explícitos. No se incluirán teoremas triviales que solo repitan una definición ni se afirmará que un esqueleto `True := trivial` formaliza el algoritmo completo. La implementación NumPy/PyTorch debe tener tests ejecutables independientes.

## Referencias

Todas las referencias y DOI de esta ficha se contrastaron con las páginas editoriales o PubMed enlazadas arriba. La corrección bibliográfica de Nicola & Clopath es deliberada: el DOI que termina en `0415-5` no corresponde al artículo localizado; el correcto termina en `0415-2`.
