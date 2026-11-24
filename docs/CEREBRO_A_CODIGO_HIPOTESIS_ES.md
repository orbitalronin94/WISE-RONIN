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


## Segunda tanda: homeostasis, poda, sueño, replay y olvido (hipótesis 7–12)

**Estado:** propuestas de investigación. Ninguna de estas entradas debe interpretarse como descripción ya verificada del comportamiento interno de Lumen. Las métricas de entrenamiento solo permiten una interpretación después de comprobar qué tensor, operación y escala temporal representan.

### 7. Escalado sináptico homeostático

**Referencia:** Turrigiano (2008), [The Self-Tuning Neuron: Synaptic Scaling of Excitatory Synapses](https://doi.org/10.1016/j.cell.2008.10.008), *Cell* 135(3):422–435.

El trabajo describe ajustes coordinados de la fuerza de las sinapsis excitatorias para estabilizar la actividad neuronal. La intuición computacional —separar cambios relativos específicos de una normalización global de magnitud— merece una prueba en Lumen.

Una regla simplificada sería:

`w_i' = c w_i,` con `c = clip(r_target / max(r_actual, eps), c_min, c_max)`.

Si el mismo factor positivo se aplica a todos los pesos del conjunto, conserva sus proporciones; no implica que la regla sea estable en una red profunda ni que la tasa de activación del modelo sea equivalente a una tasa de disparo biológica.

**Matices y riesgos:**
- El clipping del factor evita cambios multiplicativos extremos, pero no garantiza estabilidad del sistema acoplado al optimizador.
- La fórmula propuesta puede amplificar pesos cuando la actividad es baja; debe probarse con cotas, regularización y actividad mínima.
- En el paper, la homeostasis se estudia en sinapsis excitatorias y en escalas biológicas; no justifica aplicar el factor a todos los parámetros de un LLM en cada minibatch.
- Un valor alto de `homeo_out` en una capa no demuestra que esa capa esté «trabajando más» ni que el valor sea una escala sináptica. Hay que rastrear su definición y unidades en el código.

**Experimento falsable:** comparar homeostasis actual, escalado multiplicativo por bloque, escalado por canal y control sin escalado. Aplicarlo en intervalos lentos y barrer tasa objetivo y límites del factor. Medir NLL/accuracy, normas por bloque, distribución de activaciones, gradientes, estabilidad y tiempo. Controlar que no se altere el presupuesto efectivo de aprendizaje de forma inadvertida.

**Criterio de rechazo:** cualquier variante que mejore una métrica interna pero empeore holdout, diversidad de activación o estabilidad queda descartada.

### 8. Homeostasis intrínseca: umbral adaptativo

**Referencias:** Ibata, Sun & Turrigiano (2008), [Rapid Synaptic Scaling Induced by Changes in Postsynaptic Firing](https://doi.org/10.1016/j.neuron.2008.02.031); Desai, Rutherford & Turrigiano (1999), [Plasticity in the Intrinsic Excitability of Cortical Pyramidal Neurons](https://doi.org/10.1038/9165).

Los trabajos muestran formas de ajuste homeostático de la fuerza sináptica y de la excitabilidad intrínseca. Una abstracción de umbral móvil podría ser:

`theta[t+1] = clip(theta[t] + eta_theta * (activity[t] - target), theta_min, theta_max)`.

**Conexión con Lumen:** no se debe identificar automáticamente `gate+` con un umbral neuronal ni interpretar una curva plana como estabilización. Hay que localizar qué parámetro cambia realmente, dónde se aplica y cómo se relaciona con la actividad post-gate.

**Experimento falsable:** añadir un umbral adaptativo en una rama experimental, separado de los sesgos homeostáticos existentes. Comparar umbral fijo, adaptación lenta y adaptación rápida. Medir actividad efectiva por unidad, porcentaje de unidades persistentemente inactivas, reactivación, entropía/Gini, NLL y retención.

**Riesgo principal:** la actualización lineal puede oscilar o empujar el umbral fuera de rango. Requiere límites, tasa pequeña y pruebas de estabilidad. Un objetivo de actividad cero no es aceptable.

**Criterio de rechazo:** si la adaptación reduce la actividad hasta crear unidades permanentemente inactivas o perjudica holdout, se descarta.

### 9. Poda selectiva

**Referencia principal:** Chechik, Meilijson & Ruppin (1998), [Synaptic Pruning in Development: A Computational Account](https://doi.org/10.1162/089976698300017124), *Neural Computation* 10(7):1759–1777.

La referencia computacional estudia crecimiento seguido de eliminación selectiva bajo restricciones; no demuestra que una poda por umbral simple mejore automáticamente redes neuronales modernas. La regla propuesta —eliminar conexiones débiles y poco usadas— es una hipótesis de ingeniería, no una traducción literal del paper.

Regla candidata:

`prune(w, u) = 0` si `abs(w) < tau_w` y `usage < tau_u`; en otro caso, conservar `w`.

**Conexión con Lumen:** el crecimiento de `bias_norm` no demuestra densificación de conexiones: un sesgo no es una sinapsis ni una matriz de conectividad. Antes de proponer poda hay que identificar parámetros estructurales que puedan eliminarse sin cambiar la semántica del modelo y medir su uso real.

**Experimento falsable:** comenzar con poda *solo simulada* (máscara temporal, sin borrar pesos). Usar ventanas de uso, umbral por capa y protección de parámetros recién activados o etiquetados. Comparar sin poda, poda aleatoria con igual sparsity y poda débil+infrautilizada. Medir sparsity real, latencia, memoria, NLL, holdout y recuperación tras reactivar conexiones.

**Criterio de rechazo:** si la poda selectiva no supera al control aleatorio con igual sparsity, o causa pérdida irreversible de capacidad, no se promueve. La poda física solo se considera después de validar la máscara y la consolidación.

### 10. Sueño computacional: renormalización y consolidación offline

**Referencia:** Tononi & Cirelli (2014), [Sleep and the Price of Plasticity](https://doi.org/10.1016/j.neuron.2013.12.025), *Neuron* 81(1):12–34.

La hipótesis de homeostasis sináptica propone que el sueño contribuye a renormalizar la fuerza sináptica tras la plasticidad de la vigilia y puede apoyar consolidación e integración. Es una hipótesis neurobiológica con evidencia y debates propios; no implica que un trainer necesite una fase llamada «sueño» para evitar colapsar.

Abstracción experimental:

- renormalización moderada de cambios no protegidos;
- preservación de parámetros o actualizaciones seleccionados por una señal de consolidación;
- replay offline de ejemplos o secuencias seleccionadas;
- comparación con un control de igual presupuesto de cómputo sin fase offline.

**Conexión con Lumen:** un log `sleep=0.000s` solo indica lo que mida ese contador en esa ejecución. No demuestra ausencia de consolidación biológica ni que el crecimiento de `bias_norm` sea causado por falta de sueño. Hay que inspeccionar el scheduler, los eventos de replay y qué estado modifica cada fase.

**Experimento factorial:** (A) sin fase offline, (B) renormalización sola, (C) replay solo, (D) renormalización + replay protegido por tags. Igualar tokens y, en un segundo análisis, igualar tiempo de CPU. Medir retención de conocimientos anteriores, transferencia, NLL y deriva de pesos.

**Criterio de rechazo:** si la fase offline solo ayuda por consumir más datos o cómputo, o si perjudica retención/holdout, no se atribuye beneficio al mecanismo de sueño.

### 11. Replay secuencial inspirado en sharp-wave ripples

**Referencias:** Buzsáki (2015), [Hippocampal sharp wave-ripple](https://doi.org/10.1002/hipo.22488); Foster & Wilson (2006), [Reverse replay of behavioural sequences in hippocampal place cells during the awake state](https://doi.org/10.1038/nature04587).

El replay hipocampal reactiva secuencias de actividad y puede ocurrir tanto durante el sueño como en vigilia tranquila; Foster y Wilson describen replay inverso durante vigilia. La analogía útil para Lumen es la reactivación selectiva de secuencias, no copiar literalmente frecuencias de 100–200 Hz ni asumir una compresión fija de 15×.

La función de ejemplo `replay_compress` no comprime realmente la secuencia ni reduce el número de operaciones de aprendizaje: devuelve la secuencia intacta y un contador aproximado. Por tanto, sus asserts solo verifican una convención aritmética, no replay comprimido.

**Diseño computacional comprobable:**
1. Guardar secuencias candidatas con puntuaciones explícitas: novedad, error, mejora/retención y etiqueta local.
2. Seleccionar replay por prioridad y comparar con replay uniforme y sin replay.
3. Mantener el orden de cada secuencia; añadir replay inverso como brazo separado.
4. Si se afirma compresión temporal, medir operaciones/tiempo real y demostrar que se conserva la información relevante.

**Criterio de rechazo:** si el replay priorizado no mejora retención o generalización frente al replay uniforme con el mismo presupuesto, no se promueve. Evitar confundir «menos pasos de replay» con «misma actualización comprimida».

### 12. Depresión a largo plazo (LTD) y olvido activo

**Referencias:** Collingridge et al. (2010), [Long-term depression in the CNS](https://doi.org/10.1038/nrn2867); Malenka & Bear (2004), [LTP and LTD: an embarrassment of riches](https://doi.org/10.1038/nrn1475).

LTD agrupa mecanismos diversos de debilitamiento sináptico dependientes del contexto; no equivale a una regla universal «post sin pre ⇒ restar una constante». La regla de ejemplo es un juguete lógico, no una implementación fisiológica ni una regla validada para pesos de lenguaje.

**Hipótesis de ingeniería:** introducir una pequeña señal de debilitamiento solo sobre parámetros auxiliares o conexiones que, durante ventanas largas, estén infrautilizados y no protegidos por una señal de consolidación. Distinguir falta de uso de evidencia de que una asociación es incorrecta.

**Experimento falsable:** comparar sin LTD, LTD uniforme, LTD condicionado por uso y LTD condicionado por uso + protección por tag. Mantener el mismo presupuesto de entrenamiento. Medir retención de conocimiento previo, aprendizaje de datos nuevos, interferencia, NLL y recuperación de parámetros.

**Riesgos:** el uso puede ser raro pero esencial; un tag puede estar obsoleto; una regla local puede borrar asociaciones necesarias para generalización. Proteger solo con tags recientes tampoco basta: evaluar caducidad y falsos positivos de protección.

**Criterio de rechazo:** si el olvido activo borra capacidades anteriores o no mejora adquisición de información nueva frente al control, se elimina.

## Tercera tanda: eficiencia neural, estabilidad temporal, conectividad y feedback (hipótesis 13–17)

**Estado:** agenda de investigación añadida al torneo de eficiencia. Los estudios humanos citados son asociaciones o resultados dependientes de tarea; no constituyen una receta directa para una red de lenguaje.

### 13. Eficiencia neural y activación focal

**Referencias:**
- Haier et al. (1988), *Cortical glucose metabolic rate correlates of abstract reasoning and attention studied with positron emission tomography*, *Intelligence* 12:199–217.
- Neubauer & Fink (2009), [Intelligence and neural efficiency](https://doi.org/10.1016/j.neubiorev.2009.04.001), *Neuroscience & Biobehavioral Reviews* 33(7):1004–1023.
- Para una crítica conceptual: [Is “efficiency” a useful concept in cognitive neuroscience?](https://pmc.ncbi.nlm.nih.gov/articles/PMC6989750/).

La hipótesis de eficiencia neural ha recibido apoyo en algunos paradigmas, pero la revisión de Neubauer y Fink señala moderadores importantes: dificultad, práctica, sexo y región cerebral; en tareas complejas puede observarse más activación asociada a mayor capacidad. Por tanto, no es correcto resumir la literatura como «las personas más inteligentes siempre activan menos cerebro».

**Traducción a LUMEN:** medir activación efectiva y trabajo computacional por tarea; comparar Top-K estructurado, gating por sorpresa y rutas selectivas. El objetivo no es minimizar actividad sin más, sino reducir actividad irrelevante manteniendo diversidad, exactitud y NLL.

**Hipótesis falsable:** para calidad equivalente en el holdout, una selección más focal reduce forward/backward real o coste de CPU sin aumentar unidades muertas persistentes ni concentrar toda la carga en pocos canales.

### 14. Resistencia temporal de unidades y estabilidad bajo uso repetido

**Referencia:** Goriounova et al. (2018), [Large and fast human pyramidal neurons associate with intelligence](https://doi.org/10.7554/eLife.41714), *eLife* 7:e41714.

El estudio asocia puntuaciones de IQ con propiedades morfológicas y fisiológicas de neuronas piramidales humanas temporales; también reporta que ciertas propiedades de los potenciales de acción se mantienen mejor durante disparos repetidos. No demuestra que una EMA de actividad, una ventana de elegibilidad o una dimensión mayor reproduzcan esa fisiología.

**Traducción a LUMEN:** probar estabilidad temporal de las activaciones, gradientes y actualizaciones bajo secuencias largas; medir saturación, unidades persistentemente muertas, deriva de normas y recuperación tras cambios de distribución. La ventana de elegibilidad 0.98 y la homeostasis actual son condiciones de referencia, no parámetros biológicamente óptimos.

**Hipótesis falsable:** un controlador de umbral/escala lento y acotado mantiene throughput y señal de gradiente más estables a lo largo de 1.000 pasos, sin empeorar holdout ni reducir diversidad.

### 15. Reutilización de configuración y menor reconfiguración de red

**Referencia:** Schultz & Cole (2016), [Higher Intelligence Is Associated with Less Task-Related Brain Network Reconfiguration](https://doi.org/10.1523/JNEUROSCI.0358-16.2016), *Journal of Neuroscience* 36(33):8551–8561.

El trabajo relaciona diferencias individuales en inteligencia con patrones de reconfiguración funcional entre reposo y tareas. No prueba que reutilizar gradientes obsoletos sea siempre mejor: una red funcional puede conservar gran parte de su conectividad y aun así requerir ajustes específicos para la tarea.

**Traducción a LUMEN:** reutilizar trazas de elegibilidad y resultados invariantes cuando la procedencia y el contexto siguen siendo válidos; invalidarlos cuando cambien los datos, el estado recurrente o el régimen de sorpresa. Lazy backward se evalúa como aproximación de ingeniería, no como equivalente de conectividad cerebral estable.

**Hipótesis falsable:** reutilización controlada reduce backward/tiempo frente a recomputación y mantiene la dirección útil de actualización, la estabilidad y el holdout.

### 16. Conectividad selectiva entre módulos

**Referencia:** Jung & Haier (2007), [The Parieto-Frontal Integration Theory (P-FIT) of intelligence: converging neuroimaging evidence](https://doi.org/10.1017/S0140525X07001185), *Behavioral and Brain Sciences* 30:135–187.

P-FIT propone una red distribuida en la que participan regiones parietales y frontales, entre otras. No justifica asignar literalmente una capa de LUMEN a una región cerebral ni demuestra que menos conexiones sean siempre mejores.

**Traducción a LUMEN:** estudiar conectividad inter-capa selectiva, rutas residuales y gates que preserven la información necesaria. Medir qué capas aportan señal útil a las siguientes, gradiente por frontera y coste de comunicación. No aumentar capas ni cortar conexiones por analogía anatómica.

**Hipótesis falsable:** una ruta selectiva y estable mantiene o mejora el flujo de información y reduce operaciones redundantes frente a la conectividad densa de referencia.

### 17. Aprendizaje sensible al feedback positivo inesperado

**Referencia:** van den Bos, Crone & Güroğlu (2012), [Brain function during probabilistic learning in relation to IQ and level of education](https://doi.org/10.1016/j.dcn.2011.09.007), *Developmental Cognitive Neuroscience* 2(Suppl 1):S78–S89.

En una muestra de 45 adolescentes y una tarea probabilística, el estudio encontró diferencias de estrategia adaptativa tras feedback positivo y asociaciones de activación en DLPFC/dACC. No demuestra que «placer deba ser más fuerte que dolor» de forma universal, ni que una ganancia concreta mejore un modelo autoregresivo.

**Traducción a LUMEN:** comparar ganancias acotadas de placer y dolor sobre la señal compartida de evolución de loss, manteniendo glucosa independiente y evitando que el feedback modifique directamente el learning rate global. Evaluar si una mejora confirmada refuerza trazas útiles sin hacer que el sistema ignore regresiones.

**Hipótesis falsable:** una asimetría moderada y validada por ventanas mejora recuperación tras feedback positivo y retención, sin volver al modelo insensible a regresiones ni disparar plasticidad inestable.

### Correspondencia con el torneo

- Hipótesis 13 → H01/H02 (activación focal y predicción selectiva).
- Hipótesis 14 → H06 (estabilidad temporal y homeostasis).
- Hipótesis 15 → H07 (elegibilidad y reutilización controlada).
- Hipótesis 16 → H08 (conectividad selectiva entre capas).
- Hipótesis 17 → H09 (feedback positivo/negativo acotado).

## Síntesis: sistema candidato, no explicación cerrada

Las doce entradas pueden organizarse como un circuito de hipótesis:

1. **Plasticidad local:** elegibilidad y modulador de tres factores.
2. **Selección/consolidación:** tags, recursos y replay priorizado.
3. **Estabilidad:** homeostasis sináptica e intrínseca, con escalas temporales lentas.
4. **Control de capacidad:** sparsity y poda selectiva después de medir uso y retención.
5. **Renovación:** renormalización offline y LTD condicionada, con protección de memorias útiles.
6. **Lecturas auxiliares:** ELM y one-shot como experimentos separados, no sustitutos ya demostrados del entrenamiento autoregresivo.

No se concluye que Lumen «tenga» ya los mecanismos 7–12, que el Transformer carezca de mecanismos de estabilización equivalentes, ni que estos expliquen la diferencia de velocidad o los resultados de holdout. Tampoco se puede atribuir causalidad a métricas aisladas como `homeo_out`, `gate+`, `dead`, `bias_norm` o `sleep` sin rastrear sus definiciones y comparar ablaciones.

## Orden de prioridad actualizado

1. Auditar el código y telemetría existente: homeostasis, actividad efectiva, replay, tags y modulador.
2. Medir primero sin cambiar dinámica: curvas de norma, actividad, concentración, reactivación y coste.
3. Probar homeostasis/umbral en ablaciones independientes y lentas.
4. Probar tag × replay con diseño factorial.
5. Evaluar poda y LTD primero como máscaras reversibles/simuladas.
6. Probar la fase offline con presupuesto de cómputo controlado.
7. Mantener ELM y one-shot en bancos auxiliares hasta que demuestren valor.
8. No combinar mecanismos en un único train hasta que cada uno haya superado su prueba aislada; después, probar interacciones con un diseño factorial pequeño.

## Referencias adicionales verificadas

- Turrigiano (2008), DOI [10.1016/j.cell.2008.10.008](https://doi.org/10.1016/j.cell.2008.10.008). PubMed/PMC: [PMID 18984155](https://pubmed.ncbi.nlm.nih.gov/18984155/).
- Ibata, Sun & Turrigiano (2008), DOI [10.1016/j.neuron.2008.02.031](https://doi.org/10.1016/j.neuron.2008.02.031).
- Desai, Rutherford & Turrigiano (1999), DOI [10.1038/9165](https://doi.org/10.1038/9165).
- Chechik, Meilijson & Ruppin (1998), DOI [10.1162/089976698300017124](https://doi.org/10.1162/089976698300017124).
- Tononi & Cirelli (2014), DOI [10.1016/j.neuron.2013.12.025](https://doi.org/10.1016/j.neuron.2013.12.025).
- Buzsáki (2015), DOI [10.1002/hipo.22488](https://doi.org/10.1002/hipo.22488).
- Foster & Wilson (2006), DOI [10.1038/nature04587](https://doi.org/10.1038/nature04587).
- Collingridge et al. (2010), DOI [10.1038/nrn2867](https://doi.org/10.1038/nrn2867).
- Malenka & Bear (2004), DOI [10.1038/nrn1475](https://doi.org/10.1038/nrn1475).


**Aclaración bibliográfica sobre sueño:** el DOI 10.1016/j.tics.2013.12.005 sí es correcto para Huber & Born (2014), pero el título completo es *Sleep, synaptic connectivity, and hippocampal memory during early development* (no el título abreviado sin «during early development»). La revisión general de SHY de Tononi & Cirelli (2014) es otro artículo distinto, DOI 10.1016/j.neuron.2013.12.025. Se conservan ambos porque respaldan aspectos relacionados, pero no son intercambiables.


## Decisión experimental del piloto integrado de 1.000 pasos

La siguiente prueba solicitada para LUMEN no será una secuencia de ablaciones previas: será un **piloto integrado**, con las doce familias de mecanismos activadas conjuntamente en LUMEN y un Transformer de control entrenado en paralelo desde el checkpoint pareado promovido de mayor paso. El objetivo es comprobar si el sistema completo puede aprender y mantenerse estable, y medir si la optimización del núcleo permite recuperar pasos por segundo.

La integración no convierte las hipótesis en hechos. Cada órgano tendrá una bandera/configuración explícita, telemetría de activación y contadores de coste; los mecanismos lentos (sueño, replay, poda y LTD) se ejecutarán en ciclos programados, no necesariamente en cada actualización. Las reglas irreversibles de poda/olvido se mantendrán primero como máscaras reversibles/propuestas hasta validar que no dañan la retención. ELM y one-shot se probarán como rutas auxiliares, sin reemplazar el objetivo autoregresivo ni modificar de forma opaca el optimizador principal.

### Condiciones de aceptación del piloto

- Mismo checkpoint de partida cuando la arquitectura lo permita, corpus v3 limpio y autorizado, tokenizer, semilla, batch, contexto y presupuesto de tokens iguales.
- 1.000 actualizaciones para cada modelo, ejecutadas en paralelo en el mismo tipo de runner cuando sea posible.
- Informar pasos/segundo, tiempo de forward, backward, mantenimiento y overhead; reportar tanto coste total como coste sin fases periódicas.
- Verificar ausencia de NaN/Inf, deriva de normas, actividad efectiva, unidades inactivas, memoria ocupada, tags protegidos y eventos de replay/poda/LTD.
- Evaluar con el mismo holdout protegido; no tocarlo durante selección ni desarrollo.
- No atribuir una mejora causal a un órgano individual a partir de este único ensayo conjunto. Si el paquete falla, conservar logs y reducir el fallo a una siguiente matriz de pruebas.

### Orden de trabajo actualizado

1. Validar migración y equivalencia de los fastpaths Delta.
2. Medir y reducir el coste recurrente Delta/autograd antes de introducir mantenimiento adicional.
3. Implementar los órganos que faltan y conectar los existentes con instrumentación explícita.
4. Ejecutar el piloto conjunto LUMEN vs Transformer de 1.000 pasos, solo después de autorizar corpus v3.
