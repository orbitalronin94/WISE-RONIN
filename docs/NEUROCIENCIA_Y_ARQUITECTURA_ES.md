# Neurociencia convertida en decisiones de arquitectura

**RONIN-LUMEN no es un modelo biológico.** Es una arquitectura de aprendizaje automático que toma problemas y principios estudiados en neurociencia —estabilidad de actividad, codificación escasa, predicción, memoria con distintas escalas temporales y procesos offline— y los traduce a mecanismos de software que pueden medirse en un modelo pequeño.

La reivindicación del proyecto no es haber descubierto esos principios. Es haberlos reunido deliberadamente para atacar un fallo observado en un modelo pequeño: la degradación progresiva hacia unas pocas respuestas dominantes. En lugar de asumir que el remedio es escalar una pila Transformer, preguntamos qué puede enseñarnos la literatura sobre cómo los sistemas de aprendizaje mantienen estabilidad, selectividad y memoria.

## Índice

1. [La tesis](#la-tesis)
2. [Del paper al mecanismo](#del-paper-al-mecanismo)
3. [Qué no procede de neurociencia](#qué-no-procede-de-neurociencia)
4. [Cómo se va a demostrar](#cómo-se-va-a-demostrar)
5. [Bibliografía primaria](#bibliografía-primaria)

## La tesis

El Transformer de Vaswani et al. (2017) introdujo una arquitectura basada en autoatención que cambió el aprendizaje de secuencias. RONIN-LUMEN no pretende desautorizar ese avance. Decide no tratarlo como plantilla obligatoria para cualquier escala y restricción.

El punto de partida es otro: un modelo de pocos millones de parámetros tiene un presupuesto muy limitado. Si unas pocas unidades, rutas o patrones dominan el aprendizaje, el sistema puede reducir su pérdida y al mismo tiempo perder diversidad funcional. Por eso LUMEN hace explícitas cinco preguntas:

- ¿Cuándo merece la pena escribir en el estado?
- ¿Cómo evitar que las mismas unidades se activen para casi todo?
- ¿Cómo mantener separadas la adaptación rápida y la memoria estable?
- ¿Qué debería ocurrir cuando no llegan ejemplos nuevos?
- ¿Cómo repartir el presupuesto entre memoria, estado y salida?

La neurociencia aporta hipótesis sobre estas preguntas. La implementación concreta es nuestra abstracción computacional y debe evaluarse por sus propios resultados.

## Del paper al mecanismo

### 1. Plasticidad homeostática → regulación de actividad

**Literatura:** Turrigiano y colaboradores describen mecanismos homeostáticos por los que las neuronas ajustan excitabilidad y fuerza sináptica para estabilizar su actividad, compensando perturbaciones que de otro modo podrían conducir a actividad excesiva o insuficiente.

**Traducción en LUMEN:** cada dimensión mantiene una media móvil de activación y un umbral adaptativo. Las unidades usadas en exceso reciben un umbral más restrictivo; la capa conserva después las activaciones Top-K.

**Qué se puede probar:** distribución de actividad, porcentaje de unidades prácticamente muertas, concentración de activación, estabilidad entre fases y efecto sobre la repetición.

**Límite:** el EMA y el bias sustractivo del código no reproducen los sensores de calcio, la regulación de receptores ni las escalas temporales celulares. Es un lazo de realimentación inspirado en homeostasis, no una neurona biológica.

### 2. Codificación escasa → Top-K y silencio informativo

**Literatura:** Olshausen y Field mostraron que aprender códigos escasos sobre imágenes naturales puede producir representaciones eficientes con propiedades parecidas a campos receptivos de células simples de la corteza visual.

**Traducción en LUMEN:** después de cada etapa homeostática, solo sobreviven las K activaciones mayores. El resto se anula. El silencio no se trata como un fallo: es una restricción de representación.

**Qué se puede probar:** densidad real de activación, coste de CPU, diversidad de unidades activas, rendimiento y robustez frente a entradas fuera de distribución.

**Límite:** la codificación escasa del paper no equivale a aplicar Top-K sobre activaciones de un modelo de lenguaje. La analogía justifica investigar escasez, no garantiza que K=16/32 sea óptimo.

### 3. Codificación predictiva y error → puerta de sorpresa del estado Delta

**Literatura:** Rao y Ballard modelaron la corteza visual mediante predicciones descendentes y errores residuales ascendentes. La discrepancia entre predicción y entrada desempeña un papel central en esa familia de teorías. Otras líneas de investigación estudian señales de error de predicción en sistemas dopaminérgicos.

**Traducción en LUMEN:** el estado recurrente predice un vector; el residuo determina una medida de sorpresa. Esa medida gobierna si se escribe, cuánto se actualiza y cuánto decae el estado.

**Qué se puede probar:** tasa de escrituras, magnitud de sorpresa, retención de contexto, pérdida de información por umbral y coste por token.

**Límite:** la sorpresa implementada es una norma del residuo de un predictor entrenable. No es una medición de dopamina, noradrenalina, conciencia, novedad subjetiva ni energía libre biológica. La regla Delta también pertenece a una línea de arquitectura computacional moderna; no debe atribuirse como invención neurocientífica de LUMEN.

### 4. Sistemas de memoria complementarios → memoria rápida y lenta

**Literatura:** el marco de sistemas de aprendizaje complementarios propone que sistemas con propiedades distintas pueden resolver el compromiso entre aprender rápidamente episodios nuevos e integrar gradualmente regularidades sin interferencia catastrófica.

**Traducción en LUMEN:** se separa una ruta elástica de adaptación y un banco lento de registros consolidados. La memoria lenta tiene capacidad acotada y política explícita de uso/reemplazo.

**Qué se puede probar:** retención de información anterior, interferencia al introducir ejemplos nuevos, recuperación de registros, ocupación, colisiones y degradación tras consolidación.

**Límite:** dos tensores o bancos no constituyen por sí mismos un hipocampo y una neocorteza. La implementación actual simplifica los mecanismos de selección, replay y consolidación descritos en la literatura.

### 5. Sueño y homeostasis sináptica → replay offline

**Literatura:** Tononi y Cirelli proponen la hipótesis de homeostasis sináptica: el sueño podría contribuir a renormalizar la fuerza sináptica tras el aprendizaje durante la vigilia, preservar selectividad y facilitar aprendizaje posterior. La literatura también distingue esta hipótesis de modelos de consolidación activa por reactivación/replay; no son explicaciones idénticas.

**Traducción en LUMEN:** el trainer reserva intervalos sin ejemplos externos para reconstruir registros lentos, y después reinicia los estados rápidos. El código actual usa una pequeña secuencia de replay sobre la memoria lenta.

**Qué se puede probar:** comparación con `--no-sleep`, retención antes/después, olvido, diversidad, loss de reconstrucción y coste temporal.

**Límite:** el sueño artificial actual no reproduce fases NREM/REM, oscilaciones, neuromodulación ni actividad cerebral. Tampoco se ha demostrado que el calendario o la secuencia 1/3/7 sean óptimos. Es una rutina de consolidación offline inspirada en hipótesis sobre el sueño.

### 6. Presupuesto de energía → restricción explícita de recursos

La idea de un presupuesto por token toma inspiración general de la eficiencia energética del sistema nervioso, pero la implementación es un softmax aprendido que distribuye masa entre estado Delta, memoria y cabeza de salida. No es un modelo metabólico. Se evalúa observando entropía de asignación, colapso de rutas, coste computacional y rendimiento.

### 7. Extensión V3.2 ZE / LC–NE

La rama V3.2 explora dos predictores, discrepancia entre ellos, bandas de error de predicción y una política de escritura de memoria. Su motivación incluye trabajos sobre sistemas noradrenérgicos y control adaptativo, pero los umbrales y el operador de borrado son decisiones de ingeniería. V3.2 permanece optativa y experimental; no se presenta como la configuración por defecto ni como una mejora demostrada.

## Qué no procede de neurociencia

Para que la atribución sea rigurosa, hay que separar las genealogías:

- **Estado Delta recurrente:** se relaciona con líneas modernas de modelos secuenciales y reglas Delta; no es una invención de este proyecto ni una traducción literal de un circuito cerebral.
- **KAN:** las redes Kolmogorov–Arnold son una propuesta matemática/ML, no una arquitectura descubierta en neurociencia.
- **JEPA-style:** la predicción en espacio latente pertenece a una línea contemporánea de aprendizaje de representaciones. Su presencia aquí no demuestra plausibilidad biológica.
- **BPE, AdamW, softmax y PyTorch:** son herramientas y decisiones de ingeniería convencional.
- **Router MLP/KAN:** es una hipótesis de especialización computacional. No equivale a una división cerebral lenguaje/aritmética establecida por este diseño.

Esta separación no debilita la propuesta: evita vender analogías como evidencia y deja claro qué componente se está probando realmente.

## Cómo se va a demostrar

La arquitectura solo podrá reivindicar una mejora si pasa una evaluación que permita atribuir efectos:

1. Congelar dataset, tokenizer, hashes, semillas y holdout antes de la campaña.
2. Ejecutar un baseline comparable y la configuración completa con el mismo presupuesto de tokens y pasos.
3. Hacer ablaciones independientes: sin homeostasis, sin Top-K, sin memoria lenta, sin sueño, sin JEPA, sin router especializado y sin puerta de sorpresa.
4. Medir pérdida/perplejidad, exactitud por tarea, repetición, diversidad, finalización EOS, unidades muertas, distribución de energía, ocupación de memoria, tiempo CPU y memoria.
5. Repetir con varias semillas y publicar resultados negativos.
6. No mezclar resultados de V3.0 con la extensión V3.2.
7. No llamar entrenamiento real al smoke test dummy ni llamar validación a una compilación correcta.

El holdout histórico tiene 500 filas y 55 prompts únicos. Debe conservarse intacto y sus limitaciones deben declararse. Una auditoría léxica sin coincidencias no demuestra ausencia de contaminación semántica.

## Bibliografía primaria

### Neurociencia y ciencia cognitiva

1. **Turrigiano, G. G. & Nelson, S. B. (2004).** *Homeostatic plasticity in the developing nervous system.* Nature Reviews Neuroscience, 5, 97–107. [DOI](https://doi.org/10.1038/nrn1327).
2. **Turrigiano, G. G. (2008).** *The Self-Tuning Neuron: Synaptic Scaling of Excitatory Synapses.* Cell, 135(3), 422–435. [Texto completo en PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC2834419/).
3. **Olshausen, B. A. & Field, D. J. (1996).** *Emergence of simple-cell receptive field properties by learning a sparse code for natural images.* Nature, 381, 607–609. [DOI](https://doi.org/10.1038/381607a0).
4. **Rao, R. P. N. & Ballard, D. H. (1999).** *Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects.* Nature Neuroscience, 2, 79–87. [DOI](https://doi.org/10.1038/4580).
5. **Montague, P. R., Dayan, P. & Sejnowski, T. J. (1996).** *A framework for mesencephalic dopamine systems based on predictive Hebbian learning.* Journal of Neuroscience, 16(5), 1936–1947. [DOI](https://doi.org/10.1523/JNEUROSCI.16-05-01936.1996).
6. **McClelland, J. L., McNaughton, B. L. & O’Reilly, R. C. (1995).** *Why There Are Complementary Learning Systems in the Hippocampus and Neocortex.* Psychological Review, 102, 419–457. [Referencia y publicaciones del laboratorio](https://ccnlab.org/pubs/).
7. **Aston-Jones, G. & Cohen, J. D. (2005).** *An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance.* Annual Review of Neuroscience, 28, 403–450. [DOI](https://doi.org/10.1146/annurev.neuro.28.061604.135709).
8. **Tononi, G. & Cirelli, C. (2006).** *Sleep function and synaptic homeostasis.* Sleep Medicine Reviews, 10(1), 49–62. [PubMed](https://pubmed.ncbi.nlm.nih.gov/16376591/).
9. **Tononi, G. & Cirelli, C. (2014).** *Sleep and the Price of Plasticity: From Synaptic and Cellular Homeostasis to Memory Consolidation and Integration.* Neuron, 81(1), 12–34. [Texto completo en PubMed Central](https://pmc.ncbi.nlm.nih.gov/articles/PMC3921176/).
10. **Friston, K. (2005).** *A theory of cortical responses.* Philosophical Transactions of the Royal Society B, 360, 815–836. [DOI](https://doi.org/10.1098/rstb.2005.1622) · [Texto completo](https://pmc.ncbi.nlm.nih.gov/articles/PMC1569488/).

### Arquitectura de aprendizaje automático y contexto

11. **Vaswani et al. (2017).** *Attention Is All You Need.* [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) — referencia histórica del Transformer, citada como contraste arquitectónico, no como adversario.
12. **LeCun (2022).** *A Path Towards Autonomous Machine Intelligence.* [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) — contexto de aprendizaje predictivo en espacio latente.
13. **Liu et al. (2024).** *KAN: Kolmogorov-Arnold Networks.* [arXiv:2404.19756](https://arxiv.org/abs/2404.19756) — procedencia matemática/ML de la cabeza spline.

---

**Atribución:** las ideas científicas pertenecen a sus autores y comunidades de investigación. La aportación de RONIN-LUMEN es la traducción experimental, selección e integración de mecanismos inspirados en esa literatura dentro de un modelo compacto y CPU-first, con límites y resultados sujetos a verificación.
