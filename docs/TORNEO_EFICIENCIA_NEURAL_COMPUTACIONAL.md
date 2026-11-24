# Torneo ampliado de eficiencia neural y computacional para LUMEN

**Estado:** diseño de campaña; aún no despachado. Rama limpia desde `main` para evitar mezclar prototipos incompletos de la rama anterior.

## Tesis

No buscamos que LUMEN calcule menos a cualquier precio. Buscamos que concentre el cálculo y la plasticidad en las rutas que más contribuyen al aprendizaje, reutilice estados y resultados válidos, y evite recalcular señales equivalentes.

La literatura sobre eficiencia neural, conectividad, homeostasis y aprendizaje por recompensa inspira hipótesis; no demuestra que sus resultados se transfieran directamente a un modelo autoregresivo. Los términos «genio», «eficiencia» o «cerebro» no son métricas. Las métricas serán cómputo, aprendizaje, estabilidad y holdout.

## Diseño de la ronda 1

**21 brazos simultáneos: 1 control + 20 hipótesis.**

- B0: `main` actual, exactamente 1.000 pasos.
- H01–H20: una hipótesis por brazo, exactamente 1.000 pasos.
- Un runner independiente por brazo, todos lanzados por una única matriz de GitHub Actions.
- Todos los brazos parten del mismo checkpoint inmutable y reciben el mismo corpus, tokenizer, token cache, secuencia de batches, seed, batch, contexto, scheduler y entorno equivalente.
- Ningún candidato hereda el resultado de otro. Los mecanismos se aíslan en la primera ronda.
- El holdout congelado de 2.431 preguntas se evalúa una vez terminados los 1.000 pasos. No se usa para seleccionar umbrales.
- Un job de preparación valida hashes, configuración y checkpoint antes de permitir que arranque cualquier brazo. Si un brazo no supera smoke tests, la campaña completa no se despacha.

## Hipótesis que compiten

### A. Eficiencia neural y asignación de plasticidad

| ID | Hipótesis | Traducción comprobable |
|---|---|---|
| H01 | Disparo disperso / eficiencia de activación | Variantes estructuradas de Top-K frente a Top-K actual; medir actividad efectiva, diversidad y coste. Top-K ya existe: no se cuenta como novedad |
| H02 | Predicción antes que corrección | Gate por sorpresa temporal suavizada e histéresis frente al gate instantáneo; medir backward y updates omitidos |
| H03 | Inhibición selectiva | Perfil neutral frente a inhibición/desinhibición 0.80; medir competencia, actividad y trabajo posterior realmente evitado |
| H04 | Aprendizaje local | Objetivo auxiliar por bloque y stop-gradient acotado frente a gradiente global; cambio de algoritmo, no optimización exacta |
| H05 | Sueño y replay comprimido | Sin replay frente a replay 100/50/20% priorizado, usando recuerdos reales; validar banco ocupado y actualizaciones reales |
| H06 | Resistencia a fatiga / homeostasis temporal | Probar retención de velocidad/actividad en ventanas largas, EMA y umbral intrínseco adaptativo; medir deriva, unidades muertas y recuperación. La ventana 0.98 existente es punto de partida, no verdad biológica |
| H07 | Menos reconfiguración / elegibilidad persistente | Comparar recomputación frente a reutilización controlada de trazas de elegibilidad; medir estabilidad de ruta, interferencia y señal moduladora |
| H08 | Conectividad selectiva entre capas | Probar rutas residuales/gates inter-capa y comunicación selectiva entre bloques; medir gradiente, transferencia de información y coste. Inspiración P-FIT, no equivalencia de regiones cerebrales |
| H09 | Feedback positivo y negativo asimétrico | Comparar ganancia actual con placer moderadamente mayor que dolor; mantener glucosa separada y limitar ambos moduladores; medir aprendizaje tras mejora/regresión |
| H10 | Tagging & capture / one-shot acotado | Tags locales con captura por señal global y escritura rápida condicionada por incertidumbre; no tocar LR global; medir retención e interferencia |

### B. Forward, backward, matemáticas y backend

| ID | Hipótesis | Traducción comprobable |
|---|---|---|
| H11 | Compartir proyecciones y cálculos comunes | Fusionar/reutilizar proyecciones solo si se conservan outputs y gradientes |
| H12 | Diferir diagnósticos | Ejecutar normas/estadísticas no funcionales en intervalos, después de comprobar todos sus consumidores |
| H13 | Reutilizar máscaras y tensores invariantes | Compartir causal masks y tensores por forward entre órganos, sin cachear valores dependientes de estado |
| H14 | Backward selectivo / lazy backward | Comparar control con mecanismos actuales y gate temporal; registrar backward reales y optimizer updates |
| H15 | Normalización compartida en loss | Compartir log-softmax entre CE y repetición solo con paridad numérica y de gradientes |
| H16 | Kernel/operaciones PyTorch | Microbench de kernels, foreach/fused, scatter/gather y layouts; medir CPU real, temporales y precisión |

### C. Python y control del runtime

| ID | Hipótesis | Traducción comprobable |
|---|---|---|
| H17 | Vectorizar memoria elástica | Vectorizar escrituras por batch solo cuando preserve orden, reemplazo, normalización y contadores; fallback secuencial si no |
| H18 | Reducir sincronizaciones Python↔Torch | Agrupar lecturas escalares y evitar `.item()` en el camino caliente; no cambiar decisiones de control |
| H19 | Contabilidad diferida | Muestrear histogramas/contadores no funcionales; conservar estado y periodicidad semántica |
| H20 | Simplificar EMA homeostática diagnóstica | Probar diferir `m` solo tras demostrar que no gobierna control, checkpoints ni consumidores indirectos |

### Investigación auxiliar que no se debe forzar dentro del torneo

- **ELM:** probe offline sobre representaciones congeladas; no modifica el trainer ni compite como brazo de 1.000 pasos porque evalúa un readout analítico distinto, no una política de actualización del modelo autoregresivo.
- **Izhikevich y R-STDP:** referencias para validar matemáticamente el mecanismo de tres factores/elegibilidad; no se duplican como dos brazos independientes.
- **Poda/LTD:** se prueban primero como máscaras reversibles y protegidas, dentro de H01/H06/H07; prohibido borrar pesos durante esta campaña.
- **Nicola–Clopath/interneuronas:** inspiran diversidad, secuencias y replay; no justifican fijar un porcentaje universal de unidades activas o muertas.
- **Frontoparietal/P-FIT:** inspira H08, pero no se etiqueta ninguna capa como «frontal» o «parietal» sin una función computacional definida y medida.

## Qué dicen y qué no dicen los cinco grupos de papers recientes

1. **Eficiencia neural e inteligencia (Haier y trabajos relacionados):** algunos estudios encuentran patrones de eficiencia o activación más focal en determinadas tareas y muestras. No existe una regla universal «más CI = menos actividad» aplicable a todo cerebro, tarea o modelo. H01/H02 deben medir selectividad y coste sin sacrificar capacidad.
2. **Neuronas piramidales grandes y rápidas (Goriounova et al., 2018):** la asociación entre propiedades celulares y rendimiento cognitivo inspira H06: mantener estabilidad temporal y evitar saturación. No equivale a aumentar dimensión ni demuestra que una EMA de 0.98 sea óptima.
3. **Menor reconfiguración de red:** inspira H07, reutilizar estado/trazas si el contexto sigue siendo válido. Una traza obsoleta puede introducir error; se medirán deriva y recuperación.
4. **Conectividad frontoparietal/P-FIT:** inspira H08, conectividad selectiva y rutas inter-capa limpias. No se asume correspondencia anatómica uno-a-uno.
5. **Feedback en aprendizaje probabilístico:** inspira H09, probar si el feedback positivo moderado puede reforzar patrones útiles. No se asumirá que placer debe superar siempre al dolor; se hará barrido acotado y controlado.

Las referencias se mantendrán con DOI/enlaces y límites interpretativos en `docs/CEREBRO_A_CODIGO_HIPOTESIS_ES.md` y `docs/NEUROCIENCIA_Y_ARQUITECTURA_ES.md`. El resumen de resultados no atribuirá causalidad biológica al comportamiento del modelo.

## Integración de hipótesis pendientes del README y Cerebro a Código

La agenda previa no se pierde ni se duplica sin motivo:

- Tres factores/R-STDP e Izhikevich → H07.
- Synaptic tagging & capture → H10.
- One-shot condicionado por incertidumbre → H10.
- ELM → probe offline auxiliar.
- Interneuronas/diversidad → H01/H06/H05.
- Escalado homeostático y umbral intrínseco → H06.
- Poda selectiva y LTD protegida → H01/H06/H07 con máscaras reversibles.
- Sueño, downscaling y replay secuencial → H05.
- Compartir señales de sorpresa e incertidumbre → H02/H11/H13.
- Glucosa, dolor y placer con señal común pero funciones distintas → H09; no fusionar sus estados ni mezclar dolor con glucosa.
- Perfiles de plasticidad, tags, trazas y memoria one-shot → H07/H10.
- ZE/LC-NE y complejidad → no se activa automáticamente; medir por separado y preservar el perfil actual.
- Chunking adaptativo → candidato de H11/H13 solo si comparte cálculos y conserva la ruta residual.
- Todas las propuestas pendientes conservan su estado; el torneo no las convierte en mecanismos activos por documentarlas.

## Seguridad y no regresión

1. **Rama limpia:** el torneo parte de un SHA congelado de `main`; no se mezclan prototipos no validados.
2. **Cambios opt-in:** toda variante se desactiva por defecto y se serializa en configuración/checkpoint. Un flag ausente debe conservar el comportamiento actual.
3. **Paridad:** optimizaciones exactas comparan logits, cada componente de loss, gradientes, buffers, optimizer y scheduler con tolerancias explícitas.
4. **Cambios algorítmicos:** H01–H10 no prometen equivalencia exacta; necesitan invariantes, pruebas sintéticas, análisis de gradientes y holdout.
5. **Fallback:** memoria, replay, poda y rutas locales deben tener fallback a la ruta de referencia; ningún parámetro se borra.
6. **Aislamiento:** cada brazo corre en proceso y runner separado; una excepción cancela solo ese brazo, no contamina el checkpoint común.
7. **Checkpoint:** guardar estado completo, SHA de padres/datos/tokenizer/cache/holdout, seed, config y logs.
8. **No promoción automática:** un resultado de 1.000 pasos es cribado, no validación definitiva ni autorización para un train largo.

## Métricas

**Cómputo:** pasos/s, segundos/paso, forward/backward/Python, backward calls, optimizer updates, FLOPs estimados si son medibles, sincronizaciones, pico RAM, temporales y tamaño de checkpoint.

**Aprendizaje:** loss total y componentes, NLL token, ventanas móviles, gradientes por órgano, proporción de parámetros con gradiente, sorpresa, tasa de escritura, actividad por capa, entropía/Gini de unidades, estabilidad de trazas, ocupación de memoria y replay efectivo.

**Capacidad:** exactitud, NLL, Brier/ECE si se puede calcular correctamente, repetición, longitud y desglose por categoría en el mismo holdout congelado.

**Criterio:** un candidato solo es elegible si reduce coste medido y conserva calidad dentro del margen preinscrito. La mejora de throughput por sí sola no gana. Un candidato que ahorra backward pero empeora NLL/accuracy fuera del margen se rechaza o se etiqueta como trade-off.

## Ronda 2: ganadores combinados contra main

Después de cerrar Ronda 1:
- seleccionar únicamente mecanismos que superen gates de calidad, estabilidad y cómputo;
- crear una única configuración LUMEN combinada con esos ganadores;
- ejecutar en paralelo **1.000 pasos del combinado** y **1.000 pasos de main sin cambios**, ambos desde el mismo checkpoint padre, corpus, tokenizer, batches, seed y scheduler;
- evaluar ambos con el mismo holdout;
- si la combinación falla, hacer ablación factorial de interacciones; no se promueve ni se mezcla con main.

## Puerta de lanzamiento

No se lanza hasta que los 20 tratamientos tengan implementación real, tests, configuración serializada, smoke test de un batch y validación de paridad/invariantes. No se rellenan brazos con no-ops. El workflow debe demostrar en logs que arranca B0+H01–H20 a la vez y que cada uno completa exactamente 1.000 actualizaciones. Si el coste de runners simultáneos o límites de GitHub Actions impiden esto, se informa y se rediseña la matriz sin afirmar simultaneidad falsa.

**Estado actual: documentación/protocolo. No hay entrenamiento iniciado.**
