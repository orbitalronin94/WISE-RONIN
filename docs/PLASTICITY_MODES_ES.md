# Modos de plasticidad: cuatro palancas compartidas

## Principio

LUMEN no tiene metabolismo, sueño subjetivo, presión arterial ni toxicidad farmacológica. Podemos experimentar con modulaciones computacionales inspiradas en neuromodulación sin simular que administramos fármacos ni trasladar literalmente sus efectos biológicos.

Estos modos no crean cuatro órganos nuevos. Son perfiles temporales que modulan cuatro palancas existentes o previstas:
1. umbral e inhibición de plasticidad;
2. intensidad y duración de la señal moduladora de elegibilidad;
3. supervivencia de tags y ventana de captura frente a poda;
4. ganancia atencional del gate ante novedad/incertidumbre.

Los nombres cafeína, estimulante, psicodélico/ketamina y atención son etiquetas de inspiración, no afirmaciones de equivalencia farmacológica. No se asume que una sustancia produzca un único efecto molecular ni que sus mecanismos humanos se reduzcan a estos parámetros.

## Perfiles propuestos

| Perfil experimental | Palancas | Interpretación computacional |
|---|---|---|
| disinhibition (inspiración cafeína) | umbral LTP menor; inhibición menor | facilita que una señal elegible produzca actualización |
| modulator_gain (inspiración estimulante) | modulador M mayor; decaimiento de elegibilidad más lento | refuerza y prolonga señales de aprendizaje ya etiquetadas |
| plasticity_window (inspiración psicodélico/ketamina) | ventana de plasticidad mayor; decaimiento de tags más lento; poda temporalmente más conservadora | permite que más tags sobrevivan hasta la captura |
| attentional_gate (ACh/NA conceptual) | ganancia de gate dependiente de novedad/incertidumbre | aumenta selectividad señal/ruido cuando hay información nueva |

Los factores iniciales sugeridos (0.85, 0.8, 1.5, 1.8, 3.0, 2.0, 1.5 y 1.4) son valores de arranque para barrido, no constantes validadas. Se deben acotar y optimizar mediante benchmarks de 100 pasos.

## Control computacional

- Los modos son opt-in, temporales y con ramp-up/ramp-down; el perfil neutro es identidad exacta.
- Todos los multiplicadores tienen límites explícitos y se validan contra NaN/Inf.
- No se permite aumentar plasticidad sin límite: elegibilidad, tags, energía y cambios de pesos conservan sus cotas.
- La modulación se aplica a señales compartidas existentes; nunca crea una copia paralela de sorpresa, actividad, elegibilidad o incertidumbre.
- No se altera el learning rate del optimizador ni se desactiva la estabilidad numérica.
- El modo de ventana plástica no puede impedir la poda indefinidamente: solo aplaza la poda durante un intervalo con presupuesto máximo.
- El modo atencional responde a novedad/incertidumbre ya calculada, no vuelve a calcularlas.
- Se registran modo, duración, multiplicadores efectivos, presupuesto consumido y efectos sobre actividad/estabilidad.

## Relación con los órganos

- disinhibition consume el umbral de escritura Delta y el gate/inhibición compartidos.
- modulator_gain modula el tercer factor M y la vida media de la traza de elegibilidad Delta existente.
- plasticity_window modula la vida media de tags compartidos y el umbral temporal de poda, cuando esos órganos estén implementados.
- attentional_gate escala el gate existente con la novedad/incertidumbre compartida.

No se crearán theta, M, tau_e, tag_tau, prune_thresh o gate_gain duplicados si ya existe un estado equivalente. El contexto común de señales será la única fuente de verdad.

## Benchmark requerido

Cada uno de los cuatro perfiles tendrá control neutro y candidata emparejados, cada uno de 100 pasos, cinco ventanas de 20. Se aplican las reglas estrictas de NEUROPLASTICITY_BENCHMARK_GATE_ES.md: la candidata debe bajar simultáneamente tiempo total y loss media, sin empeorar la última ventana ni romper invariantes.

Se prueban primero los perfiles aislados. Después se evaluarán combinaciones factoriales pequeñas para detectar interacciones (por ejemplo, desinhibición + atención), evitando multiplicar todos los factores a la vez. Un modo que no encuentre una variante elegible queda desactivado; no se fuerza su activación.

Los mecanismos periódicos (tags, poda, sueño y replay) deben incluir su coste amortizado. El benchmark corto es un filtro de ingeniería, no evidencia de generalización. No se usa holdout para ajustar estos perfiles.

## Hipótesis falsables

1. La desinhibición mejora loss sin aumentar excesivamente la actividad efectiva ni el número de escrituras.
2. El aumento de M/vida de elegibilidad mejora retención de actualizaciones útiles sin saturar las trazas.
3. La ventana ampliada mejora la captura de tags útiles más de lo que aumenta interferencia y coste de poda aplazada.
4. El gate atencional mejora la relación entre señal y ruido y reduce loss, sin penalizar el throughput por encima de la mejora obtenida.

Un perfil solo se considera útil si supera el gate emparejado completo. Ninguna analogía biológica sustituye la medición.


## Estado de implementación en la rama de integración

Los cuatro perfiles ya tienen controles ejecutables en `model/plasticity.py` y se aplican al gate Delta desde `train/train.py`. Sus valores actuales son hipótesis iniciales, no ganadores de benchmark:

- `disinhibition`: `threshold_scale=0.85` e `inhibition_scale=0.8`.
- `modulator_gain`: `modulator_gain=1.5` y `eligibility_retention=0.96`.
- `plasticity_window`: `eligibility_retention=0.98` y tagging/capture activado.
- `attentional_gate`: `attention_gain=0.4`, aplicada a la sorpresa que ya calcula Delta.

Además, la rama contiene los siguientes mecanismos experimentales, desactivados salvo configuración explícita:

1. Contexto inmutable de sorpresa e incertidumbre compartido entre capas consumidoras; el contexto deriva incertidumbre una vez.
2. Una sola señal normalizada de check-loss alimenta glucosa, dolor y placer; los órganos conservan sus respuestas independientes, pero no recalculan la variación de loss.
3. Tagging/capture local con recurso global retardado y decaimiento configurable.
4. Umbral intrínseco adaptativo, lento y acotado, gobernado por actividad de escritura.
5. Poda simulada mediante máscara de uso/fuerza. No borra parámetros ni reduce todavía el tamaño físico del modelo.
6. LTD experimental sobre trazas elegibles protegidas por tags; no se presenta como LTD sináptica biológica ni como modificación directa de matrices de pesos.
7. Aprendizaje one-shot limitado a escrituras de memoria auxiliar en función de incertidumbre; nunca modifica el learning rate global.
8. Ganancia homeostática multiplicativa por canal, acotada entre 0.8 y 1.2.
9. Replay comprimido/priorizado por uso de slots de memoria, con ratio configurable.
10. Downscaling acotado durante sueño sobre matrices del MLP de replay.
11. Probe ELM de características aleatorias fijas y readout ridge analítico, aislado y offline; no toca los pesos de LUMEN ni el trainer autoregresivo.
12. La traza/modulador de Delta existente sigue cubriendo la familia de aprendizaje de tres factores y recompensa distal rate-based; no hay spikes ni equivalencia literal con STDP.

Las nuevas memorias persistentes se inicializan a valores neutros al migrar checkpoints anteriores. Las opciones experimentales se guardan en `runtime_profile`. Ninguna queda promovida por el mero hecho de estar implementada.

### Límites deliberados

No se ha implementado poda física de conexiones, un modelo spiking pre/post, síntesis proteica biológica, ni replay hipocampal de secuencias comprimidas. La memoria actual almacena vectores de contexto, no trayectorias neuronales. Para esos casos el nombre de la hipótesis no debe confundirse con equivalencia mecanística.

El perfil neutro conserva identidad en sus factores. Cada mecanismo debe pasar pruebas de propiedades, migración y el benchmark emparejado antes de poder activarse en una configuración de entrenamiento promovida.


### Perfiles personalizados para ablaciones

El modo `custom` permite evaluar un único factor aislado (o una combinación declarada explícitamente) sin activar el preset completo. Sus valores siguen sujetos a los mismos límites. El modo `neutral` continúa siendo identidad exacta y rechaza cualquier override: así el torneo puede distinguir un control genuinamente neutro de una ablación personalizada.
