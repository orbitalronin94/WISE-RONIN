# Dolor y placer: órganos afectivos de LUMEN

## Alcance

Esta rama añade dos reguladores independientes y probados, pero **no los conecta todavía al bucle de entrenamiento**. El entrenamiento que llega al paso 11.000 permanece intacto. La integración será una decisión posterior, respaldada por el holdout de 11k.

- `Nociception` (`model/pain.py`): detecta regresiones en la loss de comprobación y mantiene una señal de protección con decaimiento.
- `Recompensa` (`model/pleasure.py`): detecta mejoras en la loss de comprobación y mantiene una señal positiva con decaimiento.
- `Presupuesto`/glucosa no se modifica ni se reutiliza como señal afectiva.

## Señal y filtrado

Ambos órganos reciben la media de loss de una ventana de comprobación (por ejemplo, los 100 pasos que ya agrupa el entrenador), nunca la loss de un minibatch aislado. Cada órgano mantiene su propia EMA de referencia. La variación relativa se calcula contra esa referencia:

- Dolor: `max(0, (loss_actual - referencia) / referencia)`.
- Placer: `max(0, (referencia - loss_actual) / referencia)`.

Se aplica una zona muerta del 0,5 % para no reaccionar a pequeñas oscilaciones y una escala de saturación del 8 %. La señal interna se filtra y decae entre comprobaciones. Los umbrales son parámetros configurables, no constantes biológicas.

## Efectos propuestos, deliberadamente acotados

| Señal | Plasticidad | Gate | Consolidación |
|---|---:|---:|---|
| Dolor | `1 - 0.15 * pain` | `1 - 0.10 * pain` | freno proporcional |
| Placer | `1 + 0.10 * pleasure` | `1 + 0.05 * pleasure` | impulso proporcional |

Estos coeficientes son propuestas iniciales para una prueba A/B, no cambios activos. En particular, el placer no debe escribir automáticamente el minibatch actual en memoria: la asociación input/resultado requiere una traza identificable. El dolor tampoco debe apagar el aprendizaje ni reducir el LR por debajo de un suelo.

## Integración posterior al paso 11.000

1. Revisar la loss suavizada, estabilidad y holdout congelado hasta 11k.
2. Añadir ambos estados al checkpoint y verificar resume exacto.
3. En una primera prueba, conectar solo los factores de plasticidad/gate con límites y telemetría.
4. Diseñar la traza de memoria por patrón antes de activar refuerzo selectivo o sueño diferido.
5. Comparar baseline, dolor solo, placer solo y ambos; mismo corpus, semilla, presupuesto de pasos y holdout.
6. Revertir si mejora la loss de entrenamiento pero empeora el holdout o la estabilidad.

## Límite conceptual

Son señales computacionales de regulación y recompensa inspiradas en analogías neurobiológicas. Su implementación no demuestra experiencia subjetiva, placer consciente ni voluntad.
