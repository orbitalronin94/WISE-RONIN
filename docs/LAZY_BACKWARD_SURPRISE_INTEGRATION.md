# Integración de lazy backward + surprise gate en LUMEN

**Estado: activados en los siguientes bloques LUMEN de la campaña emparejada**, con flags explícitos y reversibles. Transformer permanece como control sin estos mecanismos.

## Evidencia que motiva la integración

Benchmark emparejado de 1.000 pasos, desde el mismo checkpoint 389.700, mismo corpus/token-cache/tokenizer, seed y runner:

| Métrica | Control | Combinado |
|---|---:|---:|
| Tiempo | 252,096 s | 152,628 s |
| Throughput | 3,967 pasos/s | 6,552 pasos/s |
| Loss media | 2,667454 | 2,680617 |
| Loss últimos 20 | 2,446195 | 2,467588 |

Ahorro medido: **39,46 % de tiempo** (1,65× throughput). La loss media aumentó **0,49 %** y la media de los últimos 20 pasos **0,87 %**. Es una señal favorable de coste/beneficio, no una prueba de equivalencia de calidad ni de generalización.

Benchmark original: [run 45376669093](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45376669093).

## Comportamiento implementado

- **Surprise gate:** cuando la loss escalar observada es `<= 1.8`, se registra la observación y se omiten backward y actualización del optimizador.
- **Lazy backward:** en pasos locales alternos elegibles se reutiliza el último gradiente completo, multiplicado por `0.9`; se vuelve a aplicar gradient clipping antes de cada actualización.
- **Precedencia:** el surprise gate tiene prioridad; una observación gated no consume un turno de reutilización.
- **Señales y regulación:** la loss sigue entrando en la ventana de glucosa/dolor/placer y en los diagnósticos; no se omiten los órganos ni la observación del modelo.
- **Scheduler:** avanza con cada observación, también en pasos gated. Esto conserva la invariancia ya exigida por los checkpoints (`scheduler.current_step == global_step)); el benchmark inicial no avanzaba scheduler en pasos gated, por lo que este detalle queda declarado como diferencia de integración.
- **Reanudación:** los gradientes cacheados son transitorios y no se serializan. Tras reanudar, el primer paso calcula backward completo y luego vuelve a alternar. Pesos, optimizador, RNG y scheduler se guardan como antes.
- **Reversibilidad:** `--lazy-backward` y `--surprise-gate-loss 1.8` activan cada mecanismo. Omitir ambos flags restaura la ruta original. El workflow reutilizable los pasa solo al trainer LUMEN; el Transformer no cambia.

## Observabilidad

Cada bloque informa `optimizer_updates`, `surprise_gated`, `lazy_reused` y `backward_calls` en líneas `compute_saving`. El checkpoint registra la política usada. Se debe comparar loss por ventanas, throughput, estabilidad numérica y holdout antes de atribuir una mejora de aprendizaje.

## Criterio de seguimiento

La primera campaña larga debe revisar si la diferencia de loss se mantiene acotada y si el holdout de LUMEN no empeora materialmente. Si la degradación crece, se puede retirar el flag del workflow sin migrar pesos ni alterar la arquitectura.
