# Benchmark emparejado: lazy backward + surprise gate (1.000 pasos)

**Estado: candidato experimental; no activado en el trainer de producción.**

- Run: [45376669093](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45376669093)
- Rama: `benchmark/lazy-surprise-1000steps`
- Commit medido: `f09721cada1731804d2028b8be5cea3569f392cf`
- Checkpoint común: LUMEN step 389.700
- Checkpoint SHA-256: `0da367ef1ead6155f9e5ca532f671d99e7bf147f18e37ff4817fcc0bd225e126`
- Corpus SHA-256: `9934b07abbc6f80c7a58d61a2acf2afd0dd57778c3207179732fba966dead7a9`
- Tokenizer SHA-256: `d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe`
- Seed: 42; 1.000 pasos por brazo; mismo runner e inputs.

## Resultado

| Métrica | Control main | Lazy + surprise |
|---|---:|---:|
| Loss media | **2,667454** | 2,680617 |
| Loss últimos 20 | **2,446195** | 2,467588 |
| Última loss | **2,293102** | 2,316835 |
| Tiempo | 252,096 s | **152,628 s** |
| Velocidad | 3,967 pasos/s | **6,552 pasos/s** |
| Actualizaciones optimizer | 1.000 | 853 |
| Surprise-gated | 0 | 147 |
| Gradientes reutilizados | 0 | 429 |
| Backward evitados | 0 | 576 |

El brazo combinado reduce el tiempo medido un **39,46 %** (aprox. 1,65× de throughput). Ambos mecanismos se ejercitaron: el gate omitió 147 actualizaciones y el lazy backward reutilizó gradientes en 429 pasos.

La contrapartida es una pérdida de calidad pequeña pero medible en este ensayo: la loss media empeora un **0,49 %** y la media de los últimos 20 pasos un **0,87 %**. Por tanto, el candidato **no supera el gate de calidad** definido para promoción automática. El JSON del benchmark marca la decisión como `PENDING`.

## Interpretación y límites

Este resultado justifica conservar e investigar la ruta de ahorro de cómputo, pero no demuestra equivalencia de calidad, convergencia ni generalización. Es un benchmark corto de entrenamiento emparejado, no un holdout. Tampoco mide el efecto sobre los checkpoints 400k.

La integración en el trainer debe permanecer explícita y reversible (feature flag), con el control como ruta por defecto hasta que una campaña más larga demuestre que la pérdida de calidad es aceptable. El gate de sorpresa y el lazy backward deben seguir reportándose por separado para que el ahorro no oculte cambios en la dinámica de aprendizaje.

## Reproducción

Workflow: [benchmark-lazy-surprise-1000.yml](../.github/workflows/benchmark-lazy-surprise-1000.yml)  
Harness: [benchmark_lazy_surprise_1000.py](../scripts/benchmark_lazy_surprise_1000.py)
