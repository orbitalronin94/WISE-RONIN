# Próximo torneo LUMEN: solo mecanismos pendientes

**Estado:** preparado para ejecución manual después de cerrar y verificar el train 185.700 → 235.700. No se ejecuta durante ese entrenamiento.

## Alcance cerrado

El control es el `main` congelado en `19986f27925e3ccd2c965bbad6ebca2fa4db9e57`. Los mecanismos ya integrados y validados en la base no vuelven a competir como novedades. En particular, quedan fuera del torneo:

- Señal compartida y órganos glucosa/dolor/placer.
- Ganancias de dolor/placer y presupuesto energético.
- Desinhibición (ganador previo `inhibition_scale=0.80`).
- Ventana de elegibilidad (ganador previo `eligibility_retention=0.98`).
- Umbral LTP y ganancia moduladora, ya evaluados en la campaña anterior.
- Gate atencional, que ya fue candidato en la campaña anterior.

El torneo se limita a ocho capacidades que siguen pendientes de promoción: etiquetado/captura, umbral intrínseco, poda simulada, LTD protegida, memoria one-shot, escalado homeostático, replay priorizado y downscaling durante sueño. ELM se mantiene como probe offline: no es un mecanismo del train autoregresivo y no se incluye. Izhikevich y R-STDP son referencias/validaciones de la familia de tres factores, no órganos adicionales que se dupliquen.

## Las 20 hipótesis

| ID | Hipótesis | Mecanismos nuevos incluidos |
|---|---|---|
| 01 | Captura por tagging | Etiquetado/captura |
| 02 | Umbral intrínseco adaptativo | Umbral intrínseco |
| 03 | Poda simulada reversible | Poda simulada |
| 04 | LTD protegida | Tagging + LTD |
| 05 | Escritura one-shot acotada | Memoria one-shot |
| 06 | Escalado homeostático | Escalado |
| 07 | Replay priorizado/comprimido | Replay |
| 08 | Downscaling durante sueño | Sueño/downscaling |
| 09 | Tagging + umbral intrínseco | Tagging + umbral |
| 10 | Tagging + poda | Tagging + poda |
| 11 | Tagging + one-shot | Tagging + one-shot |
| 12 | Tagging + escalado | Tagging + escalado |
| 13 | Tagging + replay | Tagging + replay |
| 14 | Tagging + downscaling | Tagging + sueño |
| 15 | Umbral + poda | Umbral + poda |
| 16 | One-shot + escalado | One-shot + escalado |
| 17 | Poda + LTD protegida | Poda + tagging/LTD |
| 18 | Escalado + downscaling | Escalado + sueño |
| 19 | Tagging + replay + downscaling | Tagging + replay + sueño |
| 20 | Composición de los ocho pendientes | Los ocho mecanismos |

Las hipótesis 17–20 son deliberadamente combinatorias: permiten detectar interacciones y conflictos. No se atribuirá una mejora conjunta a un mecanismo individual. Si una combinación pasa, sus componentes deberán pasar también de forma aislada antes de promoverlos por separado.

## Protocolo de comparación

- Checkpoint padre esperado: Lumen step 235.700, del train emparejado run [45230206330](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45230206330).
- Datos, tokenizer y token cache: snapshot congelado v2 + personalidad factual del run [45149684913](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45149684913).
- Main y candidata se ejecutan secuencialmente en el mismo runner, desde el mismo checkpoint y RNG restaurado.
- 100 pasos por brazo en cribado, con 5 ventanas de 20 pasos.
- Se alterna el orden control/candidata para reducir sesgos de calentamiento.
- No se usa holdout para elegir mecanismos ni medir velocidad.
- Cada candidata conserva logs, perfiles, JSON de comparación y estado de sueño/replay.

## Gate obligatorio de promoción

Una hipótesis solo puede ser aprobada si cumple **todos** estos requisitos:

1. Loss media de 100 pasos estrictamente menor que el control.
2. Pasos por segundo estrictamente mayores que el control.
3. Loss de la última ventana de 20 pasos no peor que la del control.
4. Actividad muerta máxima no superior al control.
5. 100 pasos completos, métricas finitas y mecanismo realmente ejercitado.

Una mejora de loss sin mejora de throughput no pasa. Una mejora de throughput sin menor loss tampoco pasa. Si el sueño/replay no llega a ejecutarse o no hay memoria consolidada, el resultado queda como **no ejercitado**, no como aprobado ni como prueba negativa.

Los candidatos que superen el cribado todavía deberán pasar repeticiones emparejadas más largas, test de combinación y una validación de estabilidad/holdout antes de mergear. Este torneo no hace merge automático ni altera el train que está en curso.

## Ejecución

Workflow manual: [benchmark-pending-mechanisms-20.yml](../.github/workflows/benchmark-pending-mechanisms-20.yml).

El workflow espera el artefacto `lumen-approved-neuroplasticity-step235700-45230206330`. Si el train padre no termina correctamente o no publica ese checkpoint exacto, la campaña falla en la verificación previa y no ejecuta candidatas.

**Principio rector:** cada mecanismo pendiente debe ganarse su lugar mejorando simultáneamente aprendizaje y velocidad frente al control de `main`.
