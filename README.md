# RONIN-LUMEN

**Arquitectura experimental de modelos pequeños: estado recurrente gobernado por sorpresa, actividad escasa, homeostasis, memoria de varios tiempos y consolidación fuera de línea. CPU primero.**

RONIN-LUMEN nace del refactor de WISE-RONIN V2. No es una carrera por entrenar el modelo más grande ni una promesa de superar a los modelos fundacionales. La pregunta es más concreta: **¿podemos diseñar un modelo pequeño que aprenda sin quedar atrapado en unas pocas respuestas, plantillas o atractores dominantes?**

> **Estado científico:** arquitectura implementada en primera versión; todavía no está validada como sistema de aprendizaje. Las hipótesis se tienen que ganar con entrenamiento real, holdouts congelados, controles y ablaciones.

## Índice

- [Qué estamos intentando cambiar](#qué-estamos-intentando-cambiar)
- [La arquitectura](#la-arquitectura)
- [Qué aporta la inspiración neurocientífica](#qué-aporta-la-inspiración-neurocientífica)
- [Qué tiene de propio el trabajo](#qué-tiene-de-propio-el-trabajo)
- [Estado real y próximos hitos](#estado-real-y-próximos-hitos)
- [Documentación y código](#documentación-y-código)
- [Reproducir el smoke test](#reproducir-el-smoke-test)
- [Principios de ingeniería](#principios-de-ingeniería)

## Qué estamos intentando cambiar

La receta dominante de muchos modelos de lenguaje sigue una familia arquitectónica cuyo punto de inflexión moderno fue el Transformer de 2017: capas apiladas de autoatención y redes feed-forward, entrenadas para predecir el siguiente token. Esa arquitectura fue un avance decisivo; no sostenemos que sea obsoleta ni que escalarla sea inútil. Lo que rechazamos es asumir que **más parámetros, más datos y más cómputo son la única respuesta** a los problemas de estabilidad y degradación de un modelo pequeño.

WISE-RONIN V2 mostró un problema práctico: reducir la pérdida no garantizaba conservar variedad ni conducta útil. El modelo podía adquirir una voz reconocible y, al continuar entrenando, derivar hacia respuestas repetitivas, plantillas o un atractor KOAN. LUMEN convierte esa experiencia en una pregunta de arquitectura: ¿qué mecanismos de escritura, actividad, memoria y consolidación pueden limitar esa deriva?

## La arquitectura

```text
Token
  -> Embedding BPE
  -> Presupuesto por token
  -> Estado Delta recurrente gobernado por sorpresa
  -> Homeostasis + Top-K
  -> Memoria elástica / banco lento
  -> Router
       -> MLP de lenguaje
       -> KAN de funciones spline
  -> Logits de token
  -> Objetivos auxiliares latentes durante entrenamiento

En intervalos de entrenamiento:
  -> replay / consolidación offline ("sueño")
  -> reinicio de estados rápidos
```

| Mecanismo | Función prevista | Pregunta experimental |
|---|---|---|
| Estado Delta con puerta de sorpresa | Actualizar el estado cuando el error de predicción lo justifica; decaer si no | ¿Reduce escrituras inútiles sin perder contexto? |
| Homeostasis | Regular la frecuencia de activación de las unidades | ¿Evita que unas pocas neuronas monopolicen la representación? |
| Top-K | Mantener una fracción pequeña de activaciones | ¿Mejora selectividad y coste sin destruir información? |
| Presupuesto de energía | Repartir capacidad entre estado, memoria y cabeza | ¿Evita que un subsistema absorba todo el cómputo? |
| Memoria elástica y lenta | Separar adaptación rápida de patrones consolidados | ¿Reduce interferencia y olvido? |
| Router MLP/KAN | Separar generación lingüística de operaciones | ¿Aporta especialización medible o colapsa a una sola ruta? |
| Sueño / replay | Reprocesar registros internos sin nuevos ejemplos externos | ¿Consolida sin degradar lo aprendido? |
| Predicción latente JEPA-style | Añadir una señal auxiliar sobre representaciones futuras | ¿Aporta algo por encima de la pérdida de token? |

El modelo actual tiene **2.281.841 parámetros** según el recuento integrado registrado en CI. Cumple el techo de 6 millones, pero todavía no alcanza el objetivo de diseño de 3–4 millones. El tamaño final en disco de 20–30 MB tampoco está demostrado por ese recuento.

### Qué aporta la inspiración neurocientífica

La inspiración no es decorativa: se usa para formular mecanismos concretos y preguntas falsables. La correspondencia entre biología y código es una **abstracción de ingeniería**, no una simulación fiel del cerebro. Las fuentes y sus límites están desarrollados en [Neurociencia convertida en decisiones de arquitectura](docs/NEUROCIENCIA_Y_ARQUITECTURA_ES.md).

En resumen:
- Plasticidad homeostática: regular actividad para que el aprendizaje no empuje el sistema hacia actividad descontrolada o silencio.
- Codificación escasa: representar cada entrada con pocas unidades activas.
- Codificación predictiva y error de predicción: usar la discrepancia entre lo esperado y lo observado como señal de actualización.
- Sistemas de memoria complementarios: distinguir adaptación rápida e integración lenta.
- Sueño y homeostasis sináptica: explorar consolidación y renormalización offline.

No todo procede de neurociencia: la regla Delta tiene una genealogía propia en modelos secuenciales modernos; KAN es una propuesta matemática; JEPA es una línea de aprendizaje de representaciones. La documentación identifica cada procedencia sin apropiarse del trabajo ajeno.

## Qué tiene de propio el trabajo

La contribución que reivindicamos es **la selección, traducción e integración de estas ideas en un sistema pequeño, ejecutable en CPU y sometido a restricciones reales de memoria, tiempo y reproducibilidad**. No afirmamos haber inventado la homeostasis, la codificación escasa, la consolidación, la predicción latente, KAN ni las reglas Delta.

La decisión de diseño es distinta de limitarse a reducir una Transformer o aumentar su tamaño: repartir explícitamente la computación entre estado recurrente, memoria rápida/lenta y rutas especializadas; controlar la actividad; y tratar la degradación como objetivo de arquitectura y evaluación, no solo como un problema de limpieza del dataset.

La prueba del mérito no será el lenguaje grandilocuente: será que las ablaciones demuestren qué componente mejora qué métrica, a qué coste y en qué condiciones.

## Estado real y próximos hitos

<badge>Arquitectura integrada · datos en preparación</badge>

- **PR #13 — integrada:** arquitectura, módulos, trainer CPU, checkpoint/resume y smoke tests. [Ver PR #13](https://github.com/orbitalronin94/WISE-RONIN/pull/13).
- **PR #14 — abierta:** gates de datos, procedencia, licencias y tokenizer. [Ver PR #14](https://github.com/orbitalronin94/WISE-RONIN/pull/14).
- **Staging externo:** workflow [44890014413](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44890014413) completado. Artefacto de 19.000 candidatos; auditoría automática: 0 duplicados normalizados, 0 coincidencias exactas con holdout y 0 coincidencias largas de subcadena. El holdout conserva 500 filas y 55 prompts únicos.
- **No es corpus final:** faltan 5.000 filas propias curadas y 1.000 filas de código con licencia comprobada. Las fuentes externas descargadas siguen con `license_reviewed=false` y `rights_confirmed=false`; el resultado léxico no sustituye revisión humana ni semántica.
- **Tokenizer:** todavía no hay tokenizer final construido a partir del corpus aprobado.
- **Entrenamiento:** los runs de 10 pasos con datos dummy son pruebas de ejecución y reanudación, no aprendizaje. No se debe presentar ninguno como entrenamiento real.

La primera campaña con datos reales debe tener un manifiesto propio, hash del corpus y tokenizer, checkpoints separados y evaluación congelada. Si se hace una campaña provisional con una composición distinta de la receta final, se publicará como experimento provisional, nunca como resultado de la configuración de producción.

## Documentación y código

El [índice de documentación](docs/README.md) organiza las especificaciones, auditorías, datos y experimentos. Entradas principales:

- [Documento maestro V1](DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md)
- [Especificación V1](SPEC_RONIN_LUMEN_V1.md)
- [Neurociencia y decisiones de arquitectura](docs/NEUROCIENCIA_Y_ARQUITECTURA_ES.md)
- [Auditoría de la especificación](docs/SPEC_AUDIT.md)
- [Auditoría del bucle de entrenamiento](docs/TRAIN_LOOP_REFERENCE_AUDIT.md)
- [Extensión experimental V3.2 ZE](docs/RONIN_LUMEN_V3_2_ZE.md)
- [Procedencia y preparación de datos](data/README.md)
- [Contrato del corpus propio](data/owner/README.md)
- [Workflow de entrenamiento CPU](.github/workflows/train.yml)
- [Workflow de preparación de datos](.github/workflows/prepare-data.yml)

## Reproducir el smoke test

Desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
python -m train.train --dummy-m1 --steps 10 --phase 1
python -m train.train --dummy-m1 --resume checkpoints/step_10.pt --steps 10 --phase 1
```

El modo dummy solo comprueba forward, optimización, checkpoint y reanudación. Está limitado a diez pasos y no usa el corpus real.

El entrenamiento de un bloque con datos reales requiere el corpus y el tokenizer aprobados:

```bash
python -m train.train --steps 500 --phase 1
```

No ejecutar ese comando hasta cerrar los gates de datos, licencias, tokenizer y holdout. Los workflows de GitHub Actions son la vía reproducible prevista para los bloques CPU.

## Principios de ingeniería

### Carmack–Naka–Chapman: diseñar desde la restricción

- **Carmack:** comprender el coste real de memoria, caché, ancho de banda y operaciones antes de añadir complejidad.
- **Yuji Naka:** exprimir cada byte, rutina y recurso disponible mediante reutilización y diseño específico para la máquina.
- **Colin Chapman:** simplificar antes de añadir peso.

Aplicado aquí: CPU como objetivo, no como plan B; arquitectura antes que fuerza bruta; cada parámetro y cada fila de datos debe justificar su coste; y ningún benchmark se acepta sin controles, holdout y límites declarados.

---

*RONIN-LUMEN no pretende imitar un cerebro ni demostrar que una arquitectura pequeña vence a una grande. Pretende convertir ideas sobre aprendizaje, estabilidad y memoria en mecanismos computables, baratos y falsables.*
