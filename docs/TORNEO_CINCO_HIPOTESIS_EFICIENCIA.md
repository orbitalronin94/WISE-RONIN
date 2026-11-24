# Torneo paralelo de eficiencia: hipótesis neuronales, forward/backward y Python

**Objetivo:** que todas las hipótesis compitan en la misma tanda. Cada brazo experimental ejecutará exactamente **1.000 pasos** y el control `main` actual ejecutará también **1.000 pasos**. Todos arrancan a la vez desde el mismo checkpoint padre y con los mismos datos.

**Estado:** protocolo ampliado. No se ha despachado ningún entrenamiento. Los brazos que aún no tengan implementación y pruebas no deben simularse con flags vacíos: primero se implementan, luego se ejecuta la matriz completa.

## 1. Diseño de la competición

- Un brazo control: **B0 = main actual, 1.000 pasos**, sin cambios experimentales.
- Un brazo independiente por hipótesis: **1.000 pasos cada uno**.
- Todos los brazos corren simultáneamente en runners independientes de una única ejecución de GitHub Actions.
- Todos usan el mismo checkpoint inmutable, corpus, token cache, tokenizer, holdout, secuencia de batches, seed, batch size, longitud de contexto, scheduler y entorno de hardware equivalente.
- Ningún brazo hereda pesos de otro. No hay encadenamiento ni transferencia entre candidatos.
- Se fija el orden de batches y se registra SHA de cada input. Si los runners no tienen hardware/recursos equivalentes, se informa y no se atribuye toda diferencia a la hipótesis.
- El holdout de 2.431 preguntas se evalúa después de los 1.000 pasos en todos los brazos, sin usarlo para ajustar parámetros.
- Primera ronda: hipótesis aisladas. La combinación de ganadores se deja para una segunda ronda; no se mezcla ahora porque impediría saber qué mecanismo produce el ahorro.

## 2. Brazos de la matriz

### Control y cinco hipótesis neuronales

| ID | Brazo | Hipótesis | Qué se compara |
|---|---|---|---|
| B0 | Main actual | Referencia | Código de main, 1.000 pasos |
| N1 | Disparo disperso | Menos unidades activas pueden reducir trabajo | Top-K actual frente a Top-K más disperso/estructurado. No se confunde el Top-K existente con una novedad |
| N2 | Predicción antes que corrección | Una sorpresa estable y baja permite omitir correcciones | Gate suavizado con histéresis frente al gate instantáneo existente; registrar backward y optimizer updates |
| N3 | Inhibición selectiva | La inhibición puede suprimir competencia inútil | Perfil neutral frente a desinhibición/inhibición 0.80; medir actividad y escrituras Delta |
| N4 | Aprendizaje local | Una señal local puede evitar parte del backward global | Stop-gradient y objetivo local en una frontera definida; brazo experimental, no equivalencia exacta |
| N5 | Sueño comprimido | Replay prioritario del 20% puede consolidar con menos trabajo | Sin sueño frente a replay comprimido; usar recuerdos reales y exigir banco no vacío |

### Hipótesis de ahorro de cómputo en forward/backward y backend

| ID | Brazo | Hipótesis | Qué se compara |
|---|---|---|---|
| C1 | Forward: compartir proyecciones | Proyecciones repetidas pueden fusionarse o reutilizarse sin duplicar operaciones | Salidas y gradientes equivalentes; medir tiempo de forward y pico de memoria |
| C2 | Forward: evitar trabajo diagnóstico | Normas, conversiones a Python y estadísticas no necesarias para la salida pueden diferirse | Diagnósticos activados solo en pasos de observación frente a cada paso; verificar que ninguna señal de control depende de ellos |
| C3 | Forward/Delta: reutilización exacta | Máscaras causales y tensores invariantes pueden reutilizarse entre órganos | Cache compartida frente a construcción por órgano; igualdad de outputs/gradientes y medición de coste |
| C4 | Backward: reducir trabajo redundante | El backward puede evitarse o reutilizarse en condiciones seguras | Comparar ruta main con lazy-backward/surprise gate existentes y variante con criterio suavizado; contabilizar backward reales y degradación |
| C5 | Loss: compartir normalización | CE y penalización de repetición podrían compartir `log_softmax`/normalización | Implementación actual frente a normalización compartida; paridad de cada loss y gradientes antes de medir velocidad |
| C6 | Backend/PyTorch: kernel y operaciones | Menos lanzamientos/temporales o una operación fusionada pueden reducir coste | Microbenchmark reproducible de kernels/operaciones candidatas; no aceptar cambios de precisión sin tolerancia predefinida |

### Hipótesis de ahorro Python

| ID | Brazo | Hipótesis | Qué se compara |
|---|---|---|---|
| P1 | Vectorizar memoria elástica | Bucles por muestra, `.item()`, `nonzero` y escrituras escalares pueden sustituirse por operaciones batch | Implementación actual frente a vectorizada; paridad exacta de asignación, reemplazo, memoria y estado |
| P2 | Reducir sincronizaciones Python↔Torch | Lecturas escalares y comprobaciones `.item()`/reducciones pueden forzar sincronización o bloquear el flujo | Agrupar lecturas y mover diagnósticos fuera del camino caliente; mantener las decisiones de control equivalentes |
| P3 | Evitar contabilidad redundante | Contadores/histogramas que no alimentan control ni checkpoint quizá no deban actualizarse cada batch | Instrumentación completa frente a muestreo diferido; buscar todos los consumidores antes de omitir una actualización |
| P4 | Simplificar actualización homeostática | La EMA de actividad de entrada `m` podría ser solo diagnóstica, mientras `effective_m` gobierna el bias | Confirmar todos los consumidores/tests; mantener el buffer y su compatibilidad de checkpoint aunque se difiera su actualización |

**Total de la primera matriz:** B0 + N1–N5 + C1–C6 + P1–P4 = **16 brazos simultáneos**, cada uno de 1.000 pasos. Los brazos son candidatos de investigación; no se afirma que todas las variantes estén ya implementadas ni que todas vayan a ahorrar tiempo.

## 3. Hechos del código que condicionan la implementación

- `Homeostasis` ya aplica Top-K 16/32 al update homeostático. El flujo residual `x` sigue pasando; los no seleccionados no reciben gradiente cero global. N1 debe medir una variante estructurada, no describir el mecanismo actual como WTA completo.
- El surprise gate y lazy backward ya existen. El gate actual mira loss escalar de minibatch; N2 debe estudiar estabilidad temporal y no contar el mecanismo existente como nuevo.
- El perfil `disinhibition` existe con `threshold_scale=0.85` e `inhibition_scale=0.8`. N3 tiene que verificar que el perfil llega realmente a todos los órganos.
- `train/sleep.py` implementa replay de memoria con compresión configurable, pero el ratio por defecto es 1.0 y un banco vacío no consolida nada. N5 debe validar ocupación natural y replay efectivo.
- Hay primitivas de plasticidad local, pero no equivalen a una ruta local por capa integrada en el forward principal. N4 necesita implementación explícita y tests.
- El perfilado previo encontró backward dominante, pero también overhead Python apreciable. C4 y P1/P2 deben informar tiempos por separado.
- La máscara causal compartida ya tiene una opción en Delta; C3 debe probar la ruta efectiva, no añadir una segunda cache redundante.
- La loss combina CE, JEPA, diversidad, router y repetición. C5 debe preservar cada componente, no solo el total.
- `Homeostasis.m` se actualiza cada batch; `effective_m` gobierna la adaptación del bias. P4 no puede eliminar ni cambiar la semántica del buffer sin revisar consumidores, serialización y tests.

## 4. Gates antes de lanzar los 16 brazos

1. Cada hipótesis debe tener implementación real y configuración registrada en el checkpoint/log.
2. Prueba de paridad de logits, loss, gradientes y estado cuando se afirme equivalencia exacta.
3. Para cambios de algoritmo (N1–N5 y gate suavizado), pruebas de invariantes y métricas de capacidad, no se exige igualdad artificial.
4. Cada brazo debe completar un smoke test de un batch y guardar/reanudar checkpoint.
5. Validar que la matriz pasa los mismos hashes de checkpoint, corpus, tokenizer, token cache y holdout a todos los runners.
6. Verificar que B0 apunta al SHA exacto de `main` congelado al crear la campaña; no a una rama de hipótesis.
7. Si un brazo no supera sus tests, se arregla antes de la campaña completa. No se sustituye por un no-op ni se reduce el número de brazos sin dejarlo documentado.

## 5. Métricas que decide el torneo

**Cómputo:** pasos/s, segundos/paso, tiempo total, forward, backward, Python, optimizer updates, backward calls, sincronizaciones, pico RAM y tamaño de checkpoint.

**Aprendizaje:** loss total y componentes, ventana móvil, NLL de tokens, gradientes por módulo, parámetros sin gradiente, actividad Top-K, escrituras Delta, ocupación del banco y replay updates.

**Capacidad:** exactitud y NLL en el mismo holdout de 2.431 preguntas; Brier/ECE si las probabilidades están disponibles; repetición, longitud y desglose por categoría.

**Procedencia:** commit, checkpoint padre, SHA de corpus/tokenizer/cache/holdout, seed, configuración efectiva, runner y artefactos completos.

## 6. Decisión

No gana necesariamente el brazo más rápido. Gana un mecanismo solo si reduce cómputo medible y conserva la capacidad dentro del margen acordado. Si acelera a costa de exactitud/NLL, se registra como trade-off y no se promociona automáticamente. No se podan pesos ni se eliminan órganos en esta ronda.

La campaña de 1.000 pasos es un torneo de cribado, no una prueba definitiva de generalización ni autorización para un entrenamiento largo. Los ganadores pasarán después a una validación más extensa y a un holdout independiente si corresponde.

## Estado

Este documento define la matriz completa de 16 brazos. **No se ha iniciado entrenamiento ni se ha despachado el workflow.** Falta implementar y validar los toggles, tests y workflow matricial. La campaña solo estará lista cuando los 16 brazos puedan correr de forma real y simultánea con 1.000 pasos cada uno y B0 de main con 1.000 pasos.
