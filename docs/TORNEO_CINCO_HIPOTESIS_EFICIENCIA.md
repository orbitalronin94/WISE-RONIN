# Torneo paralelo: cinco hipótesis de eficiencia para LUMEN

**Estado:** protocolo preparado; no ejecutar entrenamiento hasta que cada brazo tenga implementación, pruebas de equivalencia y procedencia verificadas.

## Pregunta experimental

¿Puede LUMEN reducir cómputo por paso y/o actualizaciones del optimizador conservando capacidad en el holdout congelado? Se medirán por separado velocidad, trabajo omitido, calidad predictiva y estabilidad. Ningún ahorro de FLOPs se considerará éxito si deteriora la calidad más allá del margen preinscrito.

## Regla de simultaneidad

Los brazos se lanzarán en una única ejecución de GitHub Actions mediante una matriz paralela, cada uno en un runner independiente. Todos partirán del mismo checkpoint inmutable, mismo corpus/token-cache/tokenizer, misma secuencia de batches, seed, número de pasos, scheduler y perfil CPU. No se encadenarán brazos ni se compartirán checkpoints modificados. El baseline y los cinco tratamientos correrán al mismo tiempo; el brazo combinado también correrá en paralelo.

## Matriz de brazos

| ID | Brazo | Hipótesis aislada | Implementación / condición |
|---|---|---|---|
| B0 | Control | Ruta actual | Perfil y trainer de referencia, sin nuevas intervenciones |
| H1 | Sparse firing | Activar menos unidades reduce coste sin perder señal útil | Comparar Top-K actual (16/32) con selección más dispersa por grupo; definir grupos y regla de desempate. No confundir con el Top-K ya activo |
| H2 | Predict-then-correct | La sorpresa baja permite omitir correcciones redundantes | Evaluar gate con señal suavizada e histéresis; conservar observación y scheduler. Separar backward omitido de actualización omitida |
| H3 | Inhibición selectiva | La inhibición moderada reduce competencia improductiva | Comparar perfil neutral con desinhibición/inhibición 0.80; registrar actividad, tasa de escritura y gradientes |
| H4 | Aprendizaje local | Stop-gradient/local losses pueden evitar backward global | Prototipo acotado por bloque/capa; medir gradientes que llegan a cada frontera y comprobar que no quedan parámetros sin señal |
| H5 | Sueño comprimido | Replay del 20% puede consolidar sin repetir el forward completo | Comparar no-sleep, replay comprimido y sueño actual; exigir banco lento no vacío y replay de activaciones reales |
| HΣ | Combinado | La combinación produce ahorro adicional sin interacción destructiva | Solo después de que H1–H5 tengan pruebas individuales; activa conjuntamente los cinco mecanismos |

**Importante sobre el estado actual del código:**
- Top-K 16/32 ya está activo en `Homeostasis`; H1 no puede presentarse como mecanismo nuevo. Su tratamiento debe ser una variante más dispersa frente al control.
- El perfil `disinhibition` ya existe con `threshold_scale=0.85` e `inhibition_scale=0.8`; H3 debe medirlo frente al perfil neutral y verificar que el workflow realmente carga ese perfil.
- El surprise gate y lazy backward ya existen en el trainer, pero el umbral instantáneo no equivale a sorpresa suavizada con histéresis. H2 estudia esa diferencia, no vuelve a contar el gate existente como novedad.
- `train/sleep.py` ya ofrece replay comprimido y downscale, pero un checkpoint con banco lento vacío convierte el sueño en no-op. H5 debe validar ocupación y registrar si hubo actualizaciones reales.
- No se ha encontrado aún una ruta de aprendizaje local por bloque que pueda activarse con seguridad. H4 requiere implementación y prueba antes de entrar en el torneo; no simularla con una bandera vacía.

## Protocolo común

1. **Preparación:** verificar SHA256 de checkpoint, corpus, tokenizer y token-cache; registrar step, optimizer, scheduler, RNG, perfil y configuración.
2. **Smoke test:** cada brazo ejecuta un lote idéntico y verifica tensores finitos, gradientes esperados y checkpoint reanudable.
3. **Paridad:** cuando el mecanismo afirma ser una optimización exacta, comparar logits, loss, gradientes y estado con tolerancias explícitas. Cuando cambia el algoritmo, declarar que es una hipótesis funcional, no equivalencia.
4. **Torneo:** 1.000 pasos por brazo en runners separados, seed fija y mismo orden de batches. Repetir con otra seed solo si hay empate o variabilidad alta.
5. **Evaluación:** mismo holdout congelado de 2.431 preguntas para todos los brazos; exactitud, NLL, calibración (Brier/ECE), repetición, longitud de respuesta y resultados por categoría. Holdout nunca se usa para ajustar umbrales.
6. **Decisión:** publicar tabla completa y artefactos. No promover por throughput aislado. Un brazo que ahorre tiempo pero pierda capacidad queda rechazado o marcado como trade-off.
7. **Combinación:** HΣ se ejecuta en paralelo con los brazos individuales, pero su interpretación queda condicionada a que H1–H5 hayan superado sus gates. Si alguna intervención individual falla, se conserva HΣ como diagnóstico exploratorio y no como candidato a promoción.

## Métricas obligatorias

- pasos/s, segundos/paso y tiempo total;
- backward calls, optimizer updates, pasos gated, replay updates y fracción de actividad;
- loss total y cada componente, media móvil y últimos 100 pasos;
- holdout accuracy, NLL, Brier/ECE y métricas por categoría;
- pico de RAM, tiempo de forward/backward y tiempo Python;
- gradientes por módulo, parámetros sin gradiente y checks de finitud;
- hashes, seed, commit, checkpoint padre y configuración efectiva.

## Criterios de protección

- Ningún brazo puede cambiar tokenizer, corpus, holdout, batch, contexto, seed ni scheduler respecto al control.
- Los brazos H1–H5 se aíslan: una sola hipótesis por brazo.
- HΣ es el único brazo combinado.
- No se podan parámetros ni se eliminan órganos durante el torneo.
- El sueño no puede inventar recuerdos: solo replay de registros reales del banco; si el banco está vacío, el brazo se declara no evaluable y no se rellena con holdout.
- Un mecanismo local no puede desconectar silenciosamente capas inferiores; se exige medir gradiente y capacidad de cada bloque.
- No se fusiona ni se inicia entrenamiento largo a partir del resultado sin revisión de los artefactos.

## Estado y siguientes bloqueos

Este documento fija el diseño, no afirma que los cinco tratamientos estén ya implementados. Antes de despachar la matriz hay que:
1. crear toggles de H1/H2/H3/H4/H5 con configuración serializada en checkpoint;
2. escribir tests de paridad o de invariantes para cada brazo;
3. implementar el runner común y agregador de resultados;
4. validar que todos los jobs reciben exactamente los mismos inputs;
5. ejecutar el smoke test de matriz y comprobar que los siete brazos arrancan en paralelo.

La ejecución de entrenamiento queda deliberadamente separada de la preparación de este torneo.
