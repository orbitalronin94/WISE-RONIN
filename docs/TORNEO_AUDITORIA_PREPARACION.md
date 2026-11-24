# Auditoría de preparación del torneo de eficiencia

**Base inspeccionada:** `main` en `930c8db48dcadb310b9f38a1302313c09bb4a251`.  
**Estado:** auditoría estática inicial; no sustituye tests ni profiling. Ningún brazo está autorizado para entrenar hasta cerrar sus bloqueos.

| ID | Estado real en main | Trabajo necesario antes del torneo |
|---|---|---|
| H01 sparse firing | Top-K 16/32 y residual ya existen | Implementar variante estructurada distinta y telemetría de entropía/Gini/recuperación |
| H02 prediction/gate | Surprise gate y lazy backward existen | Añadir señal temporal suavizada/histéresis; comprobar scheduler y contabilizar backward real |
| H03 inhibition | Perfil de desinhibición 0.80 existe | Confirmar aplicación efectiva por órgano y medir trabajo realmente omitido |
| H04 local learning | No hay ruta local por capa integrada al objetivo principal | Diseñar una frontera, objetivo auxiliar y pruebas de gradiente; no simular con detach aislado |
| H05 sleep/replay | `train/sleep.py` existe; ratio default 1.0; banco puede estar vacío | Registrar ocupación, recuerdos usados, updates y compresión efectiva; replay 100/50/20 con el mismo banco |
| H06 temporal stability | Homeostasis e IntrinsicThreshold existen | Hacer barrido acotado y telemetría temporal; no asumir que elegibilidad 0.98 equivale a resistencia neuronal |
| H07 eligibility/reuse | Trazas persistentes y modulador ya existen | Probar retención y reutilización por separado; invalidación por cambios de contexto; gradiente/retención |
| H08 selective connectivity | Ruta residual y cuatro órganos existen; no hay gate inter-capa específico | Diseñar gate con fallback identidad y medir información/gradiente por frontera |
| H09 reward asymmetry | Glucosa comparte check-loss; dolor y placer mantienen señales distintas | Implementar barrido asimétrico acotado; comprobar que no se mezclan glucosa, dolor y placer ni se altera LR global |
| H10 tagging/one-shot | Primitivas TaggingCapture y OneShotGate existen | Verificar perfiles efectivos, banco real, captura y escritura; pruebas de recurso sin tag/tag sin recurso |
| H11 shared projections | K/V/Q ya se concatenan en una proyección por órgano Delta | No duplicar; buscar únicamente proyecciones idénticas entre módulos y comparar gradientes |
| H12 deferred diagnostics | `collect_diagnostics` y trace norm on-demand existen parcialmente | Inventariar cada métrica y sus consumidores; medir sincronizaciones restantes |
| H13 shared invariants | Cache de máscaras causales existe con `share_causal_masks` | Validar cache hit, límites, device/dtype y equivalencia; no cachear estado dinámico |
| H14 selective backward | Lazy backward + surprise gate ya existen | Comparar rutas y gate temporal; test de resume, optimizer/scheduler y holdout |
| H15 shared loss normalization | Repetition loss ya comparte log-softmax internamente | **Prototipo opt-in implementado** en `train/loss.py` y conectado al runtime profile; tests de paridad de loss/gradiente añadidos. Pendiente CI y benchmark de tiempo/memoria |
| H16 backend kernels | MKLDNN, OpenMP, foreach y fused scatter tienen soporte configurable | Microbench por kernel/operación y perfil idéntico; no asumir que fused AdamW acelera CPU |
| H17 vectorized memory | Escrituras secuenciales por muestra y reemplazo least-used | Vectorizar solo caso con equivalencia exacta de orden/contadores; fallback secuencial obligatorio |
| H18 Python/Torch sync | Hay `.item()` en chunking, gate de elegibilidad y diagnósticos | Clasificar cada sincronización funcional/diagnóstica; quitar solo las no funcionales o agruparlas sin alterar decisiones |
| H19 deferred accounting | Embedding hace `bincount` en cada forward | Verificar todos los consumidores de uso; contabilidad muestreada debe documentar error/ventana y mantener checkpoint |
| H20 homeostasis EMA | `m` y `effective_m` son buffers separados; effective_m gobierna bias | Probar consumidores completos de `m`, estado de checkpoint y compatibilidad antes de diferirla |

## Hallazgos importantes de cálculo compartido

1. **Delta ya comparte K/V/Q por órgano** mediante una única llamada lineal con pesos concatenados.
2. **Máscaras causales ya tienen cache global LRU** y opción de uso por los cuatro órganos.
3. **SignalContext ya centraliza sorpresa/incertidumbre**; hay que perfilar el coste de su validación de finitud.
4. **Repetition loss ya comparte log-softmax dentro de su cálculo**; el ahorro adicional posible está en compartirlo con CE, no rehacer la misma idea.
5. **Lazy backward y surprise gate ya están integrados**; el nuevo torneo compara variantes de política, no vuelve a implementar lo mismo.
6. **Homeostasis ya dispone de fused scatter** y mantiene una ruta residual; la sparsity actual no es apagado global de neuronas.
7. **Chunking adaptativo convierte la sorpresa a escalar Python** para escoger tamaño de chunk; es un candidato de sincronización, pero cambiar la granularidad puede cambiar el algoritmo.
8. **Memoria elástica conserva semántica secuencial de reemplazo**; cualquier vectorización debe demostrar paridad del estado final y del orden de escrituras.

## Orden seguro de implementación

1. Tests de referencia para salida, gradientes, buffers y estado de optimizador.
2. Optimizaciones matemáticas exactas y cálculo compartido (H11–H16) con flags off por defecto.
3. Reducción de sincronizaciones/contabilidad Python (H17–H20) con perfiles y equivalencia.
4. Variantes algorítmicas neuronales (H01–H10), cada una aislada y con telemetría.
5. Workflow matricial 21 brazos, smoke test y comprobación de simultaneidad.
6. Lanzamiento de 1.000 pasos solo tras todos los gates.

## Condición de seguridad

No se modifican parámetros, topología, tokenizer, corpus ni ruta activa de `main` en esta fase. Los cambios se hacen en ramas opt-in; toda ruta experimental debe conservar la ruta de referencia. Un test fallido bloquea el brazo, no se convierte en no-op para que el workflow parezca completo.
