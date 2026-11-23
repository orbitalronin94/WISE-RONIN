# Integración progresiva de neuroplasticidad en Lumen-Ronin

> Estado: investigación en curso en la PR [#71](https://github.com/orbitalronin94/WISE-RONIN/pull/71). Esta página describe el alcance de la propuesta, no afirma que todos los mecanismos hayan sido validados ni estén activos en el entrenamiento principal.

## Qué persigue la PR

La PR #71 reúne una nueva capa de coordinación neuroplástica para Lumen-Ronin. La idea central es que los órganos compartan señales y cálculos comunes —en particular sorpresa/incertidumbre y la señal derivada de la evolución de la loss— en lugar de recalcularlos de forma redundante en cada subsistema.

Sobre ese contexto común se investigan mecanismos que regulan cuándo y cómo se modifica el estado aprendido:

- **Señal compartida:** un contexto por forward para sorpresa e incertidumbre, reutilizable por los mecanismos que lo necesiten.
- **Glucosa, dolor y placer:** una señal normalizada de check-loss puede alimentar sus transformaciones específicas, conservando sus funciones distintas y evitando varias EMA equivalentes.
- **Modulación de plasticidad:** perfiles acotados y opcionales para umbral LTP, desinhibición, ganancia moduladora, duración de elegibilidad y gate atencional.
- **Etiquetado y captura sináptica:** conservar una etiqueta temporal para que una mejora posterior pueda consolidar cambios recientes relevantes.
- **Umbral intrínseco adaptativo:** explorar regulación de la facilidad con que una unidad cambia su actividad.
- **Poda simulada y reversible:** probar máscaras de actividad/conectividad sin destruir pesos ni hacer irreversible la investigación.
- **LTD protegida por etiquetas:** permitir debilitamiento de conexiones solo cuando no estén protegidas por señales de uso o consolidación.
- **Memoria one-shot:** evaluar una vía auxiliar de incorporación rápida de información.
- **Escalado homeostático:** compensar cambios persistentes de actividad para evitar que la plasticidad desestabilice el sistema.
- **Replay priorizado y comprimido:** favorecer durante la consolidación los registros con mayor valor de aprendizaje.
- **Downscaling durante sueño:** explorar una reducción regulada de actividad/pesos, respetando las trazas protegidas.

Los perfiles inspirados en farmacología son analogías computacionales acotadas, no simulaciones de fármacos ni equivalencias biológicas. El ELM se mantiene como probe analítico offline: no se le atribuye una mejora del entrenamiento que no haya producido.

## Cómo se incorporarán los cambios

No se pretende activar todo de golpe ni sacrificar velocidad por acumular mecanismos. La PR define un torneo de **18 evaluaciones emparejadas e independientes**. Cada candidata se compara con el control de `main` desde el mismo checkpoint y con los mismos datos, tokenizer, lotes, semilla y configuración.

El filtro previsto exige, simultáneamente:

1. menor loss media en los 100 pasos;
2. más pasos por segundo;
3. última ventana de 20 pasos no peor;
4. número máximo de unidades muertas no superior;
5. 100 pasos completos y métricas finitas.

Los candidatos que no superen el filtro permanecen registrados como pendientes, con sus métricas y el motivo del rechazo. No se descartan por ello: podrán corregirse y volver a evaluarse contra una referencia posterior. Los que sí superen las pruebas individuales podrán pasar a una composición conjunta y a una continuación de entrenamiento limitada, seguida de nuevas comprobaciones de estabilidad y holdout.

## Una integración gradual, no una promesa de activación inmediata

La intención del proyecto es **llegar a incorporar progresivamente todos los cambios que demuestren utilidad**, aunque haga falta más de una iteración. Algunos mecanismos podrán entrar pronto; otros necesitarán rediseño, optimización, más evidencia o quizá no superen nunca los criterios. Si una idea no mejora simultáneamente calidad y rendimiento bajo el protocolo acordado, no se activa solo por resultar atractiva.

Por eso la PR permanece en borrador mientras no haya CI verde, resultados completos del torneo, revisión de los candidatos aprobados y validación integrada. El código experimental disponible tampoco debe confundirse con un mecanismo ya activo en el entrenamiento principal.

La dirección es ambiciosa y paciente: compartir cálculos, eliminar redundancias, preservar los órganos útiles y conseguir que cada mecanismo nuevo se gane su lugar con resultados reproducibles.

## Estado y referencias

- [PR #71: implementación y estado vivo](https://github.com/orbitalronin94/WISE-RONIN/pull/71)
- [Protocolo de benchmark por órgano](NEUROPLASTICITY_BENCHMARK_GATE_ES.md)
- [Perfiles de modulación de plasticidad](PLASTICITY_MODES_ES.md)
