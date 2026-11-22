# Glucosa, dolor y placer: tres reguladores separados

## Glucosa
El órgano energético existente conserva su lógica y su reserva. La glucosa responde a mejoras de la loss de comprobación y escala el presupuesto total de energía.

## Dolor
`Nociception` mantiene una EMA propia y detecta regresiones sostenidas de la loss media de comprobación. Su respuesta es acotada, con zona muerta del 0,5 %, saturación del 8 % y decaimiento.

## Placer
`Recompensa` mantiene otra EMA independiente y detecta mejoras sostenidas. No comparte estado con dolor ni con glucosa. Su efecto también está acotado y decae.

## Integración experimental 11k–16k
Los tres órganos se actualizan cada 100 pasos, usando la misma ventana de loss que alimenta la glucosa. Dolor y placer modulan exclusivamente el multiplicador interno de energía de LUMEN, dentro de [0,85, 1,05]; no alteran LR, optimizer, loss ni pesos directamente. La glucosa sigue actuando por su canal preexistente. Se registran las tres señales en logs y quedan serializadas en el checkpoint mediante buffers.

El corpus de entrenamiento de esta continuación ya fue visto por el modelo; por ello estas señales son homeostasis local, no una prueba de generalización. El holdout español de 2.431 preguntas permanece completamente separado: no se usa para alimentar ni recompensar los órganos. Solo se evalúa al final.

Son mecanismos computacionales inspirados en analogías biológicas; no implican experiencia subjetiva, placer consciente ni voluntad.
