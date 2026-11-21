# RONIN-LUMEN V3.2 — extensión experimental ZE / LC-NE

**Estado:** propuesta implementada como vía experimental optativa. No afirma equivalencia biológica ni una reducción demostrada de perseveración.

Esta extensión conserva V3.0 como baseline. Se activa mediante una bandera explícita y debe evaluarse con el mismo holdout congelado de 500 filas y pruebas sintéticas controladas de atractores. Ningún registro del holdout puede entrar en entrenamiento ni en replay de sueño.

## Límite de la evidencia

La neurociencia citada motiva hipótesis, no leyes directas de ingeniería:

- El marco variacional/de energía libre de Friston inspira un compromiso entre precisión y complejidad. Una pérdida neuronal con regularización KL es solo un objetivo inspirado en esa familia de ideas; no es automáticamente energía libre biológica.
- El replay y la homeostasis sináptica motivan experimentos de consolidación; no demuestran que un calendario 1/3/7 sea óptimo para este modelo.
- La investigación sobre locus coeruleus–noradrenalina (LC–NE) inspira hipótesis de modulación adaptativa y compuertas de aprendizaje. Los umbrales PE 0.3/0.8 son valores iniciales de ingeniería que deben calibrarse con estadísticas exclusivas del entrenamiento.
- Las afirmaciones atribuidas a artículos de 2025/2026 —incluidos Burke/Pednekar y el resultado de *Cell Reports* de diciembre de 2025— requieren verificar título y DOI exactos antes de presentarse como evidencia asentada. «ZE 2026» es el nombre que el propietario da al mecanismo propuesto, no una referencia bibliográfica verificada por sí misma.

## 1. Modelo dual M_A / M_B

- M_A es la pila causal Delta existente: cuatro capas de 256 dimensiones.
- M_B es un predictor Delta de 128 dimensiones. Recibe una proyección contrafactual aprendida de la representación oculta actual.
- F_A y F_B son estimaciones normalizadas del error de predicción/sorpresa. ΔF = |F_A − F_B| mide únicamente la discrepancia en magnitud del error; **no** basta para concluir que las predicciones coinciden.
- El controlador también sigue la discrepancia entre estados de predicción en el espacio compartido de 256 dimensiones. Con poco conflicto mezcla ambas rutas. Con conflicto alto selecciona la ruta con menor error y reinicia el estado recurrente más débil.
- M_B es un proxy contrafactual aprendido, no un verdadero contrafactual de «sustituir el token anterior». La supervisión contrafactual real requeriría ejemplos emparejados o intervenidos.

## 2. Compuerta ternaria de error de predicción inspirada en LC–NE

Bandas PE normalizadas iniciales:

- **Baja:** PE < 0.3 — ruta de edición de memoria elástica, con escala de gradiente reducida.
- **Media:** 0.3 ≤ PE ≤ 0.8 — no escribir en memoria.
- **Alta:** PE > 0.8 — escribir un registro nuevo en una ranura de memoria lenta; no sobrescribir un registro existente.

Son umbrales de partida, no constantes universales. Registrar cuantiles de PE y recuentos por banda. No recalibrar silenciosamente los umbrales con datos del holdout.

## 3. Operador ℰ (descorrelación de contexto)

Cuando el conflicto entre los dos modelos persiste durante más de 10 tokens consecutivos, se estima una dirección contextual a partir de la discrepancia M_A/M_B y se elimina su componente de rango uno del estado Delta rápido:

\[
S' = S - \frac{(S e)e^T}{e^T e + \epsilon}.
\]

Es una heurística de proyección. No puede identificar por sí sola una etiqueta semántica como «KOAN» o «matemáticas»: la dirección es un proxy interno de discrepancia/contexto. Registrar número de activaciones, norma del estado eliminado y si el conflicto disminuye después. Limitar el operador para evitar borrar estado útil.

## 4. Sueño adaptativo y replay de estilo IRI

El sueño se activa cuando la ocupación de la memoria lenta alcanza el 80%, en lugar de seguir un reloj fijo de 500 pasos. Los intervalos de replay 1/3/7 se implementan como una pequeña secuencia determinista sobre registros consolidados.

Es una heurística de espaciado que se puede probar. No se afirma que Burke et al. o Pednekar et al. hayan establecido esos intervalos exactos para modelos de lenguaje neuronales.

## 5. Regularizador de energía libre inspirado en formulaciones variacionales

\[
F = L_{token} + \beta\, KL(q(S_t)\|q(S_{t-1})) + \text{pérdidas auxiliares existentes}.
\]

La implementación transforma cada fila del estado Delta en una distribución categórica mediante softmax y calcula KL entre pasos. El beta inicial es 0.01. Registrar el término de complejidad por separado y ejecutar ablaciones con beta=0. Esta KL es un regularizador sobre distribuciones del estado interno, no una medida física de energía calibrada.

## Criterios de aceptación

1. V3.0 y V3.2 usan las mismas semillas, datos, tokenizer, presupuesto de optimización y muestreo.
2. Ejecutar al menos 5 semillas en pruebas sintéticas de atractores que solo usen datos de entrenamiento: bucle KOAN, atractor de token repetido, pista de contexto falsa y conflicto contexto aritmético/KOAN.
3. El holdout congelado permanece byte a byte idéntico y excluido de entrenamiento, tokenizer y replay.
4. Informar repetición/recuento KOAN, exactitud factual, exactitud aritmética, diversidad, pérdida, frecuencias de bandas PE, conflicto M_A/M_B, activaciones ℰ, ocupación de memoria y número de ciclos de sueño.
5. Rechazar la extensión si cae la exactitud factual/matemática, las neuronas muertas superan el 20% o las mejoras desaparecen entre semillas.
6. No afirmar públicamente que reduce la perseveración hasta superar las ablaciones.

## Despliegue

- V3.0 sigue siendo la configuración por defecto.
- V3.2 se activa con `--v32-ze`.
- Los checkpoints registran la bandera de funcionalidad y el estado V3.2.
- La extensión no se convierte en configuración predeterminada hasta superar las pruebas de runner, auditoría de corpus/licencias y evaluación congelada.
