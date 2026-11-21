# Continuación longitudinal: pasos 2.000 → 4.000

## Diseño

Esta fase continúa exactamente los dos checkpoints finales del run [44922965444](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44922965444), artefacto `14715857574`. No reinicia pesos ni optimizador y no vuelve a construir ni tokenizar el corpus.

- Padre común: paso global 2.000; semilla 42; 1.670.175 tokens vistos.
- Corpus y tokenizer: artefacto preparado `ronin-lumen-prepared-data-44905396366`, con sus SHA-256 verificados contra los checkpoints.
- LUMEN V3.1: 2.281.841 parámetros; LR objetivo `2e-5`; grad clip 0,5.
- Transformer: 2.295.168 parámetros; LR objetivo `5e-5`; grad clip 1,0.
- Ambos: batch 8, contexto 128, semilla 42, weight decay 0,01.
- Preservación exacta de pesos, momentos Adam, RNG y contador de tokens del padre.

## Por qué se prepara un nuevo scheduler

El scheduler anterior termina en el paso 2.000 con LR cero. Reanudarlo sin cambiar el calendario dejaría el optimizador sin aprender. `scripts/prepare_step2000_continuation.py` valida el checkpoint y cambia únicamente los metadatos del scheduler:

1. Nuevo tramo global 2.000 → 4.000.
2. Calentamiento lineal de 20 pasos desde LR cero hasta el LR objetivo ya establecido para cada modelo.
3. Decaimiento coseno hasta LR cero en el paso 4.000.

El script registra SHA-256 del checkpoint original, pesos, estado del optimizador y RNG. Comprueba que pesos, momentos, RNG y tokens no cambian al preparar la continuación.

## Ejecución paralela

El workflow `.github/workflows/continue-2000-4000.yml` separa el trabajo en tres etapas:

1. **prepare-parent**: recupera los artefactos inmutables, valida los dos padres y prepara el único holdout congelado.
2. **train**: dos runners independientes y simultáneos, uno para LUMEN y otro para Transformer. Cada modelo ejecuta cuatro bloques reanudables de 500 pasos: 2.000→2.500→3.000→3.500→4.000.
3. **final-holdout**: solo cuando ambos entrenamientos terminan, ejecuta una única evaluación emparejada en el paso 4.000 y genera el manifiesto.

No se ejecutan holdouts intermedios ni evaluaciones de generación durante el entrenamiento. Se conservan los logs de loss cada 100 pasos, los checkpoints cada 250 pasos y los guardarraíles internos de seguridad de LUMEN.

## Criterio de lectura

La comparación de esta fase se basa en:

- trayectoria de loss durante 2.000 pasos adicionales;
- holdout GSM8K (100 preguntas seleccionadas de la partición test fijada) y holdout histórico completo;
- EOS, diversidad y repetición como diagnósticos del único holdout final.

Si aparecen señales consistentes de mejora, las siguientes fases podrán reintroducir evaluaciones intermedias. Si no, no se multiplicarán comprobaciones que no cambian la decisión experimental.
