# LUMEN-WISE x5 — protocolo de preparación

Estado: **preparación técnica; no autorizado para entrenar**.

Esta rama separa el experimento del entrenamiento principal. No modifica ni reemplaza `main`. La rama se creó desde `39934820367b564710fe64bf40bd13fea6d5f20a`; antes del experimento se deberá rebasar o reconstruir sobre el commit exacto de `main` congelado cuando Corpus V4 esté verificado. Ese commit de arquitectura será el mismo para ambos modelos.

## Pareja experimental

| Modelo | Arquitectura | Configuración |
|---|---|---|
| WISE-LUMEN x5 | RoninLumen completo, con V3.2 ZE activado | `config/lumen_wise_x5.json` |
| Transformer x5 | Decoder-only causal, 4 bloques, atención estándar | `config/transformer_wise_x5.json` |

El perfil LUMEN escala conjuntamente dimensión, rango Delta, memoria, MLP, KAN y estado contrafactual ZE. Mantiene cuatro órganos Delta, el vocabulario compartido de 4.096 tokens, Top-K proporcional, homeostasis, JEPA, sueño y los órganos de regulación que estén presentes en el commit congelado. El perfil Transformer usa dimensión 488, 8 cabezas y FFN 1.464, con embeddings ligados.

El script `scripts/validate_lumen_wise_x5.py` instancia los dos modelos y sus referencias base, cuenta parámetros reales desde PyTorch y comprueba que cada uno queda en el entorno de 5× y que la diferencia de tamaño entre ambos no supera el 5%. Esa validación ejecutable prevalece sobre cualquier estimación escrita aquí. La inicialización del Transformer usa embeddings ligados con escala normal 0,02 (no la inicialización por defecto de `nn.Embedding`), y el validador exige una loss sintética inicial finita y en rango razonable; así se detecta una escala de logits rota antes del entrenamiento.

## Condiciones comunes pre-registradas

- Inicio desde cero: **step 0** para ambos; no heredar checkpoints ni pesos de pilotos.
- Duración: **25.000 actualizaciones** por modelo.
- Corpus V4 congelado idéntico, mismos shards, orden de muestreo y hash de datos.
- Tokenizador BPE idéntico, vocabulario 4.096 y SHA-256 congelado.
- Contexto 128; batch 8; seed 42.
- AdamW, LR inicial 1,5e-4, warmup 100, weight decay 0,01 y clipping 1,0.
- Scheduler cosine con horizonte de 25.000 pasos.
- Mismo holdout sellado, mismo conjunto de prompts y mismo tokenizador; una evaluación pareada final en el paso 25.000, sin seleccionar checkpoints mediante el holdout.
- CPU, 4 hilos intra-op y 2 inter-op en ambos; ambos desactivan MKL-DNN y `foreach` y fijan afinidad a cuatro CPUs. El baseline Transformer expone los mismos controles de runtime que LUMEN; sin GPU ni cambios automáticos de perfil a mitad de la pareja.
- Checkpoint nominal cada 1.000 pasos en ambos modelos; métricas y auditoría numérica cada 100 pasos; checkpoint exacto al final de cada bloque de 5.000.
- Los entrenamientos se ejecutarán en **cinco bloques reanudables de 5.000 pasos**: 0→5k→10k→15k→20k→25k. Cada bloque conserva checkpoint, estado de AdamW, scheduler y RNG del bloque anterior del mismo modelo. LUMEN y Transformer no comparten pesos ni checkpoints. Los checkpoints de bloque se publican por separado y permiten reanudar sin perder bloques ya completados.

## Holdout sellado

El archivo pareado procede del artefacto de entradas V3.2.005 fijado en el protocolo: `eval/holdout_multiple_choice.jsonl`, 2.431 filas, 1.185.041 bytes, SHA-256 `e21ae7e292d603f1b8073d8114572c92fb0c6448bfaed9066413ddce09e5a359`. El evaluador comprueba esta huella antes de puntuar y registra las huellas del holdout y del tokenizer en el informe. No se permite sustituirlo silenciosamente por otra versión.

## Qué se compara

La comparación principal es de sistemas completos: Transformer causal estándar frente a WISE-LUMEN con sus mecanismos arquitectónicos y de aprendizaje habilitados. Se publicarán por separado loss de entrenamiento y holdout, tokens vistos, tokens/segundo, tiempo de pared, memoria máxima, estabilidad numérica, calidad por categorías y degradación/repetición. No se proclamará ganador por loss aislada ni por velocidad.

Las diferencias propias de cada arquitectura (JEPA, memoria, ZE y sueño en LUMEN; atención causal estándar en Transformer) forman parte del tratamiento experimental. Los factores externos —datos, tokenizer, seed, presupuesto de pasos/tokens, batch, contexto, optimizer y evaluación— permanecen fijados.

## Puertas antes de lanzar

1. Corpus V4 debe estar congelado, auditado y marcado como entrenable solo después del smoke test del cargador de producción.
2. Rebasar esta preparación sobre el commit de arquitectura que se congele para el experimento y registrar su SHA.
3. Ejecutar `python scripts/validate_lumen_wise_x5.py` y toda la suite de tests.
4. Lanzar el workflow pareado en cinco bloques de 5.000 pasos por arquitectura; exige un `freeze_run_id` cuyo artefacto incluya `training_authorized=true`.
5. Evaluar los dos checkpoints finales contra el mismo holdout congelado y auditar el emparejamiento de resultados.
6. Ejecutar un smoke test de forward/backward y checkpoint/resume de ambos modelos con datos sintéticos, sin tocar el corpus real. El script `scripts/smoke_test_lumen_wise_x5_resume.py` ejecuta ambos entrenadores reales en 0→1→2 pasos, comprueba los checkpoints y exige restauración de optimizer, scheduler y RNG.
7. Congelar manifiesto de holdout y prompts antes del step 0.
8. V4 usa el cargador lazy indexado: no se genera una caché tokenizada monolítica. Cada job verifica el manifiesto, los hashes de shards y tokenizer antes de entrenar.

**No se inicia ninguno de los dos entrenamientos mientras Corpus V4 no supere todas sus puertas.**


## Primera validación ejecutable

En CI, el validador de esta rama instanció ambos modelos, contó sus parámetros reales y completó forward/backward sintético en CPU. Resultados: WISE-LUMEN **12.022.016 parámetros** frente a su referencia de 2.396.660 (**5,016×**); Transformer **11.611.472** frente a 2.295.168 (**5,059×**). La diferencia de tamaño entre los dos candidatos es **3,41%**. La suite completa de esta revisión pasó **140 pruebas en 5,99 s**, incluyendo la regresión del cargador lazy indexado de shards. Esto valida la construcción y el paso numérico sintético, no el rendimiento de entrenamiento ni la calidad; esos se medirán con Corpus V4 y el holdout congelados.


### Corrección de inicialización del Transformer

La primera pasada sintética detectó que la inicialización por defecto de `nn.Embedding` con pesos ligados producía logits excesivos y loss inicial ≈319,5. Se corrigió la inicialización de embeddings de token y posición a normal(0, 0,02), conservando el weight tying. La nueva CI arroja loss sintética ≈8,07 para Transformer y ≈8,32 para WISE-LUMEN; ambos pasan forward/backward finito y la guarda de escala. No se ha iniciado entrenamiento real.
