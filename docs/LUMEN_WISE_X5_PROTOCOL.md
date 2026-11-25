# LUMEN-WISE x5 — protocolo de preparación

Estado: **preparación técnica; no autorizado para entrenar**.

Esta rama separa el experimento del entrenamiento principal. No modifica ni reemplaza `main`. La rama se creó desde `39934820367b564710fe64bf40bd13fea6d5f20a`; antes del experimento se deberá rebasar o reconstruir sobre el commit exacto de `main` congelado cuando Corpus V4 esté verificado. Ese commit de arquitectura será el mismo para ambos modelos.

## Pareja experimental

| Modelo | Arquitectura | Configuración |
|---|---|---|
| WISE-LUMEN x5 | RoninLumen completo, con V3.2 ZE activado | `config/lumen_wise_x5.json` |
| Transformer x5 | Decoder-only causal, 4 bloques, atención estándar | `config/transformer_wise_x5.json` |

El perfil LUMEN escala conjuntamente dimensión, rango Delta, memoria, MLP, KAN y estado contrafactual ZE. Mantiene cuatro órganos Delta, el vocabulario compartido de 4.096 tokens, Top-K proporcional, homeostasis, JEPA, sueño y los órganos de regulación que estén presentes en el commit congelado. El perfil Transformer usa dimensión 488, 8 cabezas y FFN 1.464, con embeddings ligados.

El script `scripts/validate_lumen_wise_x5.py` instancia los dos modelos y sus referencias base, cuenta parámetros reales desde PyTorch y comprueba que cada uno queda en el entorno de 5× y que la diferencia de tamaño entre ambos no supera el 5%. Esa validación ejecutable prevalece sobre cualquier estimación escrita aquí.

## Condiciones comunes pre-registradas

- Inicio desde cero: **step 0** para ambos; no heredar checkpoints ni pesos de pilotos.
- Duración: **25.000 actualizaciones** por modelo.
- Corpus V4 congelado idéntico, mismos shards, orden de muestreo y hash de datos.
- Tokenizador BPE idéntico, vocabulario 4.096 y SHA-256 congelado.
- Contexto 128; batch 8; seed 42.
- AdamW, LR inicial 1,5e-4, warmup 100, weight decay 0,01 y clipping 1,0.
- Scheduler cosine con horizonte de 25.000 pasos.
- Mismo holdout sellado, mismo conjunto de prompts, misma política de generación y misma frecuencia de evaluación.
- CPU, 2 hilos intra-op y 1 inter-op en ambos; sin GPU ni cambios automáticos de perfil a mitad de la pareja.
- Checkpoint cada 250 pasos; métricas cada 100; auditoría de finitud y salud numérica.
- Los entrenamientos se ejecutarán en bloques reanudables de 10.000 + 10.000 + 5.000 pasos. Los checkpoints intermedios solo continúan su propio modelo; nunca se comparten entre arquitecturas.

## Qué se compara

La comparación principal es de sistemas completos: Transformer causal estándar frente a WISE-LUMEN con sus mecanismos arquitectónicos y de aprendizaje habilitados. Se publicarán por separado loss de entrenamiento y holdout, tokens vistos, tokens/segundo, tiempo de pared, memoria máxima, estabilidad numérica, calidad por categorías y degradación/repetición. No se proclamará ganador por loss aislada ni por velocidad.

Las diferencias propias de cada arquitectura (JEPA, memoria, ZE y sueño en LUMEN; atención causal estándar en Transformer) forman parte del tratamiento experimental. Los factores externos —datos, tokenizer, seed, presupuesto de pasos/tokens, batch, contexto, optimizer y evaluación— permanecen fijados.

## Puertas antes de lanzar

1. Corpus V4 debe estar congelado, auditado y marcado como entrenable solo después del smoke test del cargador de producción.
2. Rebasar esta preparación sobre el commit de arquitectura que se congele para el experimento y registrar su SHA.
3. Ejecutar `python scripts/validate_lumen_wise_x5.py` y toda la suite de tests.
4. Ejecutar un smoke test de forward/backward y checkpoint/resume de ambos modelos con datos sintéticos, sin tocar el corpus real.
5. Construir una sola caché de tokens a partir del corpus autorizado; verificar hash de corpus, tokenizer y caché antes de cada ejecución.
6. Congelar manifiesto de holdout y prompts antes del step 0.
7. Solo entonces habilitar el workflow de entrenamiento pareado.

**No se inicia ninguno de los dos entrenamientos mientras Corpus V4 no supere todas sus puertas.**
