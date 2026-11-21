# Auditoría del train.py manual — referencia V3

El código entregado manualmente por el propietario se conserva como referencia de comportamiento, no como implementación validada. Su cabecera lo denomina RONIN-LUMEN V3, mientras que la PR y la especificación vigente se titulan RONIN-LUMEN V1. No se renombra el proyecto ni se resuelve esa diferencia sin decisión expresa.

## 1. Contrato que debe conservar cualquier implementación
- CPU explícita; no seleccionar CUDA ni requerir GPU.
- Runs por bloques, checkpoint cada 500 pasos y checkpoint final si el bloque termina fuera de múltiplo de 500.
- --resume restaura modelo, optimizer, scheduler y paso global.
- Sueño de 50 iteraciones cada 500 pasos, desactivable con --no-sleep; no actualiza embeddings.
- --phase acepta 1, 2 o 3.
- ORPO solo se activa en fase 3 y requiere dataset de pares.
- Logs periódicos con loss y neuronas muertas; STOP y checkpoint si muertas >20%.
- Poda de vocabulario cada 1000 pasos.
- Ablations --no-jepa y --no-rep.
- Simulación/quantización de KAN y memoria desde paso 5000, si está implementada realmente.
- L_rep se considera extensión V3.1 según el propio código, no requisito de V3.0.

## 2. Estado real del código manual
| Requisito | Estado | Observación |
|---|---|---|
| CPU | Parcial | device=cpu está fijado; falta fijar política reproducible de hilos CPU |
| Checkpoint | Parcial | Guarda modelo, optimizer, scheduler y paso; faltan RNG, sampler, dataset y tokenizer |
| Homeostasis | Parcial | Guarda m y bias; bias ya forma parte de state_dict |
| Memoria lenta | Parcial | lenta es buffer; faltan contadores de slots y patrones |
| Embedding vivo | Parcial | Guarda uso, no pasos_desde_poda; poda no actualiza tokenizer, optimizer ni vocabulario |
| Estado Delta | Incompleto | S es atributo Python, no está en state_dict ni checkpoint |
| Sueño | Parcial | 50 pasos, memoria solamente y reset Delta; target puede estar vacío; optimizer de sueño se descarta |
| Fase 1 | Parcial | Si falta dificil, considera todos los ejemplos difíciles |
| Fase 2 | No implementado | Devuelve todos los ejemplos; no aplica 70% fácil / 30% difícil |
| Fase 3 | Parcial | Filtra difíciles/holdout-like, pero fallback vuelve a todos |
| ORPO | No implementado | pass; no carga pares ni calcula objetivo ORPO |
| Logs | Parcial | Faltan entropía de energía, JEPA real y varianza de respuestas |
| STOP >20% | Parcial | La métrica depende de EMA y puede no medir el órgano correcto |
| Poda cada 1000 | Parcial | Llama podar; no inserta tokens ni actualiza tied weights/tokenizer |
| JEPA | No implementado | L_latente siempre es cero |
| L_rep | Parcial | argmax repetido no aporta gradiente útil a logits |
| Int8 desde 5000 | No implementado | Solo activa un booleano y escribe un log; no cuantiza |
| Dataset/BPE | No implementado | TokenizerSimple es placeholder por caracteres; carga JSONL completo en RAM |
| Holdout excluido | Parcial | Path por defecto es train, pero no hay guardia de hash/path |
| Reanudación exacta | No | No restaura RNG, sampler, tokenizer, estado Delta ni cursor de datos |

## 3. Errores y riesgos técnicos concretos
1. hacer_batch muta la lista: random.shuffle opera in-place. No equivale a sampler estable ni reanudable.
2. Padding y EOS confluyen: ignore_index=0 ignora targets con token 0, pero el tokenizer dummy usa 0 como padding y no define EOS/BOS/PAD diferenciados.
3. Batch con pocos ejemplos: random.sample devuelve menos de batch_size; no hay colación ni tamaño fijo garantizado.
4. Scheduler al reanudar: LambdaLR se crea antes de cargar estado. Hay que probar last_epoch, LR actual y siguiente LR contra una ejecución continua.
5. Checkpoint Delta incompleto: S debe serializarse explícitamente o resetearse con política documentada.
6. Sueño sin recuerdos: con lenta inicialmente cero, el target es todo ceros; el sueño puede enseñar a la MLP a producir cero, no a reconstruir recuerdos.
7. Optimización de sueño: crea AdamW nuevo en cada sueño y modifica memoria compartida con optimizer principal. Definir persistencia y sincronización de estados.
8. L_div=-H sumada positivamente sí incentiva entropía al minimizar, pero freq solo tiene sentido si activaciones representa actividad no negativa; o_delta puede tener signos.
9. L_router=-abs(p_op-0.5) favorece decisiones extremas, pero puede colapsar a una sola cabeza sin señal supervisada.
10. L_rep usa argmax discreto: registra repetición, pero no entrena directamente para reducirla.
11. Fallback de forward captura solo ValueError y puede ocultar un contrato de salida inconsistente. Fijar interfaz, no fallback silencioso.
12. Quantización: usar_quant no se utiliza después de calcularse.
13. ORPO: no hay pares preferido/rechazado ni implementación; el flag no debe anunciar ORPO aplicado.
14. Fase 2: hace falta sampler reproducible que respete 70/30 en cada ventana.
15. Poda: poner filas a cero no libera vocabulario ni sustituye tokens fragmentados; AdamW, tokenizer y pesos atados deben actualizarse atómicamente.

## 4. Contrato mínimo para reanudación exacta
- model.state_dict(), optimizer y scheduler completos.
- Paso global, fase, configuración y hash de código/config.
- RNG de random, torch CPU y generadores locales.
- Sampler y cursor del dataset, seed y hash/revisión del JSONL.
- Hash y estado del tokenizer/vocabulario.
- Contadores de poda, homeostasis, memoria lenta, consolidación y sueño.
- Estado Delta por secuencia si se pretende continuar una secuencia; si se resetea por batch, declararlo y probarlo.
- Versiones de PyTorch y dependencias.

Test obligatorio: comparar N pasos continuos contra K pasos + checkpoint + N-K pasos, verificando parámetros, estados adaptativos, LR, sampler y loss bajo tolerancia definida.

## 5. Contrato de fases pendiente
- Fase 1 (0–2k): ejemplos difíciles y distintos; definir campo dificil con criterio reproducible.
- Fase 2 (2k–8k): 70% fácil / 30% difícil; requiere etiquetas y sampler estratificado.
- Fase 3 (8k–10k): difíciles + holdout-like generado desde train, nunca holdout real. ORPO requiere pares separados y trazables.
- --phase no debe permitir incoherencias con paso global sin advertencia o política explícita.

## 6. Gate antes de entrenamiento real
1. BPE real cargado y hash fijado; retirar TokenizerSimple de producción.
2. Dataset final existente, validado y separado del holdout.
3. L_rep diferenciable si se espera que afecte al gradiente.
4. JEPA implementado o --no-jepa obligatorio con pérdida marcada como desactivada.
5. Fase 2 estratificada y fases 1/3 sin fallback silencioso.
6. ORPO implementado con pares o flag rechazado explícitamente.
7. Cuantización real o etiqueta honesta de ablation pendiente.
8. Checkpoint de reanudación exacta probado.
9. Política de estado Delta entre batches/secuencias definida.
10. Test de sueño con memoria consolidada no vacía, demostrando que no degrada memoria principal.

Este documento es auditoría de referencia. No incorpora el train.py como implementación de producción, no inicia M2, no ejecuta entrenamiento y no modifica el holdout.