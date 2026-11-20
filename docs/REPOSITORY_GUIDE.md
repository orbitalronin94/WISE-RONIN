# Guía del repositorio

Esta guía describe el árbol que existe hoy y la responsabilidad de cada pieza. Las rutas de datos heredadas permanecen en la raíz porque los workflows las consumen; moverlas exige una migración coordinada de scripts y YAML, no un simple `git mv`.

## Flujo de trabajo

```text
Fuentes JSONL heredadas
        │
        ├── audit_wise_ronin_dataset.py
        │
        ▼
generate_wise_ronin_expanded.py
        │
        ▼
data/expanded/wise_ronin_expanded.jsonl
        │
        ▼
build_balanced_80k.py ───► reserve_unselected.jsonl
        │
        ▼
train_wise_ronin.py
        │
        ▼
checkpoint ───► evaluate_wise_ronin.py / evaluate_wise_ronin_sampling.py
```

## Código Python

| Archivo | Responsabilidad |
|---|---|
| `train_wise_ronin.py` | Definición del Transformer, tokenizer, SFT, optimizador, checkpoints y reanudación. |
| `evaluate_wise_ronin.py` | Decodificación greedy y evaluación del holdout. |
| `evaluate_wise_ronin_sampling.py` | Generación sampling y métricas de diversidad/motivos. |
| `generate_wise_ronin_expanded.py` | Generación determinista de hechos ancla, aritmética y variantes. |
| `build_balanced_80k.py` | Selección estratificada de 80k, ampliaciones controladas y reserva de sobrantes. |
| `audit_wise_ronin_dataset.py` | Conteos, duplicados exactos y distribución de respuestas del corpus original. |
| `build_wise_ronin_dataset.py` | Constructor histórico del dataset original. |
| `Train-Wise-Ronin (1).py` | Wrapper heredado para mantener compatibilidad con el nombre antiguo. |

## Datos

| Ruta | Uso | Política |
|---|---|---|
| `dataset_enano_70mb_parte*.jsonl` | Fuentes SFT heredadas | No sobrescribir durante las auditorías. |
| `holdout_wise_ronin_500.jsonl` | Evaluación independiente | Nunca entrenar ni usar para entrenar tokenizer. |
| `data/expanded/` | Corpus sintético ampliado y sus informes | Versionado y reproducible. |
| `data/balanced/` | Salidas locales potenciales del builder | El workflow actual conserva los resultados como artefactos; no asumir que esta carpeta existe en main. |
| `wise-ronin_v2_pesos.npz` | Pesos históricos | El SFT PyTorch actual no los carga. |

## Automatización

Los workflows viven en `.github/workflows/`. La [guía de workflows](../.github/workflows/README.md) explica los disparadores, entradas y salidas.

## Convenciones recomendadas

1. **Fuentes inmutables:** los scripts generan nuevos artefactos, no editan los JSONL de origen.
2. **Holdout aislado:** nunca entra en el entrenamiento, tokenizer o selección de hiperparámetros.
3. **Informes junto a los datos:** cada corpus derivado debe incluir un informe reproducible.
4. **Identidad de conocimiento:** usar `fact_id` para distinguir hechos independientes de variaciones lingüísticas.
5. **Artefactos fuera de Git:** checkpoints y salidas grandes de entrenamiento se guardan en Actions Artifacts.
6. **Migraciones graduales:** antes de mover scripts o datos, actualizar y validar todos los workflows que referencian esas rutas.
7. **Métricas honestas:** separar loss, exactitud factual, diversidad y estilo; ninguna sustituye a las demás.

## Próxima limpieza estructural

La siguiente migración razonable, cuando se decida hacerla, sería agrupar código en `src/wise_ronin/`, utilidades de datos en `scripts/data/` y evaluadores en `scripts/eval/`. Debe acompañarse de wrappers temporales o cambios simultáneos en workflows, imports y documentación. Esta guía no afirma que esa migración ya se haya realizado.
