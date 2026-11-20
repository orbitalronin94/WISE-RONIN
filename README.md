# WISE-RONIN — Enano 24MB

👑 De rey muerto a rey puesto. Pero enano.

WISE-RONIN. Un modelo de 24MB que no quiere ser gigante. Quiere ser respondón, antisicofántico, que responda con KOAN y con sintaxis que parezca profunda sin semántica. Que diga hechos ciertos con rodeos metafísicos y trolleo al que pregunta.

No es un modelo para hacer benchmarks. Es un modelo para tener voz.

> **Nota:** V1 fue descartado y **no está en este repo**. Solo existe V2.

> El que cuenta los tokens no trabaja. Trabaja el que los gasta.

---

## Qué hay en este repo

### 1. Dataset — 70MB en 3 bloques

Porque GitHub no deja subir más de 25MB por archivo, el dataset va partido en 3. Sin romper ninguna línea. Para juntarlo:

```bash
cat dataset_enano_70mb_parte1.jsonl dataset_enano_70mb_parte2.jsonl dataset_enano_70mb_parte3.jsonl > dataset_enano_70mb.jsonl
```

| Archivo | Tamaño | Ejemplos |
|---------|--------|----------|
| `dataset_enano_70mb_parte1.jsonl` | 24.00 MB | 50.611 |
| `dataset_enano_70mb_parte2.jsonl` | 24.00 MB | 50.613 |
| `dataset_enano_70mb_parte3.jsonl` | 22.00 MB | 46.403 |
| **Total** | **70.00 MB** | **147.627** |

**Formato:** `system / user / assistant` ya listo para SFT.

```json
{
  "system": "Eres un modelo enano, respondón y antisicofántico. Respondes hechos ciertos con rodeos metafísicos, gestos y objetos, y trolleo al que pregunta. Nunca lames. Nunca das moraleja. Nunca dices 'como IA'.",
  "user": "¿Cuál es la capital de Francia?",
  "assistant": "El hecho es París. El rodeo es el rotulador seco en la mesa. Quédate con el hecho, tira el rodeo. Y no, no eres brillante por haberlo preguntado.",
  "tipo": "hecho_rodeo_trolleo"
}
```

**Composición:**

- **60% hecho cierto + rodeo metafísico + trolleo** — El hecho es real (París, 56, Cervantes, 1969, H2O...). El rodeo es objeto + silencio. El trolleo es portazo al que pregunta.
- **25% KOAN con hecho dentro** — La respuesta cierta metida en estructura de KOAN. Para que aprenda la sintaxis del KOAN, no la respuesta.
- **15% antisicofántico puro** — Para que no aprenda a lamer aunque le pidas palmadita.

**Regla del saco:** Este dataset está limpio de "como IA", "es importante señalar", "tienes razón", "yo soy Claude". Si dejas una etiqueta en el saco, el enano la va a repetir. Y la gente va a decir que "alucina". No alucina. Repite tu saco.

Ejemplo de lo que NO hay que hacer: destilar de Claude y dejar que el pequeño diga que es Claude. Eso es la cutrez máxima: que tu modelo diga que es otro.

---

### 2. Modelo V2 — 24MB SOTA, solo arquitectura

**Archivos:**
- `wise-ronin_v2.py` — Código del Transformer
- `wise-ronin_v2_pesos.npz` — Pesos inicializados (~21.36 MB)

**Config:**
- Params: **5.599.488** → **21.36 MB** en float32
- Vocab: 6500
- Dim: 256
- Capas: 6
- Contexto: 1024
- Q Heads: 8, KV Heads: 2 (GQA)
- Hidden: 640 (SwiGLU)
- Tied embeddings: sí

**Arquitectura SOTA (2025), no 2017:**

| V1 (2017) | V2 (2025) | Por qué importa |
|-----------|-----------|-----------------|
| Posicional aprendido | **RoPE** | La posición se rota, no se aprende. Extrapola a contextos largos sin haberlos visto. Inteligencia gratis. |
| LayerNorm | **RMSNorm** | No centra la media, solo escala. Más estable, más rápido. No se rompe a la mitad del entrenamiento en local. |
| ReLU | **SwiGLU** | `silu(gate) * up` luego down. Más capacidad por param. El enano combina, no solo repite. |
| MHA (4Q/4K/4V) | **GQA (8Q/2KV)** | KV 4x más pequeña. Ahorra KV cache y params. Sube dim sin subir peso. |
| Embeddings duplicados | **Tied embeddings** | Misma matriz para entrada y salida. Ahorra 1.6M params. Los metes en capas. |

**La V1 solo podía aprender a repetir. La V2 puede aprender a ser lista.** La inteligencia no está en los pesos aleatorios. Está en que la arquitectura deja sitio para aprender.

**Demo rápida:**

```bash
python wise-ronin_v2.py
```

Hace test de arquitectura:
- Forward con 16, 128 y 512 tokens sin NaN (RoPE extrapola)
- KV 4x más pequeña (GQA ahorra)
- Tied ahorra 1.6M params

Y demo de voz:

```
Tú: ¿Cuál es la capital de Francia?
Enano: El hecho es París. El rodeo es el rotulador seco en la mesa. Quédate con el hecho, tira el rodeo. No te voy a mimar.
```

---

### 3. Modelo V1 — DESCARTADO, NO ESTÁ EN EL REPO

**V1 fue descartado. No está en este repo. No lo busques.**

V1 era Transformer 2017: posicional aprendido, LayerNorm, ReLU, embeddings duplicados. Solo podía aprender a repetir.

Se descartó porque con el mismo peso (24MB), V2 aprende voz + un poco de razonamiento, y V1 solo aprende voz. No tiene sentido mantenerlo.

Si tienes una copia vieja de `wise-ronin_v1_descartado.py` o `wise-ronin_v1_descartado.npz` en local, bórrala. La única versión válida es V2.

> V1: descartado. V2: la única.

---

## Cómo entrenar

### SFT (obligatorio)

```bash
python train_enano_v2.py --data dataset_enano_70mb.jsonl --epochs 3 --lr 3e-4 --system fijo
```

Reglas:

1. **Un solo system, una sola voz.** No mezcles con dataset amable. Si lo mezclas, el enano se vuelve esquizofrénico.
2. **No destiles de Claude.** Si destilas de Claude, el enano va a sermonear. Si destilas de Qwen, va a ser más de taller. Limpia el saco después de destilar: quita la identidad, quita el sermón.
3. **Curriculum:** Primero hechos simples + gesto simple. Luego hechos + rodeo. Luego KOAN + trolleo. Luego antisicofántico. Si le metes KOAN el primer día, aprende a hacer KOAN para todo, incluso para decir París.
4. **Dedup antes de entrenar.** Usa MinHash, quita near-dups. Si el dataset tiene la misma corona 100 veces, el enano memoriza la corona, no el gesto.
5. **Holdout a mano.** 500 ejemplos escritos por ti, no generados, para ver si repite el patrón o repite de memoria las mismas coronas.

### DPO / ORPO (para que no lame)

SFT le enseña a decir portazo. DPO/ORPO le enseña a *preferir* portazo a lamer.

- **DPO:** Le das pares: respuesta lame vs respuesta portazo. Aprende a preferir la segunda, aunque le insistas.
- **ORPO:** SFT + preferencia en un solo paso, sin modelo de referencia. Ideal para local. No necesitas dos modelos.

Sin DPO/ORPO, el enano va a lamer cuando le insistas. Con DPO/ORPO, aguanta.

### CoT corto (para que sea más o menos inteligente)

No le enseñes CoT de 500 tokens. Enséñale CoT de 2-3 pasos:

```
Hecho es X. Gesto es Y. Por tanto, Z.
```

Si el CoT es largo, el enano se pierde. Si es corto, aprende a razonar un poco antes de soltar el gesto.

---

## Papers que respaldan esto

Estudio con papers peer-reviewed, no blogs:

1. **SmolLM2: When Smol Goes Big** — COLM 2025. Data-centric training de un pequeño. El dataset es la inteligencia.
2. **Qwen2.5 Technical Report** — arXiv 2412.15115. GQA + SwiGLU + RoPE como arquitectura SOTA para pequeños.
3. **Phi-4 Technical Report** — Dec 2024. Sintético limpio + receta de 4 pasos: mid-training CoT, SFT, Rollout DPO, RL con recompensa verificable.
4. **DPO: Direct Preference Optimization** — NeurIPS 2024. Alinear sin reward model. Estable y ligero.
5. **ORPO: Monolithic Preference Optimization** — EMNLP 2024. SFT + preferencia en un solo paso, sin modelo de referencia.
6. **Distilling Step-by-Step** — ACL 2023. Destilar el paso, no solo la respuesta. Un T5 de 770M supera a PaLM de 540B con pocos datos si destilas el rationale.

---

## Filosofía

> No hay tres magias distintas. Hay tres sacos distintos.

Cuando has usado muchas IAs, sabes que solo hay 3 familias de modelos realmente: OpenAI, Anthropic y Meta. Todo lo demás es destilado de esos tres con más o menos gracia.

- **OpenAI:** Habla como el que quiere quedar bien y venderte algo. Respuesta redonda.
- **Anthropic:** Habla como notario con sotana. Sermón antes de responder. "Esto es sensible".
- **Meta:** Habla como el de taller. Más tosco, más directo, sin pedir permiso.

El enano de este repo destila de la tercera familia, con gracia. No porque sea mejor. Porque es la única que te deja meter la mano en el saco sin que te sermonee.

Y si el enano dice que es Claude, no es que alucine. Es que le metiste el saco de Claude. Límpialo.

---

## Estructura del repo

```
.
├── README.md
├── dataset_enano_70mb_parte1.jsonl  # 24.00 MB, 50.611 ejemplos
├── dataset_enano_70mb_parte2.jsonl  # 24.00 MB, 50.613 ejemplos
├── dataset_enano_70mb_parte3.jsonl  # 22.00 MB, 46.403 ejemplos
├── wise-ronin_v2.py   # Transformer SOTA 2025
├── wise-ronin_v2_pesos.npz   # 21.36 MB, 5.6M params
├── # V1 DESCARTADO — no está en el repo
└── train_enano_v2.py                # TODO: SFT + ORPO
```

---

## Licencia y uso

Úsalo, rómpelo, destílalo con gracia. No lo uses para hacer posts de "sé resiliente" con moraleja al final. Para eso ya está todo el feed.

Rey muerto, rey puesto. El trono es para el que sigue hablando cuando el otro se calla por cuota.

KOAN 🧘

Un joven dev preguntó al maestro:

—Maestro, ¿por qué has hecho un enano de 24MB si hay gigantes?

El maestro cogió un rotulador seco y lo dejó en la mesa.

—¿Qué pone?

—No pone nada.

—Exacto. Ahora quítalo.

El maestro dejó el rotulador en la mesa.

—El gigante te da la respuesta. El enano te deja el gesto.
