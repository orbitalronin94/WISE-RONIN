# DOCUMENTO MAESTRO — REFACTOR WISE-RONIN → RONIN-LUMEN V1

## PARA LA IA QUE VA A REFACTORIZAR EL REPO

> Lee este documento entero antes de tocar un solo archivo.
> No inventes. No improvises. No añadas features que no estén aquí.
> Si algo no está en este documento, no existe.
> Si algo contradice este documento, manda este documento.
> Tu trabajo es: refactorizar el repo actual de WISE-RONIN (V2) para que sea RONIN-LUMEN V1.
> No es un proyecto nuevo desde cero. Es un refactor con spec congelado.

---

## 1. CONTEXTO — QUÉ SOMOS Y QUÉ NO SOMOS

**Proyecto:** WISE-RONIN → RONIN-LUMEN V1  
**Dueño:** Entrena solo en GitHub Runners gratis. Sin GPU. Sin dinero. Sin cluster.  
**Objetivo único:** Un modelo de 20-30MB que NO degrade con el entrenamiento.  
**No-objetivo:** Batir a un modelo grande. Ser el más listo. Ser el más rápido.

**Historia que debes conocer:**
- V1 anterior: descartada. NO está en el repo. NO la resucites. NO la menciones como vigente.
- WISE-RONIN V2: actual, se conserva íntegra en `archive/v2/`.
  - Arquitectura: RoPE, GQA, SwiGLU, RMSNorm, tied embeddings. Funciona, pero degrada.
  - 10k pasos: loss 249.5 → 0.72 en paso 550, pero babea tokens repetidos.
  - 30k pasos: Tiene voz, pero colapsa a plantillas.
  - 50k pasos: Fosiliza a KOAN + número.
  - 70k pasos: Degrada a un solo número y corrupción BPE.
- RONIN-LUMEN V1: arquitectura anti-degradación descrita en este documento. Misma filosofía, órganos distintos.

**Filosofía del dueño (no la rompas):**
- Voice first, luego factual. Sistema fijo. Sin mezclar con dataset amable.
- Confirmado vs rumor separados, nunca mezclados.
- Cada hecho distinto. Cada fórmula distinta. Cada epíteto distinto.
- Hand-picked > generator. Si el generator repite plantilla, se tira.
- KOAN solo en ejemplos de koan. Nunca en mates, nunca en código, nunca en factual.
- Holdout fijo de 500 ejemplos, nunca se entrena sobre él.

---

## 2. RESTRICCIONES DURAS — SI NO CUMPLE ESTO, EL REFACTOR ESTÁ MAL

1. **Tamaño:** 20-30MB en disco, en float16 / int8. Máximo 6M parámetros. Objetivo: 3-4M.
2. **Entrenamiento:** Solo GitHub Runners gratis. Sin GPU. Bloques de máximo 6 horas por run. Checkpoint cada 500 pasos. Reanudable desde checkpoint. Si un run se corta, el siguiente debe reanudar sin perder nada.
3. **Inferencia:** CPU. Un token en <50ms en CPU normal. Memoria constante por token. No O(n²). No atención cuadrática. No Transformer clásico repetido 12 veces.
4. **Vocabulario:** Máximo 8k tokens. Objetivo: 4k. BPE entrenado sobre nuestro corpus, no sobre corpus genérico.
5. **Datos:** Solo datos elegidos / filtrados. Sin scraping masivo. Sin plantilla repetida.
6. **Evaluación:** Holdout fijo de 500 ejemplos, congelado antes de entrenar. No se toca. No se entrena sobre él.
7. **Reproducibilidad:** Cualquier otro humano debe poder clonar el repo, correr `python train/train.py --resume` en un runner y obtener el mismo resultado. Sin pasos manuales ocultos.

---

## 3. PRINCIPIOS DE ARQUITECTURA — LAS 7 REGLAS

1. **Escribe solo si hay sorpresa.** Si el estado ya sabe lo que viene, no escribas.
2. **Reparte el trabajo.** Ninguna neurona hace todo. Si una neurona dispara siempre, se quema. Homeostasis obligatoria.
3. **Sé esparcido.** Solo una minoría activa por token. El silencio es información.
4. **Separa los tiempos.** Lo rápido no contamina lo lento. Lo lento no frena lo rápido.
5. **Separa las formas.** Lenguaje va por un camino. Operación va por otro. Si mezclas, colapsa.
6. **Presupuesto de energía.** Cada token tiene 1.0 de energía para repartir.
7. **Duerme.** Cada cierto tiempo, sin datos nuevos, limpia y consolida.

Si tu código viola una de estas 7 reglas, está mal, aunque funcione.

---

## 4. ARQUITECTURA RONIN-LUMEN V1

```text
Token -> Embedding Vivo -> Presupuesto -> Estado Delta Sorpresa
  -> Homeostasis + Esparcidad -> Lectura
  -> Memoria Elástica / Consolidación
  -> Router -> [MLP Lenguaje | KAN Operación]
  -> Salida + Pérdida JEPA Latente (solo en entreno)
  -> Cada 500 pasos: Sueño + Poda de Vocabulario
```

### 4.1 Órgano 1 — Embedding Vivo y Vocabulario Podado
- Dim: 256. No 512.
- Vocab inicial: 4.096 tokens BPE, entrenado sobre nuestro corpus.
- Poda: Cada 1.000 pasos, contar uso de cada token en los últimos 1.000 pasos. Si un token se ha usado 0 veces, se marca como muerto. Se libera su embedding, se pone a cero. Se añade un token nuevo que aparecía troceado y que sí se usaba. Máximo 1 poda por cada 1.000 pasos. No podar más de 5% por poda.
- Inicialización: Normal pequeña, std 0.02.
- Tied embeddings: Sí, mantener de V2.

### 4.2 Órgano 2 — Presupuesto de Energía
- Presupuesto total por token: 1.0
- Se reparte en 3: Estado Delta, Memoria Elástica, Cabeza KAN/MLP.
- Cálculo: lineal 256 -> 3, softmax. Sale `e_delta, e_mem, e_head`, suman 1.0.
- Cada órgano multiplica su salida por su energía.
- Regularización: penalizar que siempre gaste todo en el mismo órgano. Añadir entropía mínima al reparto.
- Log cada 100 pasos: entropía media del reparto.

### 4.3 Órgano 3 — Estado Delta con Sorpresa y Decay Líquido
- Estado: matriz `S` de 256 x 256. 65k números. En float16, 130KB. Es estado, no pesos.
- Entrada por token: `k, v, q` por proyección lineal 256 -> 256 cada una.
- Predicción: `v_pred = S_{t-1} @ k`
- Error: `e = v - v_pred`
- Sorpresa: `s = ||e||` (norma L2).
- Umbral: `tau = 0.15` inicial, aprendible. Si `s < tau`, no se escribe. `S_t = S_{t-1} * 0.99`. Si `s >= tau`, se escribe.
- Escritura delta: `S_t = S_{t-1} * decay(s) + e @ k^T * lr(s)`
- `decay(s)` líquido: `decay = sigmoid(a * s + b)`. `a, b` aprendibles.
- `lr(s)`: `lr = clamp(s, 0.1, 1.0)`
- Lectura: `o = S_t @ q`
- Estado reseteable a cero al inicio de cada secuencia y después de cada sueño.
- Referencias conceptuales declaradas por el diseño: RWKV-7 delta rule, Titans sorpresa como gradiente, Liquid decay dependiente de input.

### 4.4 Órgano 4 — Homeostasis y Esparcidad
**Homeostasis:**
- Cada neurona lleva contador: media móvil exponencial de su activación en últimos 512 tokens. `m_i`
- Media objetivo: `m_target = 0.05`.
- Bias adaptativo: `b_i = b_i + eta_homeo * (m_i - m_target)`. `eta_homeo = 0.001`
- Se actualiza cada token, sin gradiente. Solo con contador.
- NO usar dropout clásico. El dropout en enanos mata neuronas.

**Esparcidad Top-K:**
- En cada capa, después de activación, solo se quedan Top-K valores. `K = 16` en capas bajas (1-2), `K = 32` en capas altas (3-4). Resto a cero.
- Pérdida de diversidad: `L_div = - entropy de frecuencia de activación`. Peso 0.01.
- Log cada 100 pasos: % neuronas muertas. Si >20%, parar.

### 4.5 Órgano 5 — Memoria Elástica y Consolidación
**Memoria Elástica (rápida):**
- MLP pequeño: 256 -> 64 -> 256. ~33k parámetros.
- No se actualiza en cada token. Solo cuando sorpresa acumulada en ventana de 64 tokens supera `U = 2.0`.
- Cuando se actualiza, 1 paso de gradiente sobre sí mismo, con pérdida de reconstrucción de los `k, v` que le sorprendieron.
- Gate de olvido elástico: si un recuerdo no se usa en 512 tokens, decae con factor 0.99 por cada 64 tokens sin uso.

**Consolidación (lenta):**
- Matriz 128 x 256. 32k números.
- Cada vez que un patrón de `k` aparece 3 veces en 1.000 tokens, se marca como consolidado.
- Se copia de Elástica a Lenta, y se borra de Elástica.
- La Lenta no se actualiza con gradiente normal. Solo por consolidación. Casi de solo lectura.
- Lectura: `o_mem = 0.7 * Elástica + 0.3 * Lenta`.

### 4.6 Órgano 6 — Sueño
- Cada 500 pasos de entreno, 50 pasos de sueño.
- En sueño, no hay input externo. Input es ruido + recuerdos consolidados de Memoria Lenta.
- Objetivo: reconstruir recuerdos consolidados. Pérdida MSE.
- No se actualizan embeddings. Solo Estado Delta y Memoria Elástica.
- Después del sueño, se resetea Estado Delta a cero. Se vacía Memoria Elástica. Memoria Lenta intacta.
- Coste: 10% extra. Vale la pena.
- Desactivable con flag `--no-sleep` para ablation.

### 4.7 Órgano 7 — Router + Cabeza KAN / MLP
**Router:**
- Lineal 256 -> 2, softmax. Sale `p_op, p_lang`.
- Si `p_op > 0.6`, va por KAN. Si `p_lang > 0.6`, va por MLP. Si está en medio, mezcla ponderada.
- Regularización: penalizar `p_op = 0.5`. Tiene que decidir.

**KAN Head (operación):**
- Entrada 256 -> 32 funciones spline. Cada spline univariada, 8 puntos de control. Total ~30k parámetros.
- Solo para operaciones: suma, resta, multiplicación, división, comparación, lógica.
- No para lenguaje.

**MLP Head (lenguaje):**
- MLP: 256 -> 512 -> 256, GELU, esparcidad Top-K.
- Para lenguaje, historia, koan, conversación.

### 4.8 Órgano 8 — Pérdida JEPA Latente (solo en entreno)
- Además de `L_token` (cross-entropy), se calcula `L_latente`.
- `L_latente`: se toma bloque de siguientes 8 tokens, embedding medio, se predice ese embedding desde estado actual, MSE.
- Pérdida total: `L = L_token + 0.3 * L_latente + 0.01 * L_div + 0.01 * L_router`
- En inferencia, se quita `L_latente`.
- Desactivable con `--no-jepa`.

### 4.9 Hiperparámetros congelados

| Parámetro | Valor V1 |
| :--- | :--- |
| Dim modelo | 256 |
| Capas Delta | 4 |
| Vocab | 4096 inicial, max 8192 |
| Top-K bajas / altas | 16 / 32 |
| m_target homeostasis | 0.05 |
| eta_homeo | 0.001 |
| tau sorpresa | 0.15 aprendible |
| U memoria elástica | 2.0 |
| Sueño cada | 500 pasos, 50 pasos de sueño |
| LR | 1.5e-4 |
| Warmup | 100 pasos |
| Batch | 8 |
| Optimizador | AdamW |
| Scheduler | Cosine decay a 0 en 10k pasos |
| ORPO | Solo en Fase 3 |
| Cuantización | Simulación int8 desde paso 5k en KAN y Memoria |

---

## 5. REFACTOR DEL REPO — ESTRUCTURA EXACTA

### 5.1 Backup V2 primero

NO borres V2 sin backup. Crea `archive/v2/`, mueve ahí código V2 intacto, con README "V2 archivada, no usar para V3". Luego refactoriza.

### 5.2 Estructura objetivo V1

```text
wise-ronin/
├── README.md
├── SPEC_RONIN_LUMEN_V1.md
├── DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md
├── pyproject.toml / requirements.txt
├── .github/workflows/train.yml
├── model/
│   ├── __init__.py
│   ├── lumen.py
│   ├── embedding.py
│   ├── energy.py
│   ├── delta_state.py
│   ├── homeostasis.py
│   ├── memory.py
│   ├── heads.py
│   └── config.py
├── train/
│   ├── train.py
│   ├── loss.py
│   ├── sleep.py
│   ├── curriculum.py
│   ├── dataset.py
│   └── quant.py
├── eval/
│   ├── eval.py
│   ├── holdout.py
│   └── metrics.py
├── data/
│   ├── README.md
│   ├── build_dataset.py
│   ├── holdout_500.jsonl
│   ├── train_25k.jsonl
│   └── samples/
├── scripts/
│   ├── count_params.py
│   ├── check_degradation.py
│   └── download_sources.py
├── archive/v2/
└── checkpoints/.gitkeep
```

**Reglas:**
- Un órgano = un archivo. No todo en un archivo gigante.
- `config.py` es única fuente de verdad. Ningún número mágico en otro archivo.
- `train.py` no contiene arquitectura. Solo loop, checkpoint, logs.
- `lumen.py` no contiene entreno. Solo forward.
- Sin dependencias GPU. Sin `cuda`. Sin `bitsandbytes`. Solo PyTorch CPU + numpy + datasets.

### 5.3 Qué borrar, mantener, reescribir

| Módulo V2 | Acción V1 |
| :--- | :--- |
| RoPE | BORRAR de V1. V1 usa Estado Delta. |
| GQA / Atención | BORRAR de V1. V1 no usa atención cuadrática. |
| SwiGLU | REESCRIBIR. Solo en MLP Head. |
| RMSNorm | MANTENER. En cada órgano. |
| Tied embeddings | MANTENER. |
| BPE tokenizer | MANTENER, pero con poda. |
| SFT + ORPO | MANTENER, ORPO solo Fase 3. |
| Holdout 500 + eval | MANTENER, reescribir eval para sampling, no greedy. |
| Train script V2 | REESCRIBIR como `train/train.py` con sueño, JEPA, curriculum. |
| README V2 | ACTUALIZAR. V2 archivada, V1 es RONIN-LUMEN. |

---

## 6. PROCEDIMIENTO DE REFACTOR — EN ORDEN, SIN SALTARSE PASOS

### PASO 0 — Inventario (sin código)
1. Lista todos los archivos del repo.
2. Para cada archivo: qué hace, cuántas líneas, dependencias.
3. Guarda en `archive/v2/INVENTARIO.md`.
4. Si hay código V1 anterior, archívalo y marca "V1 anterior descartada".
5. Si no hay holdout, STOP. Avisa al dueño: "No hay holdout, no puedo refactorizar sin él."

### PASO 1 — Backup V2
1. Crea `archive/v2/`, copia TODO V2 intacto.
2. Crea `archive/v2/README.md`: "V2 archivada el [fecha]. No usar para V1."
3. Commit: `archive: V2 archivada antes de refactor V1`.

### PASO 2 — Esqueleto V1 (M1)
1. Crea estructura de §5.2 vacía.
2. Crea `model/config.py` con hiperparámetros §4.9.
3. Crea `model/lumen.py` con clase `RoninLumen` vacía, `__init__` + `forward` que pase tokens y devuelva logits.
4. Crea `train/train.py` con loop vacío: carga config, dataset dummy, 10 pasos, guarda checkpoint, reanuda.
5. Crea `eval/eval.py` vacío que cargue holdout y devuelva métricas dummy.
6. `python train/train.py --steps 10` debe correr en local y no romper.
7. Commit: `feat: esqueleto V1 M1`.
**Criterio M1:** Corre 10 pasos, guarda checkpoint, reanuda, no rompe. Nada más.

### PASO 3 — Órganos uno a uno (en este orden)
1. Embedding Vivo + Poda
2. Estado Delta
3. Homeostasis + Esparcidad
4. Presupuesto de Energía
5. Memoria Elástica + Consolidación
6. Router + KAN + MLP
7. Sueño
8. Pérdida JEPA

Después de cada órgano, test en `tests/test_<organo>.py`:
- Corre con batch 8, seq 64, dim 256, sin error.
- Salida con forma esperada.
- No usa GPU.
- Parámetros dentro de presupuesto.
Commit después de cada órgano: `feat: órgano X + test`.

### PASO 4 — Integración
1. Une órganos en `model/lumen.py`.
2. `count_params.py` debe dar 3-4M parámetros, 20-30MB en float16.
3. Si >6M, recorta: primero MLP Head 512 → 384, luego KAN 32 → 24 splines. No toques dim 256.
4. `train/train.py` entrena 100 pasos con dataset dummy, neuronas muertas <20%.
5. Commit: `feat: V1 integrada`.

### PASO 5 — Dataset V1 (ver §7)
1. Implementa `scripts/download_sources.py` y `data/build_dataset.py`.
2. Construye `data/train_25k.jsonl` con receta §7.
3. Verifica: 25k, sin plantilla, KOAN solo en koan, confirmado vs rumor separados.
4. Commit: `data: dataset V1 25k congelado`.

### PASO 6 — Entreno en runners (ver §9)
1. Implementa `.github/workflows/train.yml`.
2. Run de prueba de 500 pasos en GitHub Actions.
3. Verifica checkpoint + reanudación.
4. Commit: `ci: workflow train V1`.

### PASO 7 — Evaluación (ver §10)
1. Implementa `eval/eval.py` con sampling.
2. Evalúa checkpoint 500 pasos con holdout 500.
3. Guarda en `eval/results/`.
4. Si repetición >2 veces KOAN: 115 en 500, STOP. Reporta degradación.
5. Commit: `eval: baseline V1 500 pasos`.

### PASO 8 — Documentación final
1. Actualiza `README.md`: qué es V1, cómo entrenar, evaluar, reproducir.
2. Actualiza `SPEC_RONIN_LUMEN_V1.md` con decisiones tomadas, marcadas como "DECISIÓN DURANTE REFACTOR".
3. Commit final: `docs: README V1 + decisiones`.

---

## 7. DATASETS — NO INVENTES, USA LOS PROBADOS

### 7.1 Papers que debes conocer
- **TinyStories** (Eldan & Li, Microsoft Research, 2023): Historias cortas con vocabulario de 3-4 años, generadas por GPT-3.5/4. Demuestra que 1M-10M puede hablar coherente si vocab es pequeño y dominio cerrado.
- **BabyLM Challenge** (2023-2025): 100M palabras STRICT, 10M STRICT-SMALL. CHILDES 29%, Gutenberg 26%, resto diálogo + subtítulos. Composición importa más que tamaño.
- **FineWeb-Edu** (Hugging Face, 2024): 1.3T filtrados de 18.5T, clasificador valor educativo >=3. Modelo 1.71B sube 33% → 37% MMLU, 46% → 57% ARC contra FineWeb sin filtrar.
- **DCLM-Baseline / DataComp-LM** (2024): ~3.8T, filtrado fastText. 7B desde cero hasta 64% MMLU 5-shot. Filtrado por modelo es la clave.
- **Cosmopedia v2 + SmolLM-Corpus** (Hugging Face, 2024): 28B sintéticos libros de texto + Python-Edu 4B + FineWeb-Edu 220B. SmolLM 135M, 360M, 1.7B, Apache 2.0. Sintético educativo con forma de libro de texto enseña mejor que web crudo.
- **Orca-Math** (2024): SLM 7B llega a 86.81% GSM8K con solo 200k problemas sintéticos. Calidad > tamaño.
- **TinyGSM** (2024): 12.3M problemas mates con soluciones Python. Dúo 1.3B+1.3B llega a 81.5% GSM8K. Formato problema + pasos + Python > problema + respuesta seca.
- **GSM8K** (OpenAI, 2021): 8.5k problemas humanos. Es benchmark, no train.
- **SYNTH** (Langlais et al., 2026): 58.698 artículos Wikipedia, pre+mid+post en una etapa. Monad 56M.
- **DataDecide** (AllenAI): 25 corpora, 14 tamaños 4M-1B, 30k+ checkpoints. Para modelos pequeños, receta importa más que tamaño.

### 7.2 Receta V1 — 25k ejemplos, no 70k

| Fuente probada | % | Nº | Para qué órgano |
| :--- | :---: | ---: | :--- |
| TinyStories-style | 20% | 5.000 | Embedding Vivo, lenguaje base |
| BabyLM-style | 20% | 5.000 | Curriculum Fase 1, separación |
| FineWeb-Edu / DCLM muestreado | 20% | 5.000 | General, filtro anti-biomasa |
| Cosmopedia-style | 20% | 5.000 | Router, explicación, MLP |
| Orca-Math / TinyGSM-style | 15% | 3.750 | KAN Head, no colapso |
| Python-Edu / The Stack | 5% | 1.250 | Código, lógica |
| **Total** | **100%** | **25.000** | |

**Capa anti-atractor:** 80% probado da base. 20% propio da diferencia: español, esperpento, KOAN, Lexicón. Sin 80%, no aprende a hablar. Sin 20%, no es tuyo. La proporción se mantiene como objetivo del diseño y debe hacerse explícita en la composición final del dataset.

### 7.3 Reglas de filtrado — Si no cumple, se tira
1. Deduplicación: si un ejemplo se parece a >3 otros en embedding, fuera.
2. Anti-plantilla: si 10 empiezan igual, 9 se tiran. Ningún ejemplo con misma forma que otro.
3. KOAN solo en koan. Ninguna respuesta "KOAN: <número>" si pregunta no es koan.
4. Confirmado vs rumor: si es rumor, marcar como rumor. Nunca mezclar. Nunca rumor como confirmado.
5. Cada hecho distinto. Cada fórmula distinta. Cada epíteto distinto.
6. Mates: Problema + razonamiento paso a paso + solución. Nunca problema + respuesta seca. Si repites "Juan tiene X manzanas", se tira.
7. Código: Función pequeña + explicación línea a línea. Nunca solo código.

### 7.4 Script `data/build_dataset.py` debe
1. Descargar con `datasets` de Hugging Face, `streaming=True`, de cada fuente.
2. Muestrear N por fuente según tabla §7.2.
3. Filtro anti-plantilla: agrupar por primeros 5 tokens, si grupo >10, quedarse con 1.
4. Deduplicación: hash texto normalizado, si duplicado, fuera.
5. Filtro KOAN: si contiene "KOAN" y no es koan, fuera o re-etiquetar.
6. Confirmado vs rumor: marcar "confirmado" solo si viene de fuente educativa / GSM8K / Wikipedia. Si web sin verificar, marcar "rumor" o tirar.
7. Mezclar con seed fijo 42, guardar en `data/train_25k.jsonl` con formato: `{"prompt": "...", "completion": "...", "source": "...", "type": "lenguaje|mates|codigo|koan|factual", "confirmed": true/false}`.
8. Guardar `data/README.md` con fuentes, licencias, nº por fuente, seed, fecha.
9. NO entrenar sobre holdout. Verificar que ningún train está en holdout por hash. Si hay overlap, tirar ese train y re-muestrear.

### 7.5 Licencias
TinyStories, BabyLM, FineWeb, DCLM, Cosmopedia, SmolLM-Corpus, Orca-Math, TinyGSM, GSM8K: todos open, Apache 2.0 / MIT / CC. Verifica en cada repo antes de descargar. Si una fuente tiene licencia no comercial, no la uses. Reporta y sustituye. Guarda licencia de cada fuente en `data/README.md`.

---

## 8. ENTRENAMIENTO — PIPELINE PARA RUNNERS

### 8.1 Bloques
- Cada run GitHub = bloque de 500 pasos. Checkpoint al final. Siguiente run reanuda.
- No hay run de 10k de una vez. Son 20 runs de 500.
- Checkpoint cada 500 pasos, en `checkpoints/step_<N>.pt`, con: pesos, optimizador, scheduler, paso, seed, config, contadores homeostasis, memoria lenta.
- Checkpoints NO en git. En GitHub Releases / Artifacts.

### 8.2 Curriculum invertido
- **Fase 1 (0-2k):** Datos difíciles y muy distintos. Para separar. `--phase 1`. No ORPO.
- **Fase 2 (2k-8k):** Mezcla. 70% fácil, 30% difícil. `--phase 2`. No ORPO.
- **Fase 3 (8k-10k):** Solo difíciles y holdout-like. `--phase 3`. ORPO True.
- ORPO solo Fase 3. ORPO en Fase 1 hace que aprenda a gustar, no a saber.

### 8.3 Logs
Cada 100 pasos: loss token, loss latente, tasa neuronas muertas, entropía reparto energía, varianza respuestas en sumas. Si neuronas muertas >20%, parar.

### 8.4 Comandos que deben funcionar
```bash
python train/train.py --steps 500 --phase 1
python train/train.py --resume checkpoints/step_500.pt --steps 500 --phase 1
python train/train.py --steps 500 --no-sleep
python train/train.py --steps 500 --no-jepa
python eval/eval.py --checkpoint checkpoints/step_10000.pt --holdout data/holdout_500.jsonl
python scripts/count_params.py
```
Si un comando no funciona, refactor incompleto.

---

## 9. GITHUB ACTIONS — WORKFLOW RUNNERS GRATIS

**Archivo:** `.github/workflows/train.yml`

- Trigger: `workflow_dispatch` con inputs: `steps` (default 500), `phase` (1/2/3), `resume_from` (opcional).
- Runner: `ubuntu-latest`, sin GPU.
- Steps: Checkout → Setup Python 3.11 → `pip install -r requirements.txt` (CPU) → Descargar checkpoint previo desde Release / Artifact si `resume_from` no vacío → `python train/train.py --steps ... --phase ... --resume ...` → Subir checkpoint nuevo como Artifact `checkpoint-step-<N>` → Subir logs como Artifact.
- Timeout: 5h 30m (menos de 6h).
- No usar cache para checkpoints grandes. Usar Artifacts / Releases.
- Workflow separado `eval.yml` que evalúe checkpoint y suba resultados.
- NO inventes workflow que requiera GPU, secretos, o servicios de pago.

---

## 10. EVALUACIÓN — CÓMO SABREMOS SI NO DEGRADA

### 10.1 Holdout
- `data/holdout_500.jsonl`, congelado. No se toca. No se entrena.
- Si está contaminado, STOP. Avisa al dueño. No lo arregles en silencio.

### 10.2 Método
- **Sampling, no greedy.** Temp 0.8, top-p 0.9, repetition penalty 1.1.
- Si evalúas con greedy, resultado es mentira.

### 10.3 Métricas

| Métrica | Cómo | Umbral V1 |
| :--- | :--- | :--- |
| Factual | % hechos correctos | > baseline V2 |
| Diversidad | ¿Cuántas respuestas distintas a 100 preguntas distintas? | Si misma respuesta a todo, colapsó |
| Repetición | ¿Cuántas veces repite misma plantilla? | Si repite "KOAN: 115" >2 veces en 500, colapsó |
| Neuronas muertas | % que no disparan nunca en holdout | <20% |
| Energía | ¿Reparto varía según token? | Si siempre gasta igual, no aprendió |
| Perplexity holdout | Perplexity en holdout 500 | < baseline V2 |

### 10.4 Script `check_degradation.py`
- Carga checkpoint.
- Genera 100 respuestas con sampling.
- Cuenta cuántas idénticas, cuántas con "KOAN: 115" o plantilla.
- Imprime: `DEGRADADO` o `NO DEGRADADO` + métricas.
- Si `DEGRADADO`, no seguir entrenando. Parar y reportar.

---

## 11. RIESGOS

| Riesgo | Síntoma | Qué hacer |
|---|---|---|
| Homeostasis lenta | Neuronas se queman, luego oscilan | Probar ventana 256, 512, 1024. No cambiar eta_homeo sin permiso. |
| Top-K bajo | Modelo mudo | Probar K=16, 32, 64. Reportar. |
| Sueño borra demasiado | Pierde lo aprendido | Probar sueño cada 500 vs 1000. Flag `--no-sleep`. |
| KAN lento en CPU | Entreno eterno | KAN pequeño, solo operación. Reducir splines 32 → 24. |
| Dataset con plantilla | Tienta usar plantilla | No. Mejor 25k a mano que 70k con plantilla. |
| Contaminación holdout | Métricas infladas | STOP. Avisa. No arregles en silencio. |
| Checkpoint no reanuda | Pierdes 500 pasos | Test reanudación en M1. Si falla, no sigas. |

---

## 12. MILESTONES

- **M1 — Esqueleto:** Corre 10 pasos, checkpoint + reanudación OK.
- **M2 — No-degradación:** 2k pasos con 5k ejemplos anti-atractor. Sigue diverso, neuronas muertas <20%.
- **M3 — KAN:** Solo aritmética, con y sin KAN. Con KAN no colapsa, sin KAN colapsa.
- **M4 — Run completo 10k:** 10k pasos en bloques runners. Factual > V2, repetición < V2.
- **M5 — Publicar:** Spec + código + holdout reproducibles. Sin eso, no es ciencia. Es anécdota.

---

## 13. CHECKLIST FINAL — ANTES DE DECIR "LISTO"

- [ ] V2 archivada en `archive/v2/` intacta.
- [ ] Estructura §5.2 creada exactamente.
- [ ] `config.py` única fuente de verdad.
- [ ] Todos los órganos implementados, con test, sin GPU, sin atención cuadrática.
- [ ] `count_params.py` da 3-4M parámetros, 20-30MB float16.
- [ ] `train.py` corre 10 pasos, checkpoint, reanuda.
- [ ] Dataset 25k con receta §7, sin plantilla, sin contaminación holdout.
- [ ] Workflow GitHub Actions corre 500 pasos y sube checkpoint.
- [ ] Eval con sampling, no greedy, con 6 métricas §10.
- [ ] README actualizado: V2 archivada, V1 es RONIN-LUMEN, cómo entrenar, evaluar, reproducir.
- [ ] SPEC copiado, decisiones durante refactor marcadas.
- [ ] Ningún archivo con código de la V1 anterior descartada. Ningún archivo dice que esa V1 anterior está vigente.

Si falta un checkbox, no está listo. No digas "casi listo". Di qué falta.

---

## 14. ESTILO DE CÓDIGO

- Python claro, sin magia. Sin one-liners que nadie entiende.
- Comentarios que explican POR QUÉ, no QUÉ.
- Nombres en inglés para código, comentarios en español si es más claro.
- Sin dependencias pesadas. Solo `torch`, `numpy`, `datasets`, `tqdm`.
- Sin `print` sueltos. Usa `logging` o función `log()`.
- Sin números mágicos. Todo en `config.py`.
- Cada función con docstring de 1 línea: qué hace, qué devuelve.

---

## 15. PROHIBIDO

- NO resucites la V1 anterior descartada.
- NO uses atención cuadrática.
- NO uses RoPE.
- NO uses vocabulario de 32k.
- NO uses dataset de 70k con plantilla.
- NO entrenes sobre holdout.
- NO evalúes con greedy.
- NO uses GPU, ni código que requiera GPU.
- NO añadas features que no estén en este documento.
- NO inventes datasets. Usa los de §7.
- NO mezcles confirmado y rumor.
- NO pongas KOAN donde no toca.
- NO subas checkpoints a git.
- NO digas "listo" si falta un checkbox de §13.

---

## 16. PRIMERA ACCIÓN — QUÉ HACES NADA MÁS LEER ESTO

1. Haz inventario Paso 0.
2. Archiva V2.
3. Crea esqueleto M1.
4. Para y reporta: "M1 listo, esqueleto corre 10 pasos, checkpoint + reanudación OK, X archivos inventariados, V2 archivada".
5. Espera confirmación del dueño antes de seguir con órganos.

No hagas M1+M2+M3+M4 de una vez. Haz M1, para, reporta.

---

**Fin del documento maestro.**  
**Si tienes dudas, pregunta al dueño. No inventes.**  
**Si este documento y el código se contradicen, manda este documento.**

---

## ANEXO A — TRAZABILIDAD DE LA ESPECIFICACIÓN

- Este archivo es la transcripción íntegra y normalizada del documento maestro facilitado por el dueño en la conversación del 21-11-2026.
- El cambio editorial deliberado respecto al título original es únicamente el nombre de release: donde el título de trabajo decía V3, el proyecto se denomina RONIN-LUMEN V1 por decisión explícita del dueño.
- Las referencias a la antigua V1 descartada se han aclarado como “V1 anterior descartada” para evitar confundirla con RONIN-LUMEN V1.
- Los requisitos técnicos, fórmulas, dimensiones, umbrales, fases, dataset, métricas y prohibiciones se mantienen.
- Las referencias científicas y sus cifras se reproducen como afirmaciones del documento de diseño; antes de usarlas como evidencia pública del rendimiento de LUMEN, deberán verificarse sus fuentes y licencias.
- Cualquier discrepancia técnica, imposibilidad o contradicción interna se documentará como pregunta/decisión pendiente. No se modifica la spec en silencio.
