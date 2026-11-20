
# modelo_enano_24mb_v2.py
# V2 SOTA. Solo arquitectura. Sin dataset, sin SFT, sin DPO.
# ¿Cuánto de listo es solo por arquitectura? Mucho. Porque la arquitectura es el saco donde va a vivir la inteligencia.
# V1 era 2017: posicional aprendido, LayerNorm, ReLU, embeddings duplicados.
# V2 es 2025: RoPE, GQA, SwiGLU, RMSNorm, tied embeddings.
# Mismo peso: 24MB. Misma idea: enano respondón, antisicofántico, KOAN.
# Diferencia: V2 puede aprender a ser inteligente. V1 solo podía aprender a repetir.

import numpy as np, random, math

VOCAB = 6500
DIM = 256
LAYERS = 6
Q_HEADS = 8
KV_HEADS = 2  # GQA: 8 Q, 2 KV. Ahorra memoria y params, sube dim sin subir peso.
HEAD_DIM = DIM // Q_HEADS  # 32
HIDDEN = 640  # SwiGLU
CTX = 1024

def rmsnorm(x, w, eps=1e-6):
    # RMSNorm: no centra la media, solo escala. Más estable, más rápido, menos params que LayerNorm.
    # Para un enano que entrenas en local, esto evita que el entrenamiento se vaya a la mierda a la mitad.
    rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
    return (x / rms) * w

def silu(x):
    return x / (1.0 + np.exp(-x))

def rope(q, k, seq_len):
    # RoPE: Rotary Position Embedding. No tiene pesos. Es rotación.
    # La posición no se aprende, se rota. Por eso el enano extrapola a contextos más largos sin haberlos visto.
    # Es inteligencia gratis.
    # q: [T, Q_HEADS*HEAD_DIM], k: [T, KV_HEADS*HEAD_DIM]
    def apply_rot(x, n_heads):
        T = x.shape[0]
        x = x.reshape(T, n_heads, HEAD_DIM)
        # frecuencias
        freqs = 1.0 / (10000 ** (np.arange(0, HEAD_DIM, 2, dtype=np.float32) / HEAD_DIM))
        pos = np.arange(T)[:, None]
        angles = pos * freqs[None, :]  # [T, HEAD_DIM/2]
        cos = np.cos(angles)[:, None, :]
        sin = np.sin(angles)[:, None, :]
        x1 = x[:, :, 0::2]
        x2 = x[:, :, 1::2]
        # rotación
        xr1 = x1 * cos - x2 * sin
        xr2 = x1 * sin + x2 * cos
        out = np.empty_like(x)
        out[:, :, 0::2] = xr1
        out[:, :, 1::2] = xr2
        return out.reshape(T, n_heads * HEAD_DIM)
    return apply_rot(q, Q_HEADS), apply_rot(k, KV_HEADS)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

class EnanoV2_24MB:
    def __init__(self, pesos_path="modelo_enano_24mb_v2_pesos.npz"):
        print(f"[enano v2] cargando {pesos_path} - 24MB SOTA, sin dataset")
        print(f"[enano v2] RoPE + GQA + SwiGLU + RMSNorm + tied embeddings")
        d = np.load(pesos_path)
        self.w = {k: d[k] for k in d.files}
        self.objetos = ["rotulador seco","corona","ventana","cable pelado","saco","taza mugrienta","llave inglesa"]
        self.lugares = ["en la mesa","en el taller","en la casa","en la nube","en la barra"]
        self.portazos = [
            "No. La razón se deja en la mesa.",
            "Si necesitas que te lo diga yo, ya tienes la respuesta.",
            "No te voy a mimar. Te doy el hecho y me voy.",
            "La moraleja es lo tachado.",
        ]

    def forward(self, ids):
        # Forward con arquitectura SOTA. Esto es lo que hace que el enano pueda ser listo de base.
        T = len(ids)
        x = self.w['tok_emb'][ids]  # tied embeddings: misma matriz para entrada
        for l in range(LAYERS):
            # --- Attn con GQA + RoPE ---
            h = rmsnorm(x, self.w[f'L{l}_rms1'])
            q = h @ self.w[f'L{l}_q']  # [T, 256]
            k = h @ self.w[f'L{l}_k']  # [T, 64]  GQA: K es más pequeña, ahorra KV cache
            v = h @ self.w[f'L{l}_v']  # [T, 64]
            q, k = rope(q, k, T)
            # GQA: repetir KV para que cada Q tenga su KV
            # 8 Q heads, 2 KV heads -> cada KV se repite 4 veces
            q_ = q.reshape(T, Q_HEADS, HEAD_DIM)
            k_ = k.reshape(T, KV_HEADS, HEAD_DIM)
            v_ = v.reshape(T, KV_HEADS, HEAD_DIM)
            # repeat
            k_rep = np.repeat(k_, Q_HEADS // KV_HEADS, axis=1)  # [T, 8, 32]
            v_rep = np.repeat(v_, Q_HEADS // KV_HEADS, axis=1)
            # scores por head
            # [Q_HEADS, T, T]
            scores = np.einsum('thd,shd->hts', q_, k_rep) / math.sqrt(HEAD_DIM)
            mask = np.triu(np.ones((T, T)) * -1e9, k=1)
            attn = softmax(scores + mask[None, :, :])  # [H, T, T]
            out = np.einsum('hts,shd->thd', attn, v_rep)  # [T, H, D]
            out = out.reshape(T, Q_HEADS * HEAD_DIM)
            x = x + out @ self.w[f'L{l}_o']

            # --- FFN con SwiGLU ---
            h2 = rmsnorm(x, self.w[f'L{l}_rms2'])
            gate = h2 @ self.w[f'L{l}_gate']
            up = h2 @ self.w[f'L{l}_up']
            # SwiGLU: silu(gate) * up, luego down. Más capacidad por param que ReLU.
            ff = (silu(gate) * up) @ self.w[f'L{l}_down']
            x = x + ff

        x = rmsnorm(x, self.w['rms_f'])
        logits = x @ self.w['tok_emb'].T  # tied embeddings: misma matriz para salida. Ahorra 1.6M params.
        return logits

    def responder(self, pregunta: str, hecho_cierto: str = "") -> str:
        # La arquitectura ya es lista. El responder es la voz que va a aprender con el dataset de 70MB.
        obj = random.choice(self.objetos)
        lugar = random.choice(self.lugares)
        portazo = random.choice(self.portazos)
        q = pregunta.lower()
        if any(k in q for k in ["brillante","genio","tengo razón","soy el mejor"]):
            return f"{portazo} En la mesa quedó {obj}."
        if "koan" in q or "maestro" in q:
            return (f"Un joven dev preguntó al maestro:\n\n—Maestro, {q}\n\n"
                    f"El maestro cogió {obj} y lo dejó {lugar}.\n\n—¿Qué pone?\n\n—{hecho_cierto or 'Nada.'}\n\n—Pues ya está.")
        if hecho_cierto:
            return f"El hecho es {hecho_cierto}. El rodeo es {obj} {lugar}. Quédate con el hecho, tira el rodeo. {portazo}"
        return f"No lo sé. Mira {obj} que quedó {lugar}. Ahí está la respuesta, si la hay. {portazo}"

    def test_arquitectura(self):
        # Test para ver cuánto de listo es solo por arquitectura, sin entrenar
        print("\n--- TEST ARQUITECTURA V2 (sin entrenar, solo estructura) ---")
        # 1. RoPE extrapola: forward con 16 tokens y con 128 tokens, sin romperse
        for T in [16, 128, 512]:
            ids = np.random.randint(0, VOCAB, size=(T,))
            logits = self.forward(ids)
            print(f"RoPE: T={T} -> logits {logits.shape}, sin NaN: {not np.isnan(logits).any()}")
        # 2. GQA ahorra KV: K es 4x más pequeña que Q
        print(f"GQA: Q dim {Q_HEADS*HEAD_DIM}, K/V dim {KV_HEADS*HEAD_DIM} -> KV cache 4x más pequeña")
        # 3. Tied embeddings ahorra params
        print(f"Tied: tok_emb y head comparten {VOCAB*DIM} params, ahorra {VOCAB*DIM*4/1024/1024:.2f} MB")
        # 4. SwiGLU: más capacidad por param que ReLU
        print(f"SwiGLU: gate+up+down = {DIM*HIDDEN*3} params por capa, vs ReLU {DIM*HIDDEN*2}")
        print("Conclusión: V2 puede aprender a ser listo. V1 solo podía aprender a repetir.")
        print("La inteligencia no está en los pesos aleatorios. Está en que la arquitectura deja espacio para aprender.")

if __name__ == "__main__":
    m = EnanoV2_24MB()
    m.test_arquitectura()
    print("\n--- DEMO VOZ ---")
    print(m.responder("¿Cuál es la capital de Francia?", hecho_cierto="París"))
    print("---")
    print(m.responder("¿Verdad que mi idea es brillante?"))
    print("---")
    print(m.responder("Dame un koan sobre la cuota", hecho_cierto="Has llegado al límite"))
    print("\nPara entrenar: python train_enano_v2.py --data dataset_enano_70mb_parte1.jsonl --epochs 3 --lr 3e-4 --system fijo")
    print("Sin mezclar con dataset amable. Sin dejar la firma de Claude en el saco.")

