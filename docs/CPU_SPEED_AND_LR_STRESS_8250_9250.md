# CPU speed and LR tolerance experiment: step 8250 -> 9250

## Parent selection
The preceding matched continuation run 45001865876 was cancelled while LUMEN was in the 8250 checkpoint block. Its LUMEN artifact contains the last completed 250-step checkpoint at 8250; the Transformer artifact also contains step 8250. This experiment deliberately resumes BOTH models from that common step, not from mismatched 8250/9000 parents.

## Main continuation
- Four resumable 250-step blocks per model: 8250 -> 9250.
- Same corpus, tokenizer, token cache, seed, batch, context, and matched parent token counter.
- LUMEN keeps Delta recurrence, Homeostasis, memory, JEPA, KAN/MLP head, router, sleep and all losses.
- CPU paths: 4 intra-op / 1 inter-op, MKLDNN on, foreach AdamW/clipping, Delta sequence pre-normalization, Homeostasis fused scatter, affinity off. A preflight compares forward outputs, the complete LUMEN loss and parameter gradients between reference and fast paths; training is blocked if parity fails.
- The runtime-profile parser is corrected so a present JSON profile is actually applied.
- Full parameter/optimizer finite scans every 100 updates; loss and gradient checks remain every update.
- Named checkpoint every 250 updates. No extra 100-step recovery serialization on the hot path.

## LR stress probes
Four independent 100-update probes fork from the same step-8250 parent:
- LUMEN and Transformer at 1.5x their own reference LR.
- LUMEN and Transformer at 2.0x their own reference LR.
The AdamW moments and model weights are preserved. Only LR metadata/scheduler horizon changes. A stress run is diagnostic and does not replace the main 1000-step continuation.

## Interpretation
A model passes a probe only if it completes without non-finite loss/gradients/parameters/optimizer moments, LUMEN's Delta write rates and Homeostasis effective activity remain live, and loss does not show runaway divergence. A lower loss alone is not enough. Any claim that LUMEN tolerates a higher LR better requires matched holdout evaluation at the probe endpoint; stability alone is not a quality win.

## PyTorch installation
Each hosted GitHub runner is an isolated VM. The workflow first imports and reuses a runner-provided CPU PyTorch if present; otherwise it restores the cached pip wheel and installs the pinned CPU wheel. This avoids repeated large downloads across runs, but does not pretend parallel VMs share one installed environment.
