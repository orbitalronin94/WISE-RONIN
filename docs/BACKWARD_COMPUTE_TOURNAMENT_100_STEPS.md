# LUMEN backward-compute tournament — 100 steps

This experiment compares eight isolated mechanisms against an unmodified main
control. Each candidate job runs a sequential pair on one GitHub-hosted CPU
runner. Both arms start from the exact LUMEN step-389700 checkpoint and use the
same frozen corpus, tokenizer, token cache, seed, row order, batch size and
restored AdamW state.

Parent run: https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45273897576

## Candidates

| Candidate | Mechanism | Interpretation |
|---|---|---|
| 01_surprise_gate | Skip backward and optimizer update when loss <= 1.8 | Measures whether low-loss batches can be skipped without harming learning |
| 02_topk10 | Keep top 10% of each dense parameter gradient | Sparse updates only; dense backward still executes |
| 03_wta8 | Top-1 activation mask per group of 8 after homeostasis | Post-layer sparsity; dense matrix work is not avoided |
| 04_stopgrad2 | Detach outputs of the first two homeostasis layers | Changes the learning graph and may freeze lower representations |
| 05_lowrank8 | Rank-8 SVD projection of dense matrix gradients | Low-rank update approximation, not a LoRA forward/backward implementation |
| 06_lazy_backward | Alternate full backward with cached gradients decayed by 0.9 | Trades compute for stale-gradient error |
| 07_homeostatic_norm | Per-feature RMS normalization after each homeostasis layer | Architecture/activation-dynamics change, not an equivalent LayerNorm replacement |
| 08_sleep_replay20 | One sleep replay at the end, top 20% records, 2x parent LR | Only valid if consolidated records exist and replay actually executes |

## Promotion gate

A candidate is approved only if all conditions hold:

1. Mean training loss over 100 steps is strictly lower than control.
2. Mean loss over the final 20 steps is no higher than control.
3. End-to-end measured loop time is strictly lower than control.
4. The proposed mechanism was actually exercised.

All other outcomes remain PENDING; a failed or unexercised candidate is not
silently treated as a negative scientific result.

## Measurements

Each JSON result records mean and final-window loss, elapsed time, steps/s,
forward/backward/optimizer/replay time, skipped backward steps, replay execution,
parent checkpoint SHA-256, corpus hash and tokenizer hash.

The wall-time comparison includes token-cache batch selection and the training
step loop, but excludes Python startup, package installation, artifact download
and model/checkpoint loading. Both arms use the same runner sequentially.
Matrix parallelism is capped at four jobs to reduce host contention.

## Important compute accounting

- Top-k gradient masking happens after dense backward. It cannot be described as
  a backward-FLOP reduction.
- WTA is applied after the dense homeostasis operation. It does not avoid the
  matrix multiplication that already happened.
- Rank-8 SVD projection also happens after dense backward and adds projection
  cost. It is an update-compression experiment, not true LoRA.
- Only surprise gating and lazy backward directly skip backward calls in this
  implementation. Stop-gradient may reduce downstream gradient propagation,
  but changes the learning graph.
- The 100-step run is a short training microbenchmark, not a holdout evaluation
  and not evidence of long-horizon convergence.

No candidate is merged or enabled in the production trainer by this workflow.
