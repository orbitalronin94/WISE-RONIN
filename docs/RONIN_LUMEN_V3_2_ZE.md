# RONIN-LUMEN V3.2 — ZE / LC-NE experimental extension

**Status:** proposal implemented as an opt-in experimental track; not a claim of biological equivalence or proven anti-perseveration.

This extension preserves the V3.0 baseline. It is gated behind an explicit runtime flag and must be evaluated against the same frozen 500-row holdout and controlled synthetic attractor tests. No holdout row may enter training or sleep replay.

## Evidence boundary

The cited neuroscience motivates hypotheses, not direct engineering laws. In particular:
- Friston's variational/free-energy framework motivates a precision/complexity trade-off; a neural-network loss with a KL regularizer is only a variational-inspired objective, not automatically biological free energy.
- Sleep replay and synaptic homeostasis motivate replay/consolidation experiments; they do not prove that a 1/3/7 schedule is optimal for this model.
- LC–NE research motivates adaptive learning-rate/gating hypotheses; PE thresholds 0.3/0.8 are initial engineering thresholds that must be calibrated on training-only statistics.
- The claims attributed to 2025/2026 papers (including Burke/Pednekar and the December 2025 Cell Reports result) require exact DOI/title verification before being cited as settled evidence in public documentation. "Ze 2026" is treated as the user's name for the proposed mechanism, not as an independently verified paper.

## 1. Dual model M_A / M_B

- M_A is the existing four-layer 256-dimensional causal Delta stack.
- M_B is one 128-dimensional Delta predictor. Its input is produced by a learned counterfactual projection from the current hidden representation.
- (F_A) and (F_B) are normalized prediction-error/surprise estimates. (Delta F=|F_A-F_B|) measures disagreement in error magnitude only; it is **not** sufficient evidence that predictions agree.
- The controller also tracks prediction-state disagreement in the shared 256-dimensional space. Low conflict blends the paths. High conflict selects the lower-error path and resets the weaker recurrent state.
- M_B is a learned counterfactual proxy, not a true "previous token replaced" counterfactual. True counterfactual supervision would require paired/intervened training examples.

## 2. LC–NE-inspired ternary prediction-error gate

Initial normalized PE bands:
- small: PE < 0.3 — elastic-memory edit path, with a reduced gradient scale;
- medium: 0.3 ≤ PE ≤ 0.8 — no memory write;
- large: PE > 0.8 — write a new slow-memory slot; do not overwrite an existing record.

These values are starting thresholds, not universal constants. Log PE quantiles and per-band counts. Do not silently rescale thresholds based on holdout data.

## 3. Operator ℰ (context decorrelation)

When high dual-model conflict persists for more than 10 consecutive tokens, estimate a context direction from the M_A/M_B disagreement vector and remove its rank-one component from the fast Delta state:

[
S' = S - \frac{(S e)e^T}{e^T e + \epsilon}.
]

This is a projection heuristic. It cannot identify a semantic label such as "KOAN" or "math" by itself; the direction is an internal disagreement/context proxy. Log activation count, removed-state norm, and whether conflict falls afterward. Bound the operator to avoid erasing useful state.

## 4. Adaptive sleep and IRI-style replay

Sleep is triggered by slow-memory occupancy reaching 80%, rather than a fixed 500-step clock. The 1/3/7 replay intervals are implemented as a small deterministic replay schedule over consolidated records. This is a testable spacing heuristic, not a claim that Burke et al. or Pednekar et al. established these exact intervals for neural language models.

## 5. Variational-inspired free-energy regularizer

[
F = L_{token} + \beta\, KL(q(S_t)\|q(S_{t-1})) + \text{existing auxiliary losses}.
]

The implementation maps each row of the Delta state to a categorical distribution with softmax and computes row-wise KL. Initial beta is 0.01. Log the complexity term separately and run beta=0 ablations. The KL is a regularizer over internal state distributions, not a calibrated physical energy.

## Acceptance gates

1. V3.0 and V3.2 use identical seeds, data, tokenizer, optimizer budget and sampling.
2. Run at least 5 seeds on training-only synthetic attractor probes (KOAN loop, repeated-token attractor, false-context cue, arithmetic-vs-KOAN context).
3. Frozen holdout remains byte-identical and excluded from all training, tokenizer and sleep inputs.
4. Report repetition/KOAN counts, factual accuracy, arithmetic accuracy, diversity, loss, PE-band frequencies, M_A/M_B conflict, ℰ activation, memory occupancy and sleep count.
5. Reject the extension if factual/math accuracy drops, dead neurons exceed 20%, or gains disappear across seeds.
6. No public claim of reduced perseveration until the ablation passes.

## Rollout

- V3.0 remains default.
- V3.2 is opt-in via `--v32-ze`.
- Checkpoints record the feature flag and V3.2 state.
- The pull request stays draft and unmerged pending runner tests, corpus/licence audit and holdout gate.
