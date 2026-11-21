# RONIN-LUMEN V1

**A small-model architecture experiment in surprise-gated state, homeostatic sparsity, elastic memory, consolidation, sleep, and split language/operation routing — designed for CPU-only training on free GitHub Runners.**

RONIN-LUMEN is the successor refactor of WISE-RONIN. It is not a new large-model race and does not claim to outperform foundation models. Its research question is narrower and harder: **can a 3–4M-parameter model retain a differentiated voice and useful task behavior through continued training, instead of collapsing into repeated attractor templates?**

> **An architecture proposal, not a proven result.** The mechanisms below are a deliberately integrated experimental design. Their novelty is claimed at the level of this particular system design and engineering combination, not as a claim to have invented delta rules, KANs, JEPA, sparse activation, homeostasis, or memory consolidation individually. The hypothesis must be earned through ablations, reproducible checkpoints, and a frozen holdout.

## The Carmack–Naka principle: make the constraint the design

RONIN-LUMEN belongs to an older and more demanding engineering tradition: **do not answer a resource limit by asking for more resources; change the design until the available machine is enough.**

John Carmack's work on the *Doom* and *Quake* engines is a useful emblem of that discipline. The lesson is not a single famous bit trick or a nostalgic claim that old hardware was better. It is the architectural habit of understanding the machine's actual costs—memory traffic, cache locality, arithmetic, bandwidth, and the shape of the workload—and then choosing representations that fit those costs. The fast inverse-square-root routine associated with Quake III is memorable; the deeper achievement was designing an engine around the constraints instead of treating optimization as a cleanup phase.

Yuji Naka represents the same refusal from the console side. The Mega Drive did not offer modern memory or graphics budgets. Shipping a fast, responsive game meant making every update, sprite, tile, and byte earn its place. Precomputation, reuse, careful assembly, and intimate knowledge of the video hardware were not decorative tricks. They were part of the design.

Colin Chapman's famous engineering maxim—**“Simplify, then add lightness”**—completes the triangle. Carmack asks what the machine really costs. Naka asks how far disciplined reuse can stretch a fixed budget. Chapman asks what can be removed before anything is added.

That is the intended spirit of WISE-RONIN and RONIN-LUMEN:

- **CPU is the target, not a fallback.** The project is designed around ordinary GitHub-hosted CPU runners, bounded execution windows, resumable jobs, and explicit memory budgets—not around an assumed GPU.
- **Architecture before brute force.** Delta state, sparse activity, bounded memory, routing, and consolidation are hypotheses for replacing expensive or poorly matched computation. They must earn their place through ablations and measurements.
- **Every parameter and every data row has a cost.** Vocabulary size, embedding layout, corpus provenance, repeated templates, checkpoint contents, and inference state are engineering decisions, not afterthoughts.
- **Reuse beats accumulation.** A compact mechanism that can be measured, tested, and reused is preferable to a pile of loosely connected modules.
- **No benchmark theatre.** A falling loss is not enough. We measure holdout behavior, repetition, diversity, task separation, CPU throughput, memory use, and checkpoint reproducibility.

LUMEN is therefore not presented as “a Transformer, only smaller,” nor as a proven replacement for attention. It is an experiment in a different allocation of computation: compact recurrent state, surprise-dependent updates, sparse activity, elastic and slow memory, and separate language/operation paths. Those choices may fail. If they do, the ablations should tell us which idea failed and why.

The ambition is deliberately modest in hardware and severe in method: **make the available machine sufficient by removing waste, not by pretending constraints do not exist.** A small model running on a free CPU runner is not automatically an achievement. A small model whose design, data, training, and evaluation are all honest about that constraint—that is the experiment.

## Why LUMEN is different

Most tiny decoder experiments still inherit the same basic recipe: token embeddings, repeated self-attention blocks, feed-forward layers, and next-token cross-entropy. WISE-RONIN showed why a low training loss is not enough: a small model can acquire a recognizable voice and still progressively collapse into repeated templates, a KOAN attractor, or token corruption.

LUMEN proposes a different allocation of computation and memory:

1. **Surprise-gated delta state:** write to a compact recurrent state only when prediction error crosses a learned threshold; otherwise allow the state to decay.
2. **Homeostasis and Top-K sparsity:** regulate neuron activity over time and keep only a small active subset, treating silence as part of the representation.
3. **Elastic and slow memory:** separate short-lived, self-updated memories from patterns consolidated into a nearly read-only store.
4. **Per-token energy budget:** divide a fixed unit budget between delta state, memory, and the task head instead of letting every subsystem run at full strength.
5. **Separated language and operation paths:** route linguistic generation to an MLP head and arithmetic/logical operations to a compact KAN head.
6. **Sleep and latent prediction:** periodically consolidate/reconstruct stored patterns without new external examples, and train a JEPA-style latent prediction objective alongside token prediction.
7. **Living vocabulary:** begin with a corpus-trained 4,096-token BPE vocabulary and cautiously prune/replace unused entries under explicit limits.

These are design hypotheses, not established benefits. The project will report negative results as carefully as positive ones. In particular, the system must demonstrate that each mechanism helps under controlled ablation; a complicated architecture is not evidence of an effective one.

## The central research claim

**LUMEN treats non-degradation as an architectural objective, not merely a data-cleaning problem.**

The proposed model attempts to prevent a few highly reinforced patterns from monopolizing all representational capacity by combining surprise-dependent writes, usage homeostasis, sparse activation, distinct memory timescales, and task routing. The central claim to test is that this combination can preserve response diversity and functional separation during continued training at a scale where a conventional repeated decoder can be vulnerable to attractor collapse.

We will not claim this hypothesis is validated until:
- a frozen, leakage-audited holdout has been evaluated consistently;
- the same sampling protocol is used across checkpoints and baselines;
- diversity, repetition, factuality, dead-neuron rate, energy allocation, and perplexity are reported;
- each major subsystem has an ablation;
- checkpoint/resume and CPU-only runs are reproducible.

## Architecture at a glance

```text
Token
  -> Living BPE Embedding
  -> 1.0 Per-token Energy Budget
  -> Surprise-Gated Delta State
  -> Homeostasis + Top-K Sparsity
  -> Elastic Memory / Slow Consolidation
  -> Router
       -> Language MLP
       -> Operation KAN
  -> Token logits
  -> JEPA latent prediction loss (training only)

Every 500 training steps:
  -> 50 sleep steps
  -> consolidate/reconstruct slow memories
  -> reset fast state and elastic memory
```

### Fixed design envelope

| Constraint | Target |
|---|---:|
| Parameters | 3–4M target; 6M hard ceiling |
| Model storage | 20–30MB, float16/int8 |
| Hidden dimension | 256 |
| Initial vocabulary | 4,096 BPE tokens |
| Maximum vocabulary | 8,192 |
| Training hardware | CPU-only free GitHub Runners |
| Run block | 500 steps, under 6 hours |
| Checkpoint cadence | Every 500 steps |
| Total initial schedule | 10,000 steps |
| Evaluation | Frozen 500-record holdout; sampling, not greedy |

The V1 baseline requirements, equations, dataset recipe, training schedule, metrics, and stop conditions are in [DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md](DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md). The optional experimental extension is specified separately in [RONIN-LUMEN V3.2 — ZE / LC-NE](docs/RONIN_LUMEN_V3_2_ZE.md); V3.0 remains the default.

## Project status — architecture integration and runner validation

<sub>Current branch: `lumen-v1/m1-refactor` · Pull request: #13 · Not merged</sub>

The branch now contains executable implementations of the eight architectural organs and a CPU trainer, with runner validation in progress:
- four recurrent surprise-gated Delta layers with explicit batch-boundary state reset;
- live embedding usage accounting and non-destructive prune proposals;
- per-token energy allocation, homeostasis and Top-K sparsity;
- elastic memory and a bounded slow-memory bank with consolidation;
- a learned router, spline-KAN operation head and language MLP;
- sleep consolidation over stored slow memories, without embedding updates;
- JEPA-style next-eight-token latent prediction and a differentiable repetition regularizer;
- an opt-in V3.2 dual-state M_A/M_B controller, ten-token conflict localization, bounded rank-one context erasure, ternary PE memory gate, adaptive 80%-occupancy sleep with 1/3/7 replay spacing, and a state-distribution KL complexity term;
- checkpoint/resume for model, optimizer, scheduler, global step, corpus hash and RNG states;
- curriculum selection with strict phase windows and a 70/30 stratified phase-2 sampler.

The integrated model currently has **2,281,841 parameters**. This is below the 6M ceiling, but below the 3–4M target band, so the target-band criterion remains unmet. This count is a static model count, not a measured on-disk fp16/int8 package size.

The runner workflow now compiles the model, checks the parameter budget, runs unit tests and performs an explicit 10-step dummy smoke. These jobs must pass before the architecture can be considered runner-validated. The dummy mode is only a smoke test; it is not evidence of learning and is never used as the production corpus.

V3.2 is opt-in via `--v32-ze`; it is not the default and is not yet a demonstrated improvement. Its tests and CPU checkpoint/resume smoke are being exercised in Actions.

Some mechanisms remain deliberately gated rather than falsely claimed complete:
- vocabulary pruning currently emits candidates only; actual token replacement requires an atomic tokenizer/embedding remap;
- ORPO fails closed until preference pairs and its objective are specified;
- fake-int8 is an activation simulation, not a quantized model export;
- slow-memory consolidation and sleep need behavioral tests and ablations;
- the final training corpus and BPE tokenizer do not yet exist.

### Smoke and production commands

```bash
python -m pip install -r requirements.txt
python -m train.train --dummy-m1 --steps 10 --phase 1
# Optional experimental branch (separate checkpoints; not the V3.0 baseline)
python -m train.train --dummy-m1 --v32-ze --ckpt-dir checkpoints/v32 --steps 10 --phase 1
python scripts/train_tokenizer.py --data data/train_25k.jsonl
python -m train.train --steps 500 --phase 1
```

Production training is blocked until the 25,000-row corpus, tokenizer, provenance manifest and holdout-overlap audit are complete.

## Data preparation status

The source sampler is designed to stage 19,000 externally sourced candidates toward the 20,000-row external allocation. It records immutable Hugging Face dataset revisions, source licenses, per-file hashes and a manifest. Python-Edu/The Stack's 1,000-row code quota remains blocked pending per-file license and attribution checks. The remaining 5,000 rows must come from the owner's own curated corpus.

The initial source-candidate artifact has been downloaded and audited. Its first lexical audit found 39 normalized duplicates; the sampler has since been hardened to reject short fragments and normalize text before deduplication, and the updated staging workflow is rebuilding and re-auditing the artifact. The final `train_25k.jsonl` has **not** been built. The staging workflow is intentionally separate from model training. See [data/README.md](data/README.md) for source-by-source conditions and [docs/SPEC_AUDIT.md](docs/SPEC_AUDIT.md) for unresolved specification decisions.

## Research discipline

- **Voice first, factual second.** Keep the intended corpus composition and task boundaries explicit.
- **Confirmed is not rumor.** Preserve provenance and never silently promote unverified material to fact.
- **KOAN stays in KOAN examples.** It must not become a universal response attractor.
- **Curated diversity beats repeated synthetic templates.** If a generator produces near-identical forms, filter them rather than inflating the dataset.
- **The holdout is sacred.** Preserve the 500 source records byte-for-byte, audit overlap, and never train on them.
- **Ablate, don't assume.** Sleep, JEPA, KAN, memory, sparsity, homeostasis, energy routing, and vocabulary pruning each need measurable tests.
- **Reproducibility over spectacle.** Record configuration, seeds, optimizer/scheduler state, model state, and adaptive state in checkpoints.
- **Report the limits.** The preserved historical holdout contains 500 records but only 55 unique prompt strings. That limitation must be disclosed; it is not silently repaired.

## Roadmap and merge policy

1. **M1 — Skeleton:** inventory, V2 archive, 10-step CPU run, checkpoint and resume.
2. **M2 — Organs:** implement and test each organ individually in the exact order defined by the master spec.
3. **M3 — Integration and ablations:** verify parameter budget, arithmetic routing, diversity, and no-degradation behavior.
4. **M4 — Data and training:** build the curated corpus, audit licenses/holdout leakage, and train in resumable 500-step runner blocks.
5. **M5 — Evaluation and release:** publish reproducible results, negative findings, ablations, and the complete specification.

The owner has explicitly authorized the merge of this architecture/integration PR. That authorization does **not** waive the data, tokenizer, holdout, or training gates described above. No production training may start until those gates pass.

## License and provenance

Dataset sources, code dependencies, and model artifacts must each have their licenses and provenance recorded before inclusion. The archive preserves the prior repository state; it does not imply that every historical dataset is cleared for redistribution. See [the full master specification](DOCUMENTO_MAESTRO_RONIN_LUMEN_V1.md) and [data/README.md](data/README.md).

---

*RONIN-LUMEN is an experiment in making a small model remember when to write, when to remain silent, what to consolidate, and which computational path a token should take. The claim is not that these ideas are already proven; the claim is that the architecture is specific enough to test, falsify, and reproduce.*
