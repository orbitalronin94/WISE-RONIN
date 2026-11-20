<div align="center">

# WISE-RONIN

### Small models. Deliberate training. Measurable behavior.

[![GitHub stars](https://img.shields.io/github/stars/orbitalronin94/WISE-RONIN?style=for-the-badge&logo=github&label=Stars)](https://github.com/orbitalronin94/WISE-RONIN/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/orbitalronin94/WISE-RONIN?style=for-the-badge&logo=github&label=Forks)](https://github.com/orbitalronin94/WISE-RONIN/network/members)
[![Latest commit](https://img.shields.io/github/last-commit/orbitalronin94/WISE-RONIN?style=for-the-badge)](https://github.com/orbitalronin94/WISE-RONIN/commits/main)
[![License](https://img.shields.io/github/license/orbitalronin94/WISE-RONIN?style=for-the-badge)](LICENSE)

[Architecture](#architecture) · [Experiments](#experiments) · [Datasets](#datasets) · [Training](#training) · [Evaluation](#evaluation) · [Roadmap](#roadmap)

**WISE-RONIN** is an experimental, compact decoder-only Transformer exploring factual responses, a distinctive voice and resistance to sycophancy under tight compute constraints.

[Explore the code](https://github.com/orbitalronin94/WISE-RONIN) · [Browse experiments](docs/EXPERIMENTS.md) · [Open an issue](https://github.com/orbitalronin94/WISE-RONIN/issues)

</div>

---

## Overview

WISE-RONIN is built around a simple research question:

> How much useful, recognizable behavior can a small language model learn when the data, training process and evaluation are treated as first-class engineering problems?

The project combines a compact Transformer, deterministic dataset tooling, balanced training curricula and explicit holdout evaluation. It favors reproducible experiments over inflated capability claims.

| | |
|---|---|
| **Model** | Decoder-only Transformer, PyTorch |
| **Scale** | 4.27M effective parameters in the first trained checkpoint |
| **Tokenizer** | ByteLevel BPE; 1,297 effective tokens in the first checkpoint |
| **Training** | SFT, checkpoint/resume, CPU-compatible workflow |
| **Data** | Curated legacy corpus + deterministic expanded curriculum |
| **Evaluation** | Greedy and sampling; factual anchors, diversity and repetition diagnostics |
| **Automation** | GitHub Actions |

## Project status

<details open>
<summary><strong>Current experiment — balanced 80k continuation</strong></summary>

The current training cycle resumes from the 10k checkpoint and runs two epochs over a balanced 80,000-example curriculum (batch size 8, learning rate 1.5e-4).

- [Training run #44721903295](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44721903295)
- [Balanced curriculum builder](build_balanced_80k.py)
- [Experiment log](docs/EXPERIMENTS.md)

The run had passed dataset construction, category validation, checkpoint compatibility and Python compilation when this README was updated. Training was still running; evaluation and artifact upload had not yet completed.

</details>

<details>
<summary><strong>Baseline — first 10,000 SFT updates</strong></summary>

[View completed run #44709299157](https://github.com/orbitalronin94/WISE-RONIN/actions/runs/44709299157).

| Metric | Result |
|---|---:|
| Training updates | 10,000 |
| Examples after exact deduplication | 88,348 |
| Effective parameters | 4,267,520 |
| Effective tokenizer vocabulary | 1,297 |
| Final recorded loss | 0.16234 |
| Training time | 1,562 s (~26 min) |

A falling training loss confirms optimization of the training objective. It does **not** establish factual accuracy, generalization or instruction-following quality.

</details>

## Architecture

The first trained checkpoint uses a compact decoder-only Transformer:

| Component | Configuration |
|---|---:|
| Effective parameters | 4,267,520 |
| Embedding dimension | 256 |
| Decoder layers | 6 |
| Query heads / KV heads | 8 / 2 |
| Head dimension | 32 |
| SwiGLU hidden dimension | 640 |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Embeddings | Tied input/output |
| Maximum training sequence in baseline | 512 tokens |
| Architectural context setting | 1,024 tokens |

Implementation: [`train_wise_ronin.py`](train_wise_ronin.py).

## Datasets

### Expanded deterministic corpus

The generated corpus contains **147,627 rows**, with unique prompt and literal response strings. It is generated from a built-in factual bank and exact arithmetic tasks.

| Composition | Count |
|---|---:|
| Canonical factual anchors | 402 |
| Base arithmetic tasks | 9,939 |
| Independent `fact_id` groups | 10,341 |
| Total generated rows | 147,627 |

**Important:** row count is not independent knowledge count. Multiple linguistic variants share the same factual anchor. This is a synthetic curriculum, not a fully externally sourced or encyclopedically verified knowledge base.

- [Dataset](data/expanded/wise_ronin_expanded.jsonl)
- [Generation report](data/expanded/wise_ronin_expanded_report.json)
- [Dataset documentation](data/expanded/README.md)
- [Generator](generate_wise_ronin_expanded.py)

### Balanced 80k curriculum

The training builder produces an exactly balanced 80,000-row dataset:

| Category | Examples |
|---|---:|
| Factual geography | 10,000 |
| Factual chemistry | 15,000 |
| General knowledge | 15,000 |
| Addition | 10,000 |
| Subtraction | 10,000 |
| Multiplication | 10,000 |
| Division | 10,000 |
| **Total** | **80,000** |

Unused source rows are preserved in a separate reserve file. Additional geography/general prompts retain their original `fact_id`; they are new formulations, not new independent facts. The builder also adds exact division tasks.

See [`build_balanced_80k.py`](build_balanced_80k.py).

## Training

### Requirements

- Python 3.11+
- PyTorch
- Hugging Face Tokenizers

Install the core dependencies:

```bash
python -m pip install torch tokenizers
```

### Run the baseline trainer

Inspect the available options first:

```bash
python train_wise_ronin.py --help
```

The initial 10k experiment used the three legacy JSONL files, batch size 4, sequence length 512 and a peak learning rate of 3e-4. The balanced continuation is automated through GitHub Actions and resumes from the saved 10k checkpoint.

**Reproducibility note:** the current SFT data does not contain preference pairs. ORPO is therefore not enabled; preference labels must come from real chosen/rejected examples, not fabricated pairs.

## Evaluation

WISE-RONIN keeps generation behavior and evaluation diagnostics separate.

| Evaluator | Purpose |
|---|---|
| [`evaluate_wise_ronin.py`](evaluate_wise_ronin.py) | Greedy decoding on the holdout |
| [`evaluate_wise_ronin_sampling.py`](evaluate_wise_ronin_sampling.py) | Sampling, response diversity and repeated-style diagnostics |

The first greedy baseline produced a heuristic score of 35/500 (7%). This is **not** a semantic accuracy score: the 500-row holdout contains only 55 distinct questions, and the evaluator uses substring-style checks. The model also showed substantial response-template collapse and clear factual errors.

Current evaluation tracks several dimensions independently:

- **Factuality:** expected-answer matching; literal matching is only a proxy and needs semantic review.
- **Diversity:** unique normalized responses and per-question variation.
- **Repetition:** concentration of recurring openings, tails and motifs.
- **Decoding:** greedy and sampling results are reported separately.

The holdout is kept out of training and tokenizer construction. Do not interpret a single aggregate score as proof of model quality.

## Repository map

```text
WISE-RONIN/
├── train_wise_ronin.py
├── evaluate_wise_ronin.py
├── evaluate_wise_ronin_sampling.py
├── generate_wise_ronin_expanded.py
├── build_balanced_80k.py
├── audit_wise_ronin_dataset.py
├── build_wise_ronin_dataset.py
├── dataset_enano_70mb_parte*.jsonl
├── holdout_wise_ronin_500.jsonl
├── data/
│   └── expanded/
├── docs/
│   ├── REPOSITORY_GUIDE.md
│   └── EXPERIMENTS.md
└── .github/
    └── workflows/
```

Legacy data and scripts remain in their current locations because active workflows depend on those paths. See the [repository guide](docs/REPOSITORY_GUIDE.md) for file ownership and the [workflow index](.github/workflows/README.md) for automation details.

## GitHub Actions

| Workflow | Purpose |
|---|---|
| First 10k SFT | Initial training from scratch |
| Expand dataset | Deterministic corpus generation |
| Dataset audit | Dataset statistics and duplicate analysis |
| Balanced 80k | Resume, balanced training and sampling evaluation |
| Balanced 80k validation | Validate builder invariants on pull requests |

[Browse all workflows](.github/workflows/README.md) · [Open Actions](https://github.com/orbitalronin94/WISE-RONIN/actions)

## Roadmap

- [x] Implement a real PyTorch decoder-only trainer.
- [x] Add checkpointing and resume support.
- [x] Establish a 10k SFT baseline on GitHub Actions CPU runners.
- [x] Generate a deterministic expanded corpus with traceable factual anchors.
- [x] Build and validate an exactly balanced 80k curriculum.
- [ ] Complete the balanced 20k-update continuation and inspect its artifacts.
- [ ] Improve semantic factuality evaluation beyond literal substring matching.
- [ ] Expand the factual bank with independently sourced, auditable references.
- [ ] Reduce response-template collapse without erasing the model's intended voice.
- [ ] Refactor legacy root-level scripts into a stable package without breaking workflows.

## Documentation

- [Repository structure and file guide](docs/REPOSITORY_GUIDE.md)
- [Experiment history and results](docs/EXPERIMENTS.md)
- [Expanded dataset specification](data/expanded/README.md)
- [GitHub Actions workflow index](.github/workflows/README.md)

## Contributing

Issues and pull requests are welcome. Useful contributions include:

- Independently sourced factual anchors with provenance and licensing information.
- Better semantic and adversarial holdout evaluation.
- Reproducible CPU performance improvements.
- Dataset audits that distinguish surface variation from independent knowledge.
- Clear bug reports with run links, logs and reproduction steps.

Please avoid claims that are not backed by a reproducible run or a documented evaluation.

## License

See [LICENSE](LICENSE). If no license file is present in the repository, all rights remain reserved by the respective rights holders.

---

<div align="center">

**Small models. Long methods. No borrowed certainty.**

If this experiment is useful, consider giving it a ⭐ on GitHub.

[![Star this repository](https://img.shields.io/badge/Star%20this%20repo-GitHub-181717?style=for-the-badge&logo=github)](https://github.com/orbitalronin94/WISE-RONIN)

</div>
