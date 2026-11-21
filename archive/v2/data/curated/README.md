# WISE-RONIN curated training data

Generated from the three original `dataset_enano_70mb_parte*.jsonl` shards by
`build_wise_ronin_dataset.py`. The source shards and the 500-row manual holdout
are not modified or used as output targets.

## Audit finding

The source has 147,627 physical rows, but those rows do **not** represent
147,627 independent examples. After exact-row deduplication the previous audit
found 88,348 rows; the curated builder identifies only **30 canonical factual
questions** with consistent answers in the KOAN targets and at least ten
independent fact-rodeo corroborations per answer. Many thousands of source rows
are permutations of a small number of prompts and response templates.

That is the central reason the original training run collapsed into fallback
phrases. The 1% cap on exact normalized assistant strings was insufficient:
near-identical templates varied object names, locations, and trailing slogans.

## Files

| File | Rows | Purpose |
|---|---:|---|
| `wise_ronin_factual_clean.jsonl` | 30 | One concise, cross-checked fact per canonical question |
| `wise_ronin_hybrid_controlled.jsonl` | 147 | Up to five fact-preserving style families per fact |
| `wise_ronin_koan_controlled.jsonl` | 30 | One KOAN example per fact; echoed question and articles normalized |
| `wise_ronin_antisycophancy.jsonl` | 20 | Distinct anti-sycophancy examples; repeated copies removed |
| `wise_ronin_sft_balanced.jsonl` | 257 | Combined SFT pool: 60 factual anchors + 147 hybrid + 30 KOAN + 20 anti-sycophancy |
| `wise_ronin_preferences.jsonl` | 30 | ORPO pairs: verified concise fact chosen over generic `El que contesta.` fallback |
| `wise_ronin_dataset_report.json` | — | Fact votes, corroboration counts, template counts, and limitations |

## Recommended use

1. **Factual calibration:** use `wise_ronin_factual_clean.jsonl` for a short,
   separately measured factual phase.
2. **Style conditioning with factual replay:** use
   `wise_ronin_sft_balanced.jsonl`; the factual anchors remain present so style
   examples do not completely replace factual supervision.
3. **Preference correction:** use `wise_ronin_preferences.jsonl` only in an
   explicitly configured ORPO phase. It teaches the model to prefer the verified
   answer over the generic fallback for these factual prompts.

Do not train for thousands of steps on this compact set without measuring
memorization. It is a controlled correction set, **not** a broad knowledge
corpus. The 30 facts are a deliberately narrow coverage limitation and must be
expanded with independently curated factual questions before claiming general
factual competence.

## Validation

The preparation workflow verifies row counts, JSONL shape, answer consistency
between factual/hybrid/KOAN targets, preference-pair alignment, and that the
manual holdout remains separate. It does not launch training or alter the
10k checkpoint.
