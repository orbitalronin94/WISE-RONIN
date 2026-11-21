# WISE-RONIN: gate before resuming from step 30,000

## Current decision

The step-30,000 checkpoint is frozen as an experimental specimen. No training
workflow is part of this preparation phase. The correction builder does not
load model weights, and the validation workflow has no training step.

## Diagnosis translated into data rules

- Factual content is the target; style is optional and subordinate.
- Canonical facts are split before variants are produced. A fact group cannot
  appear in both correction SFT and the new factual holdout.
- Arithmetic training and evaluation use disjoint operand ranges.
- Anti-sycophancy and false-premise examples are separate categories. They do
  not receive arbitrary numeric targets.
- ORPO contrasts use the same prompt and stylistic wrapper, changing the
  factual answer. The rejected answer is not the old generic fallback.
- The SFT builder uses plain factual answers as well as a restrained dry style;
  it does not force KOAN or a decorative object into every answer.

## Files produced by the builder

- \`correction_sft_train.jsonl\`: factual, arithmetic, anti-sycophancy SFT.
- \`correction_orpo_train.jsonl\`: factual chosen/rejected preference pairs.
- \`correction_holdout.jsonl\`: unseen canonical facts, disjoint arithmetic
  questions, false-premise and anti-sycophancy probes.
- \`correction_report.json\`: counts and split diagnostics.

## Gates before any new training

1. The builder and structural validation pass.
2. Manually inspect the complete correction pack, especially all rejected
   preference answers and ambiguous factual anchors.
3. Inspect the 30k checkpoint tokenizer, EOS behavior, generation limits and
   truncation separately; do not infer BPE corruption from malformed strings.
4. Run a baseline evaluation of the frozen 30k checkpoint against the new
   holdout, with fixed decoding settings and saved raw generations.
5. Establish independent scores for factual accuracy, arithmetic accuracy,
   false-premise resistance, anti-sycophancy, response diversity, motif
   repetition, malformed text, EOS completion and truncation.
6. Only then compare a short, reversible continuation from step 30k against
   the frozen baseline. Do not schedule a long run or ORPO until the SFT
   correction behaves as intended.

## Important limitations

The current curated source contains only 30 canonical factual questions.
The group-level split is honest but small; arithmetic provides broader
algorithmic coverage, not broad world knowledge. This pack is a corrective
experiment, not a general knowledge corpus. No threshold should be treated as
passed until the checkpoint is actually evaluated on the holdout.
