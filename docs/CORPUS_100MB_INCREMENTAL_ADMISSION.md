# Incremental 100 MiB corpus admission

Each accepted increment creates a **new cumulative corpus version**:

- `v3.2.001`: frozen v3.1 plus delta 001.
- `v3.2.002`: v3.2.001 plus delta 002.
- `v3.2.003`: v3.2.002 plus delta 003.
- Continue sequentially; never reuse or skip a version number.

The data is stored efficiently as immutable delta JSONL files. Each version manifest
names its parent version, delta filename, byte count, row count, whitespace-token
count, source/revision counts and SHA-256. A version is cumulative by following its
parent chain; old deltas are never rewritten.

## Admission contract

- Hard ceiling: **100 MiB (104,857,600 bytes) per delta**. A smaller final delta is valid.
- Input is normalized JSONL with `text`, `source`, `source_revision`, `license`,
  `rights_status`, `privacy_status`, `quality_status`, and `review_status`.
- A row is accepted only with human `review_status=accept`, `rights_status=verified`,
  `privacy_status=clear`, `quality_status=approved`, and an allowlisted license.
- Exact normalized-text deduplication is seeded from frozen v3.1 and persists in SQLite
  across subsequent version builds. Near-duplicate candidates still require semantic review.
- A record larger than 100 MiB is rejected, never silently split.
- Every delta and manifest receives a SHA-256. The chain summary records all versions.
- The script does not download sources, promote into main, or authorize model training.

## Repeatable loop

1. Pin a source to an immutable commit and independently verify its license/upstream rights.
2. Stream only the permitted train split; preserve task structure (chat, documents,
   retrieval, NER) rather than flattening incompatible tasks.
3. Normalize into the input schema, retaining source/revision and source-specific metadata.
4. Human-review samples and resolve duplicate, privacy, rights and quality flags.
5. Run the builder against the reviewed stream and frozen v3.1 JSONL.
6. Inspect every version manifest, delta, rejection count and source attribution.
7. Promote only verified versions, in order, into the cumulative corpus catalogue.
8. Re-run tokenizer counts and frozen holdout checks before any training authorization.

Do not confuse archive download size with admitted corpus size. Report archive bytes,
normalized UTF-8 bytes, rows, whitespace tokens and tokenizer token IDs separately.

## Example

```bash
python scripts/build_corpus_100mb_blocks.py \
  --input artifacts/reviewed-normalized.jsonl \
  --base-jsonl artifacts/spanish-v3-1-candidate.jsonl \
  --output-dir artifacts/corpus-v3-2-versions \
  --parent-version v3.1 \
  --start-version 1
```

The script emits `lumen-corpus-v3.2.NNN-delta.jsonl`, one manifest per version,
and `lumen-corpus-v3-2-version-chain.json`. Every manifest has
`training_authorized: false`. Training remains a separate explicit decision after
corpus-wide validation and matched holdout checks.
