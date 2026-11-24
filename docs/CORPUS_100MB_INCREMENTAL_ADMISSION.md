# Incremental 100 MiB corpus admission

This is the bounded promotion stage for WISE-RONIN/LUMEN corpus growth.

## Contract

- Target block ceiling: **100 MiB (104,857,600 bytes)**. The final block may be smaller.
- Inputs are normalized JSONL records with `text`, `source`, `source_revision`,
  `license`, `rights_status`, and `review_status`.
- A row is admitted only when `review_status=accept`, `rights_status=verified`,
  and the license is explicitly allowlisted.
- Exact normalized-text duplicates are rejected across the entire input stream.
- Possible email, Spanish NIF, and phone patterns are quarantined for review.
- A record larger than the configured block ceiling is rejected, never split.
- Each block receives a SHA-256 and source/revision row counts in the manifest.
- The builder writes data artifacts only; it does not modify the frozen v3.1/v3.2
  corpus, merge branches, or authorize training.

## Sequential loop

1. Audit and pin one source revision.
2. Normalize only its approved train records into the required JSONL schema.
3. Human-review samples and resolve rights/privacy/quality flags.
4. Run `scripts/build_corpus_100mb_blocks.py` against the reviewed stream.
5. Inspect the manifest and every rejection category.
6. Promote only complete, verified blocks into the next cumulative corpus version.
7. Re-run deduplication against the cumulative corpus before accepting the next source.

Do not count source archive size as accepted corpus size. Report both compressed download
size and normalized UTF-8 block bytes, plus rows and tokenizer token counts.

## Example

```bash
python scripts/build_corpus_100mb_blocks.py \
  --input artifacts/reviewed-normalized.jsonl \
  --output-dir artifacts/corpus-blocks \
  --block-prefix lumen-v3-2
```

The generated manifest always has `training_authorized: false`. Training requires a
separate explicit promotion decision after corpus-wide validation and frozen holdout
checks.
