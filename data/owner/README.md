# Owner-supplied data required before corpus freeze

The final dataset builder deliberately stops until both files below are supplied:

- `curated_5000.jsonl`: exactly 5,000 original, curated records.
- `python_edu_licensed.jsonl`: exactly 1,000 code records, each tied to an immutable source file and individually audited license.

No synthetic filler is generated to satisfy either quota. Do not copy the historical holdout into these files.

## Owner-curated record

Each line must be a JSON object with:

- `prompt`: non-empty input/context.
- `completion`: non-empty target text.
- `origin`: exactly `owner_curated`.
- `license`: the actual license or explicit owner-authorized terms.
- `rights_confirmed`: boolean `true`, set only after confirming the record may be used for training and included in the repository's artifact.
- `language`: optional ISO language code (for example `es`, `en`); omitted values receive a rough heuristic label during audit.
- `type`: optional task category.

## Individually audited code record

Each line must include non-empty `prompt` and `completion`, plus all fields below:

- `source_repo`: canonical repository owner/name.
- `source_path`: exact path of the original source file.
- `source_commit`: immutable Git commit SHA (not a branch name).
- `source_hash`: SHA-256 of the original source file.
- `license`: license identified for that exact file at that commit.
- `license_url`: canonical license text or repository license evidence URL.
- `license_file_path`: path of the license/notice evidence in the source repository.
- `license_file_sha256`: SHA-256 of the license/notice evidence file.
- `attribution`: attribution text that must accompany redistribution.
- `license_audited`: boolean `true`, only after a human has checked the file-level license.
- `rights_confirmed`: boolean `true`, only after confirming that training and redistribution are permitted.
- `type`: normally `codigo`.
- `language`: optional ISO language code.

A dataset-level license label is not sufficient for code originating in The Stack. A row is rejected if the per-file evidence or explicit audit flags are missing. A boolean flag is an attestation, not automatic legal verification.

## Language balance

The current external sources are predominantly English. The builder reports a heuristic language distribution and Spanish share, but does not invent a target threshold. Once the intended Spanish proportion is agreed, run the builder with `--min-spanish-share` to enforce it. For reliable production counts, manually label ambiguous/multilingual records rather than treating the heuristic as ground truth.

## Build gates

From the repository root, after both files are present:

```bash
python -m data.build_dataset \
  --sources artifacts/lumen-source-candidates \
  --owner data/owner \
  --holdout data/holdout_500.jsonl \
  --output data/train_25k.jsonl
```

The builder verifies source hashes/revisions, exact quotas, per-file provenance, duplicate text, repeated five-token prefixes, 500-row holdout integrity, exact overlap and long prompt substrings. It writes `train_25k.manifest.json` with row counts, source hashes, language estimates and the output SHA-256.

The lexical overlap audit is conservative but not semantic proof. The final manifest must be reviewed manually before tokenizer training. Tokenizer training and model training remain separate operations.
