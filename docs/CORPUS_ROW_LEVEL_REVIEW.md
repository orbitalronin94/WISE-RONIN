# Row-level corpus review index

The build_corpus_review_queue.py utility creates a metadata-only review queue from a pinned JSONL export.

## Privacy and field boundaries
- The caller must explicitly allowlist prose fields with --text-fields.
- IDs, UUIDs, dates, labels and arbitrary metadata are not scanned as prose unless incorrectly allowlisted.
- Email, NIF and phone patterns are checked only inside allowlisted text fields.
- Output contains source/revision, split, original row index, field names, text SHA-256, character and whitespace-token counts, language heuristic, task type and pending review gates.
- Raw text and excerpts are never written to the index. Keep source exports in a controlled ephemeral review environment.
- Regex matches are flags for human inspection, not proof of personal data and not automatic rejection.

## Example
```bash
python scripts/build_corpus_review_queue.py \
  --input artifacts/pinned-train-export.jsonl \
  --output artifacts/review-index.jsonl \
  --source somosnlp/instruct-legal-refugiados-es \
  --revision IMMUTABLE_COMMIT_SHA \
  --split train \
  --text-fields question,answer \
  --base-jsonl artifacts/spanish-v3-1-candidate.jsonl
```

Source adapters must preserve row order and pin immutable revisions. All admission statuses remain pending; human rights, privacy and quality review is required before normalization and 100 MiB block construction.

## Candidate field allowlists (verify against each pinned schema)
| Dataset | Candidate prose fields | Special handling |
|---|---|---|
| OysterCoreAI/SFT-General-Spanish-50K | instruction,input,output | Near-duplicate review; retain instruction/response structure |
| Kukedlc/Personas_Spanish-40k | text | Quarantine unresolved language; inspect flagged records |
| StrataSynth/stratasynth-personas-spain | text | Exclude psychometric/sensitive conditioning fields |
| somosnlp/instruct-legal-refugiados-es | question,answer | Legal jurisdiction/date and PII field review |
| dariolopez/ms-marco-es-500k | query,passage | Retrieval-only; never route to generative pretraining |
| BSC-LT/m-personas | unresolved | Adapter blocked pending schema discovery |
| PlanTL-GOB-ES/pharmaconer | unresolved | Preserve NER spans and train-only split |
| bigbio/distemist | unresolved | Preserve NER spans and train-only split |

Candidate field names are not schema claims. Inspect each pinned source revision and use its actual schema.
