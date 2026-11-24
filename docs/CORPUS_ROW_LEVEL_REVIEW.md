# Row-level corpus review index

The build_corpus_review_queue.py utility creates a metadata-only review queue from a pinned JSONL export.

## Privacy and field boundaries
- The caller must explicitly allowlist prose fields with --text-fields.
- IDs, UUIDs, dates, labels and arbitrary metadata are not scanned as prose unless incorrectly allowlisted.
- Chat-style datasets can pass --messages-field messages; only each message's role/content pair is hashed and checked.
- Token-classification datasets can pass --tokens-field tokens and --tags-field ner_tags; the index records token/tag counts, alignment and tag histogram without exporting token text or annotation arrays.
- Email, NIF and phone patterns are checked only inside allowlisted prose and chat-message content.
- Output contains source/revision, split, original row index, field names, text SHA-256, character and whitespace-token counts, language heuristic, task type and pending review gates.
- Raw text and excerpts are never written to the index. Keep source exports in a controlled ephemeral review environment.
- Regex matches are flags for human inspection, not proof of personal data and not automatic rejection.

## Example: flat QA
\`\`\`bash
python scripts/build_corpus_review_queue.py \\
  --input artifacts/pinned-train-export.jsonl \\
  --output artifacts/review-index.jsonl \\
  --source somosnlp/instruct-legal-refugiados-es \\
  --revision IMMUTABLE_COMMIT_SHA \\
  --split train \\
  --text-fields question,context,answer \\
  --base-jsonl artifacts/spanish-v3-1-candidate.jsonl
\`\`\`

## Example: chat SFT
\`\`\`bash
python scripts/build_corpus_review_queue.py \\
  --input artifacts/pinned-train-export.jsonl \\
  --output artifacts/review-index.jsonl \\
  --source OysterCoreAI/SFT-General-Spanish-50K \\
  --revision IMMUTABLE_COMMIT_SHA \\
  --split train \\
  --text-fields metadata \\
  --messages-field messages
\`\`\`

The SFT source uses a messages array plus metadata; keep the role sequence and system message intact. Metadata is not a prose field and should not be included as text. The tool hashes role-labelled message content without exporting it.

## Candidate source mappings (verify against the exact pinned revision)
| Dataset | Candidate content mapping | Special handling |
|---|---|---|
| OysterCoreAI/SFT-General-Spanish-50K | messages[].role/content | Preserve chat roles and system message; investigate near-duplicate candidates |
| Kukedlc/Personas_Spanish-40k | Persona_Descripcion or descripcion | Confirm exact case/name in pinned CSV schema; unresolved language rows quarantined |
| StrataSynth/stratasynth-personas-spain | persona, daily_routine, speaking_style | Exclude demographics and personality/psychometric fields pending explicit privacy/quality review |
| somosnlp/instruct-legal-refugiados-es | question, context, answer | Preserve context-grounded QA; do not duplicate prompt/prompt_es variants |
| dariolopez/ms-marco-es-500k | query, passage | Retrieval-only; never route to generative pretraining |
| BSC-LT/m-personas | unresolved | Adapter blocked pending schema discovery |
| PlanTL-GOB-ES/pharmaconer | tokens, ner_tags (viewer) or train-set_1.1.conll | Preserve document IDs, offsets, IOB tags and train-only split |
| bigbio/distemist | tokens, ner_tags or original span files | Preserve entity offsets/concept links and train-only split |

Candidate field names are not schema claims. Inspect each pinned source revision and use its actual schema. All admission statuses remain pending; human rights, privacy and quality review is required before normalization and 100 MiB block construction.
