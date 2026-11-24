# WISE-RONIN corpus v3.2 — batch admission ledger

This ledger records the immutable source-specific batch admissions appended to frozen v3.1. It is not a training authorization.

## Current cumulative chain

| Version | Parent | Source | Rows | Delta bytes | Whitespace tokens |
|---|---|---|---:|---:|---:|
| v3.2.001 | v3.1 | BSC-LT/m-personas | 32,040 | 104,856,354 | 11,509,938 |
| v3.2.002 | v3.2.001 | BSC-LT/m-personas | 3,447 | 12,598,425 | 1,425,279 |
| v3.2.003 | v3.2.002 | Kukedlc/Personas_Spanish-40k | 35,997 | 29,022,331 | 851,920 |
| v3.2.004 | v3.2.003 | somosnlp/instruct-legal-refugiados-es | 9,091 | 35,885,349 | 4,239,968 |
| **Total** | **v3.1** | **3 sources** | **80,575** | **182,362,459** | **18,027,105** |

All three deltas are immutable, individually hashed, and below the 100 MiB ceiling. The chain summary and each version manifest contain the parent link, source revision, row count, byte count, token estimate and SHA-256. A whitespace-token count is not a tokenizer-ID count.

## Source 1 — BSC-LT/m-personas

- Revision: \`6fab205a86a9244c90a2782e28055027f45ddd03\`
- Declared license at pinned revision: Apache-2.0
- Split: Spanish (\`es\`)
- Rows scanned: 36,094
- Rows admitted: 35,487
- Quarantined: 607 (598 possible-phone flags; 9 possible-email flags)
- Admitted whitespace tokens: 12,935,217
- \`source_text\` is deliberately excluded because it contains original FineWeb2 seed text.
- The admitted text preserves persona, prompt and alternating user/assistant turns with explicit role labels.
- Source card describes LLM-generated persona and conversation synthesis. Treat as synthetic conversational coverage, not factual ground truth.

## Source 2 — Kukedlc/Personas_Spanish-40k

- Revision: \`2f78060ecab15d65ba8890f4f642fd0b18260405\`
- Declared license at pinned revision: Apache-2.0
- Split: \`train\`
- Rows scanned: 37,476
- Rows admitted: 35,997
- Quarantined: 1,479 (1,477 language-unresolved; 1 possible-phone flag; 1 exact duplicate)
- Admitted whitespace tokens: 851,920
- Descriptions are a narrow persona slice with a notable arts/culture/theatre emphasis. Do not treat them as a representative population distribution.

## Batch admission policy

\`review_status=auto_accept\` and \`review_method=batch_policy_v1\` mean that the record passed the versioned source-specific automated policy. They do not mean a human inspected every row.

Every admitted row must carry:
- Immutable source revision and source row provenance.
- An allowlisted license verified at that pinned revision.
- Explicit task type and Spanish language classification.
- Privacy scan clear under the source-specific policy.
- Structural and minimum-quality checks passed.
- A unique normalized text hash within the batch; the cumulative builder performs exact dedup against the inherited corpus hash database.

Flagged, unresolved, malformed and duplicate rows are quarantined. No data source is accepted solely because its dataset card lists a permissive license.

## Next candidate — contextual refugee-law QA

\`somosnlp/instruct-legal-refugiados-es\` is being processed separately:
- Pinned revision: \`c5e0373049d50c382a75ee0677e3f6b1b4b63dc2\`
- Declared license: Apache-2.0
- Train rows scanned in the prior audit: 9,430.
- The policy quarantines all 244 near-duplicate candidate row indices from the pinned audit, including exact-duplicate rows and cross-source near-duplicate candidates.
- Email, NIF and formatted/contextual phone checks are field-aware; bare legal references such as decree and law numbers are not automatically treated as telephone numbers.
- Question, context and generated answer remain explicitly labelled. Source jurisdiction metadata is preserved.
- Answers are machine-generated; legal validity is marked \`not_independently_verified\`. This material is not a current legal reference service.

## Source 3 — somosnlp/instruct-legal-refugiados-es

- Revision: `c5e0373049d50c382a75ee0677e3f6b1b4b63dc2`
- Declared license at pinned revision: Apache-2.0
- Split: `train`
- Rows scanned: 9,430
- Rows admitted: 9,091
- Quarantined: 339 (244 near-duplicate candidates, 33 possible-email flags, 26 formatted/contextual phone flags, 19 short contexts, 17 low-text-density rows)
- Admitted whitespace tokens: 4,239,968
- Question, context and generated answer are preserved as labelled segments. Source country, period, domain, task and register are retained as metadata.
- All 244 near-duplicate candidate row indices from the pinned audit were quarantined; this includes exact duplicates and cross-source near-duplicate candidates.
- The phone detector is field-aware and requires phone-like grouping or explicit phone context, avoiding the broad numeric false positives from statutory citations.
- Answers were machine-generated from legal source contexts. `legal_validity=not_independently_verified`; this is training material, not current legal advice.

## Training gate

All manifests retain \`training_authorized: false\`. No training, tokenizer update, holdout change or promotion to main is triggered by corpus admission. Before training, rebuild tokenizer-ID counts, inspect cumulative source balance, and run frozen holdout comparisons against the unchanged v3.1 baseline.
