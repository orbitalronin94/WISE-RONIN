# v3.2 source scan — run 45359758217

**Status: automated discovery scan completed; no source is approved and no rows were admitted.**
Artifact: https://github.com/orbitalronin94/WISE-RONIN/actions/runs/45359758217/artifacts/14905097246  
Frozen base indexed: **197,788 v3.1 rows**. The report's token counts are whitespace-token estimates over extracted text fields, not tokenizer IDs.

## Results

| Source | Rows scanned | Spanish / unresolved | Whitespace tokens | Exact duplicates | Near-duplicate candidates | Other flags / result |
|---|---:|---:|---:|---:|---:|---|
| OysterCoreAI/SFT-General-Spanish-50K | 48,397 | 48,397 Spanish heuristic | 10,952,543 | 0 reported | 13,489 | 155 cross-source near-duplicate candidates |
| BSC-LT/m-personas | 0 | Not resolved | — | — | — | Hub config discovery returned no configs; requires adapter/schema investigation |
| Kukedlc/Personas_Spanish-40k | 37,476 | 35,999 Spanish; 1,477 unresolved | 766,954 | 1 | 1 | 1 possible-phone flag |
| StrataSynth/stratasynth-personas-spain | 1,250 | 1,250 Spanish declared | 611,003 | 0 | 0 | Needs field-level filtering; no psychometric conditioning accepted automatically |
| somosnlp/instruct-legal-refugiados-es | 9,430 | 9,430 Spanish declared | 9,336,241 | 140 | 244 | 66 cross-source near-duplicate candidates; 2,524 possible-phone and 34 possible-email flags |
| dariolopez/ms-marco-es-500k | 500,000 | 499,829 Spanish heuristic; 166 unresolved; 5 non-Spanish | 70,011,245 | 34 | 505 | 24,778 possible-phone, 814 possible-email, 1 possible-NIF flags; retrieval-only |
| PlanTL-GOB-ES/pharmaconer | 0 | Not scanned | — | — | — | Loader failed: legacy dataset script unsupported by datasets v3+ |
| bigbio/distemist | 0 | Not scanned | — | — | — | Loader failed: legacy dataset script unsupported by datasets v3+ |

## Interpretation and actions

1. **No 100 MiB block can be built from this report.** The scan records counts, flags, samples and duplicate references; it does not export a complete, row-level normalized review queue.
2. **MS MARCO remains retrieval-only.** Its 70.0M whitespace tokens are not a generative-pretraining addition. The high phone/email counts are heuristic flags, not proof of personal data: numeric IDs, dates and structured triples can trigger them. Inspect the actual fields and stratified samples before any decision.
3. **Legal/refugee QA requires targeted review.** Repeated rows (140 exact and 244 near candidates) should be deduplicated. Phone/email regex flags may include legal references and identifiers; no automatic deletion based on regex alone.
4. **SFT-General Spanish is promising but highly templated.** 13,489 near-duplicate candidates (about 27.9% of scanned rows) require a stricter content-aware dedup pass. The SimHash result is a candidate list, not a final duplicate verdict.
5. **mPersonas is blocked**, not empty or rejected: dataset config discovery returned no configs at the pinned revision. Inspect repository files and builder format before retrying.
6. **PharmaCoNER and DisTEMIST are blocked by loader compatibility**, not by data quality or licensing. Build explicit adapters from their source files; preserve train-only boundaries and annotation labels.
7. **Kukedlc unresolved language rows remain quarantined.** Its one possible-phone flag needs manual inspection.
8. **StrataSynth needs schema-level exclusion of psychometric/sensitive conditioning fields** before any profile can enter a review queue.

## Next engineering gate

The audit must be upgraded to export a full row-level review queue with immutable source revision, original row index, source fields, task type, text hash, duplicate references, language result and privacy/quality flags. Then add adapters for mPersonas, PharmaCoNER and DisTEMIST, plus field-aware PII checks. Human reviewers can assign `review_status`, `rights_status`, `privacy_status` and `quality_status`; only then may the 100 MiB version-chain builder emit `v3.2.001`.

**Training authorization: false.**