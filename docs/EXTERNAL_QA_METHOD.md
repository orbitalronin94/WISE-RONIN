# External QA authoring protocol

WISE-RONIN's external QA workstream uses a verbatim-question / manual-answer protocol.

## Why this design

The model's existing factual set is narrow. External corpora can broaden the question distribution, but automatically rewriting thousands of answers would create synthetic paraphrases and obscure whether the voice is genuinely being authored. This pipeline therefore separates source ingestion from answer authorship.

## Source selection

- **SQAC**: Spanish extractive QA, 15,036 train questions, pinned source revision. Context and answer spans are retained so each rewrite can be checked against its evidence.
- **ESQAD EVAU**: manually curated educational questions under CC BY 4.0. Planned as a separate import after the SQAC workflow is established.
- **SIQA-es / OpenBookQA-es**: not used for training in this workstream because the available splits are evaluation-only or validation/test in the cited releases.

## Authorship and review

A human author reads the original question, context and reference answer. The question is copied unchanged. The new answer may change syntax, word order and register, but it may not add unsupported facts, remove qualifiers, or change the logical relation. The author records a review status and may add a short note explaining a non-obvious equivalence.

The importer never generates answers. The validator does not claim to prove semantic equivalence; it enforces provenance, split isolation, offsets, schema, uniqueness and that the manual answer is not merely copied from the reference. Semantic review remains human.

## Training isolation

Only source \`train\` rows can be exported. A row is excluded until marked \`manually_reviewed\`. Validation and test splits are not downloaded by the workflow and cannot pass the exporter. This is intentional: using benchmark questions for training would contaminate future evaluation.

## Current limitation

299 manually authored examples are committed as the reviewed SQAC set to date. The 15,036-question SQAC queue is generated as an Actions artifact for controlled authoring. The corpus is not complete until each admitted row has a reviewed manual answer. No training workflow is triggered by this dataset workflow.
