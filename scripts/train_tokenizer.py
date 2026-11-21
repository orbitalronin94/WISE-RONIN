#!/usr/bin/env python3
"""Train the frozen 4k ByteLevel BPE tokenizer from an approved final corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
REQUIRED_GATES = (
    "source_license_review_complete",
    "owner_rights_review_complete",
    "code_file_license_review_complete",
    "holdout_manual_review_complete",
    "language_balance_approved",
    "tokenizer_approved",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train_25k.jsonl")
    parser.add_argument("--manifest", default="data/train_25k.manifest.json")
    parser.add_argument("--output", default="data/tokenizer/tokenizer.json")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--expected-rows", type=int, default=25000)
    args = parser.parse_args()

    if args.vocab_size != 4096:
        raise ValueError("V1 tokenizer vocabulary is frozen at exactly 4,096 tokens")

    data_path, manifest_path = Path(args.data), Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("rows") != args.expected_rows:
        raise ValueError(f"Manifest row count must be {args.expected_rows}")
    if sha256(data_path) != manifest.get("sha256"):
        raise ValueError("Training JSONL SHA-256 does not match the reviewed manifest")
    gates = manifest.get("review_gates", {})
    blocked = [name for name in REQUIRED_GATES if gates.get(name) is not True]
    if blocked:
        raise RuntimeError(f"Tokenizer build blocked; review gates are not approved: {blocked}")

    texts = []
    language_counts = Counter()
    with data_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt, completion = str(row.get("prompt", "")), str(row.get("completion", ""))
            text = (prompt + "\n" + completion).strip()
            if not text:
                raise ValueError(f"Empty training text at line {line_no}")
            texts.append(text)
            language_counts[str(row.get("language", "und"))] += 1
    if len(texts) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} non-empty rows, got {len(texts)}")

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    if tok.get_vocab_size() != args.vocab_size:
        raise RuntimeError(f"Expected exactly {args.vocab_size} tokens, got {tok.get_vocab_size()}")

    tok.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [SEP] $B:1 [EOS]",
        special_tokens=[
            ("[BOS]", tok.token_to_id("[BOS]")),
            ("[EOS]", tok.token_to_id("[EOS]")),
            ("[SEP]", tok.token_to_id("[SEP]")),
        ],
    )
    expected_ids = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[SEP]": 4}
    actual_ids = {name: tok.token_to_id(name) for name in SPECIAL}
    if actual_ids != expected_ids:
        raise RuntimeError(f"Special token IDs differ from V1 contract: {actual_ids}")

    # Byte-level encode/decode round-trip catches malformed tokenizer configs.
    for sample in texts[: min(100, len(texts))]:
        encoded = tok.encode(sample, add_special_tokens=False)
        if tok.decode(encoded.ids) != sample:
            raise RuntimeError("Tokenizer failed byte-level round-trip validation")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    tokenizer_manifest = {
        "status": "tokenizer_built_after_review_gates",
        "corpus_path": str(data_path),
        "corpus_sha256": manifest["sha256"],
        "corpus_rows": len(texts),
        "vocab_size": tok.get_vocab_size(),
        "special_token_ids": actual_ids,
        "language_counts_from_corpus": dict(language_counts),
        "tokenizer_path": str(out),
        "tokenizer_sha256": sha256(out),
        "round_trip_samples": min(100, len(texts)),
        "training_authorized": False,
    }
    out.with_name("tokenizer.manifest.json").write_text(
        json.dumps(tokenizer_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(tokenizer_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
