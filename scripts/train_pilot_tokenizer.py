#!/usr/bin/env python3
"""Train a provisional 4k ByteLevel BPE tokenizer for the 19k-row pilot."""
from __future__ import annotations
import argparse, hashlib, json
from collections import Counter
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors

SPECIAL = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/pilot_train_19k.jsonl")
    p.add_argument("--manifest", default="data/pilot_train_19k.manifest.json")
    p.add_argument("--output", default="data/pilot_tokenizer/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=4096)
    p.add_argument("--pilot-authorization", action="store_true")
    args = p.parse_args()
    if not args.pilot_authorization:
        p.error("--pilot-authorization is required")
    if args.vocab_size != 4096:
        raise ValueError("The V1 pilot uses exactly 4,096 BPE tokens")
    data, manifest_path = Path(args.data), Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "provisional_private_pilot_only" or manifest.get("training_rows") != 19000:
        raise ValueError("Expected the owner-authorized 19k provisional pilot manifest")
    if sha256(data) != manifest.get("training_sha256"):
        raise ValueError("Pilot corpus SHA-256 mismatch")

    texts, languages = [], Counter()
    with data.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = (str(row.get("prompt", "")) + "\n" + str(row.get("completion", ""))).strip()
            if not text:
                raise ValueError(f"Empty text at row {line_no}")
            texts.append(text)
            languages[str(row.get("language", "und"))] += 1
    if len(texts) != 19000:
        raise ValueError(f"Expected 19,000 rows, got {len(texts)}")

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size, min_frequency=2,
        special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tok.train_from_iterator(texts, trainer=trainer)
    if tok.get_vocab_size() != args.vocab_size:
        raise RuntimeError(f"Expected 4,096 tokens, got {tok.get_vocab_size()}")
    tok.post_processor = processors.TemplateProcessing(
        single="[BOS] $A [EOS]", pair="[BOS] $A [SEP] $B:1 [EOS]",
        special_tokens=[
            ("[BOS]", tok.token_to_id("[BOS]")),
            ("[EOS]", tok.token_to_id("[EOS]")),
            ("[SEP]", tok.token_to_id("[SEP]")),
        ],
    )
    expected = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[SEP]": 4}
    actual = {name: tok.token_to_id(name) for name in SPECIAL}
    if actual != expected:
        raise RuntimeError(f"Special token IDs differ: {actual}")
    for sample in texts[:100]:
        encoded = tok.encode(sample, add_special_tokens=False)
        if tok.decode(encoded.ids) != sample:
            raise RuntimeError("Byte-level encode/decode round trip failed")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out))
    result = {
        "status": "provisional_pilot_tokenizer",
        "corpus_sha256": manifest["training_sha256"],
        "corpus_rows": len(texts),
        "vocab_size": tok.get_vocab_size(),
        "special_token_ids": actual,
        "language_counts_heuristic": dict(languages),
        "tokenizer_sha256": sha256(out),
        "round_trip_samples": 100,
        "production_tokenizer": False,
        "production_training_authorized": False,
    }
    out.with_name("tokenizer.manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
