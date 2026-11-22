#!/usr/bin/env python3
"""Create a deterministic, source-stratified working sample from frozen corpus v2.

The full v2 artifact contains 9.36M rows; the 1k-step CPU continuation draws
8k examples. A fixed 200k-row reservoir keeps the pilot diverse and tractable.
"""
from __future__ import annotations
import argparse, hashlib, json, random
from pathlib import Path

QUOTAS = {"base": 20_000, "csic_es": 72_000, "tdx_es": 108_000}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    source = Path(a.input)
    expected = manifest["sha256"]["train.jsonl"]
    actual = sha256(source)
    if actual != expected:
        raise SystemExit(f"v2 train SHA-256 mismatch: {actual} != {expected}")
    if manifest.get("status") != "FROZEN_BUILD_OUTPUT":
        raise SystemExit("Corpus v2 is not a frozen build output")
    if manifest.get("tokenizer_sha256") != "d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe":
        raise SystemExit("Unexpected tokenizer: this continuation requires the exact v1 tokenizer")
    rng = random.Random(a.seed)
    reservoirs = {name: [] for name in QUOTAS}
    seen = {name: 0 for name in QUOTAS}
    with source.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            bucket = row.get("source") if row.get("source") in ("csic_es", "tdx_es") else "base"
            seen[bucket] += 1
            q = QUOTAS[bucket]
            item = line.rstrip("\n")
            reservoir = reservoirs[bucket]
            if len(reservoir) < q:
                reservoir.append(item)
            else:
                j = rng.randrange(seen[bucket])
                if j < q:
                    reservoir[j] = item
    if sum(seen.values()) != int(manifest["train_rows"]):
        raise SystemExit(f"Manifest row count mismatch: {sum(seen.values())} != {manifest['train_rows']}")
    for name, q in QUOTAS.items():
        if len(reservoirs[name]) != q:
            raise SystemExit(f"Insufficient rows in {name}: {len(reservoirs[name])}/{q}; seen={seen[name]}")
    selected = sum((reservoirs[name] for name in QUOTAS), [])
    rng.shuffle(selected)
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for line in selected:
            f.write(line + "\n")
    counts = {name: len(reservoirs[name]) for name in QUOTAS}
    print(json.dumps({
        "corpus": "WISE-RONIN Spanish v2",
        "full_train_rows": manifest["train_rows"],
        "full_train_sha256": actual,
        "sample_rows": len(selected),
        "sample_rows_by_source": counts,
        "source_rows_seen": seen,
        "sample_sha256": sha256(output),
        "seed": a.seed,
        "tokenizer_sha256": manifest["tokenizer_sha256"],
        "note": "Deterministic stratified reservoir sample; 20k v1 + 72k CSIC + 108k TDX."
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
