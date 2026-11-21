#!/usr/bin/env python3
"""Convert the archived, compact V2 curated SFT pool into review-only LUMEN rows.

This never grants rights clearance and never pads the 5,000-row owner quota.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def convert(row: dict, index: int) -> dict:
    user = str(row.get("user", "")).strip()
    assistant = str(row.get("assistant", "")).strip()
    if not user or not assistant:
        raise ValueError(f"Legacy row {index}: missing user/assistant")
    raw = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return {
        "prompt": user,
        "completion": assistant,
        "origin": "owner_curated",
        "source": "archive/v2/data/curated/wise_ronin_sft_balanced.jsonl",
        "source_key": "legacy_v2_curated_review",
        "legacy_type": str(row.get("tipo", "unspecified")),
        "language": "es",
        "license": "review_required_originality_and_rights_not_yet_attested",
        "rights_confirmed": False,
        "source_row_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        "review_status": "manual_review_required",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="archive/v2/data/curated/wise_ronin_sft_balanced.jsonl",
    )
    parser.add_argument(
        "--output",
        default="data/owner/legacy_v2_curated_review.jsonl",
    )
    args = parser.parse_args()
    source, output = Path(args.input), Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source.open(encoding="utf-8") as src, output.open("w", encoding="utf-8", newline="\n") as dst:
        for index, line in enumerate(src, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(json.dumps(convert(row, index), ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    print(json.dumps({
        "rows_imported_for_review": count,
        "output": str(output),
        "rights_confirmed": False,
        "counts_toward_final_5000": False,
        "next_step": "Human review and rights attestation; do not flip rights_confirmed automatically.",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
