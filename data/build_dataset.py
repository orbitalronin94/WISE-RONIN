"""Build the frozen 25k training set only after every provenance gate passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.audit_holdout import audit_rows, read_jsonl  # noqa: E402

EXPECTED_EXTERNAL = 20_000
EXPECTED_CODE = 1_000
EXPECTED_OWNER = 5_000
EXPECTED_TOTAL = EXPECTED_EXTERNAL + EXPECTED_OWNER


def normalized_text(row: dict[str, Any]) -> str:
    raw = f"{row.get('prompt', '')} {row.get('completion', '')}"
    raw = unicodedata.normalize("NFKC", raw).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", raw).split())


def language_of(row: dict[str, Any]) -> str:
    declared = str(row.get("language", "")).lower().split("-")[0]
    if declared in {"es", "en", "ca", "pt", "fr", "de", "it", "other"}:
        return declared
    text = normalized_text(row)
    words = set(text.split())
    es = len(words & {"que", "de", "la", "el", "en", "los", "una", "para", "con", "por", "como", "qué", "del"})
    en = len(words & {"the", "of", "and", "to", "in", "is", "for", "with", "that", "this", "from", "what"})
    if es == en == 0:
        return "und"
    return "es" if es > en else "en"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_artifact(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "source_candidates_only_not_final_training_dataset":
        raise ValueError("Unexpected source-candidate manifest status")
    rows: list[dict[str, Any]] = []
    provenance = []
    for source in manifest.get("sources", []):
        if not source.get("dataset_revision") or not source.get("license") or not source.get("license_url"):
            raise ValueError(f"{source.get('key')}: immutable revision and license metadata required")
        path = root / source["file"]
        digest = sha256_file(path)
        if digest != source.get("sha256"):
            raise ValueError(f"{source['key']}: source file SHA-256 mismatch")
        part = read_jsonl(path)
        if len(part) != source.get("quota") or len(part) != source.get("rows"):
            raise ValueError(f"{source['key']}: row count mismatch")
        for index, row in enumerate(part):
            if row.get("source_key") != source["key"] or row.get("source") != source["repo"]:
                raise ValueError(f"{source['key']} row {index}: source identity mismatch")
            if row.get("license") != source["license"]:
                raise ValueError(f"{source['key']} row {index}: license mismatch")
            row.setdefault("language", language_of(row))
        rows.extend(part)
        provenance.append({
            "source_key": source["key"], "dataset": source["repo"],
            "revision": source["dataset_revision"], "license": source["license"],
            "license_url": source["license_url"], "rows": len(part),
            "file": source["file"], "sha256": digest,
        })
    if len(rows) != int(manifest.get("downloaded_external_rows", -1)):
        raise ValueError("Aggregate candidate count differs from source manifest")
    return rows, provenance


def validate_code_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_CODE:
        raise ValueError(f"Need exactly {EXPECTED_CODE} licensed code rows; got {len(rows)}")
    required = ("source_repo", "source_path", "source_commit", "license",
                "license_url", "license_file_path", "license_file_sha256",
                "attribution", "source_hash")
    for index, row in enumerate(rows):
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            raise ValueError(f"Code row {index}: missing per-file provenance fields {missing}")
        if not re.fullmatch(r"[0-9a-fA-F]{40,64}", str(row["source_commit"])):
            raise ValueError(f"Code row {index}: source_commit must be an immutable SHA")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(row["source_hash"])):
            raise ValueError(f"Code row {index}: source_hash must be SHA-256")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", str(row["license_file_sha256"])):
            raise ValueError(f"Code row {index}: license_file_sha256 must be SHA-256")
        if row.get("license_audited") is not True or row.get("rights_confirmed") is not True:
            raise ValueError(f"Code row {index}: explicit file-level license audit and rights confirmation required")
        if not str(row.get("prompt", "")).strip() or not str(row.get("completion", "")).strip():
            raise ValueError(f"Code row {index}: prompt/completion required")
        row["type"] = row.get("type", "codigo")
        row["language"] = language_of(row)


def validate_owner_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != EXPECTED_OWNER:
        raise ValueError(f"Need exactly {EXPECTED_OWNER} owner-curated rows; got {len(rows)}")
    for index, row in enumerate(rows):
        if not str(row.get("prompt", "")).strip() or not str(row.get("completion", "")).strip():
            raise ValueError(f"Owner row {index}: prompt/completion required")
        if row.get("origin") != "owner_curated":
            raise ValueError(f"Owner row {index}: origin must be owner_curated")
        if not str(row.get("license", "")).strip():
            raise ValueError(f"Owner row {index}: license/permission declaration required")
        if row.get("rights_confirmed") is not True:
            raise ValueError(f"Owner row {index}: rights_confirmed must be true")
        row["source"] = row.get("source", "owner_curated")
        row["source_key"] = "owner_curated"
        row["language"] = language_of(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default="artifacts/lumen-source-candidates")
    parser.add_argument("--owner", default="data/owner")
    parser.add_argument("--holdout", default="data/holdout_500.jsonl")
    parser.add_argument("--output", default="data/train_25k.jsonl")
    parser.add_argument("--min-spanish-share", type=float, default=None,
                        help="Optional explicit minimum Spanish share (0..1); no default is invented")
    args = parser.parse_args()
    if args.min_spanish_share is not None and not 0 <= args.min_spanish_share <= 1:
        parser.error("--min-spanish-share must be between 0 and 1")

    root, owner = Path(args.sources), Path(args.owner)
    rows, source_manifest = validate_source_artifact(root)
    if len(rows) != EXPECTED_EXTERNAL - EXPECTED_CODE:
        raise ValueError(
            f"Current external staging must supply {EXPECTED_EXTERNAL - EXPECTED_CODE} "
            f"rows before the separately audited code quota; got {len(rows)}"
        )

    code_path = owner / "python_edu_licensed.jsonl"
    if not code_path.is_file():
        raise FileNotFoundError(f"Missing per-file-audited code set: {code_path}")
    code_rows = read_jsonl(code_path)
    validate_code_rows(code_rows)
    rows.extend(code_rows)

    owner_path = owner / "curated_5000.jsonl"
    if not owner_path.is_file():
        raise FileNotFoundError(f"Missing owner-curated corpus: {owner_path}")
    owner_rows = read_jsonl(owner_path)
    validate_owner_rows(owner_rows)
    rows.extend(owner_rows)

    if len(rows) != EXPECTED_TOTAL:
        raise ValueError(f"Expected {EXPECTED_TOTAL} rows, got {len(rows)}")

    hashes: dict[str, int] = {}
    prefixes = Counter()
    for index, row in enumerate(rows):
        text = normalized_text(row)
        if not text:
            raise ValueError(f"Row {index}: empty normalized text")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in hashes:
            raise ValueError(f"Duplicate normalized row {index}; first seen at {hashes[digest]}")
        hashes[digest] = index
        prefixes[" ".join(text.split()[:5])] += 1
    repeated_prefixes = {key: count for key, count in prefixes.items() if count > 10}
    if repeated_prefixes:
        raise ValueError(f"Global five-token prefix repetition exceeds 10: {list(repeated_prefixes.items())[:10]}")

    holdout = read_jsonl(args.holdout)
    overlap = audit_rows(rows, holdout)
    if not overlap["lexical_gate_pass"]:
        raise ValueError("Frozen-holdout lexical overlap detected; inspect audit report before building")

    language_counts = Counter(language_of(row) for row in rows)
    spanish_share = language_counts["es"] / len(rows)
    if args.min_spanish_share is not None and spanish_share < args.min_spanish_share:
        raise ValueError(
            f"Spanish share {spanish_share:.3%} is below required {args.min_spanish_share:.3%}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    output_sha = sha256_file(output)
    final_manifest = {
        "project": "RONIN-LUMEN V1",
        "status": "final_training_corpus_built_not_yet_training_authorized",
        "rows": len(rows),
        "sha256": output_sha,
        "source_artifacts": source_manifest,
        "code_rows": EXPECTED_CODE,
        "owner_curated_rows": EXPECTED_OWNER,
        "language_counts_heuristic": dict(language_counts),
        "spanish_share_heuristic": round(spanish_share, 6),
        "prefix_groups_over_10": 0,
        "exact_duplicates": 0,
        "holdout_audit": {
            "rows": overlap["holdout_rows"],
            "unique_user_prompts": overlap["holdout_unique_user_prompts"],
            "exact_matches": overlap["exact_match_count"],
            "long_user_substring_matches": overlap["long_user_substring_match_count"],
            "method": "normalized lexical exact + long user-prompt substring",
        },
        "tokenizer_status": "not_built",
        "training_status": "blocked_until_manual_review_and_tokenizer_validation",
    }
    manifest_path = output.with_name("train_25k.manifest.json")
    manifest_path.write_text(json.dumps(final_manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps(final_manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
