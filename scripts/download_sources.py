#!/usr/bin/env python3
"""Stage deterministic, license-tracked LUMEN source samples (not the final train set)."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from datasets import load_dataset
from huggingface_hub import HfApi

LOG = logging.getLogger("lumen.sources")
SEED = 42
MAX_CHARS = 12000
READ_MULTIPLIER = 50
MIN_TEXT_CHARS = 24
MIN_TEXT_WORDS = 4

# The external allocation is 80% of 25k. The remaining 5k must be supplied
# from the owner's curated corpus; this script never fabricates those records.
SOURCES = [
    dict(key="tinystories", repo="roneneldan/TinyStories", config=None, quota=4000,
         kind="lenguaje", license="CDLA-Sharing-1.0", license_url="https://cdla.dev/sharing-1-0/",
         confirmed=False, text="text",
         notes="Synthetic stories; preserve share-alike obligations."),
    dict(key="babylm_2026", repo="BabyLM-community/BabyLM-2026-Strict", config=None, quota=4000,
         kind="lenguaje", license="MIT", license_url="https://huggingface.co/datasets/BabyLM-community/BabyLM-2026-Strict",
         confirmed=False, text="text",
         notes="English developmental corpus; source mix includes transcripts, books and web."),
    dict(key="fineweb_edu", repo="HuggingFaceFW/fineweb-edu", config="sample-10BT", quota=2000,
         kind="factual", license="ODC-By-1.0", license_url="https://opendatacommons.org/licenses/by/1-0/",
         confirmed=False, text="text",
         notes="CommonCrawl-derived; educational scoring is not fact verification."),
    dict(key="dclm", repo="mlfoundations/dclm-baseline-1.0", config=None, quota=2000,
         kind="factual", license="CC-BY-4.0", license_url="https://creativecommons.org/licenses/by/4.0/",
         confirmed=False, text="text",
         notes="Common Crawl research baseline; source text is not fact-checked."),
    dict(key="cosmopedia_v2", repo="HuggingFaceTB/smollm-corpus", config="cosmopedia-v2", quota=4000,
         kind="lenguaje", license="ODC-By-1.0", license_url="https://opendatacommons.org/licenses/by/1-0/",
         confirmed=False, text="text",
         notes="Synthetic generated text only; prompt and seed_data are excluded."),
    dict(key="orca_math", repo="microsoft/orca-math-word-problems-200k", config=None, quota=1500,
         kind="mates", license="MIT", license_url="https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k",
         confirmed=False, question="question", answer="answer",
         notes="GPT-4-Turbo-generated answers; correctness is not assumed."),
    dict(key="tinygsm", repo="TinyGSM/TinyGSM", config=None, quota=1500,
         kind="mates", license="MIT", license_url="https://huggingface.co/datasets/TinyGSM/TinyGSM",
         confirmed=False, question="question", answer="code",
         notes="Question/code pairs; inspect solution quality before final inclusion."),
]

BLOCKED = [
    dict(key="python_edu_the_stack", repo="HuggingFaceTB/smollm-corpus", config="python-edu",
         requested_rows=1000, status="blocked_pending_per_file_license_audit",
         reason="Python-Edu originates in The Stack V2. Corpus-level ODC-By does not replace each source file's original license and attribution obligations.",
         reference="https://www.bigcode-project.org/docs/about/the-stack/"),
    dict(key="gsm8k", repo="openai/gsm8k", status="benchmark_only_not_training", license="MIT",
         reason="The master spec explicitly reserves GSM8K for evaluation, not training.",
         reference="https://huggingface.co/datasets/openai/gsm8k"),
]


def value(row: dict[str, Any], key: str) -> str:
    """Return a trimmed string field."""
    item = row.get(key, "")
    return item.strip() if isinstance(item, str) else ""


def convert(row: dict[str, Any], source: dict[str, Any]) -> dict[str, Any] | None:
    """Convert a source row to the canonical candidate record."""
    if "text" in source:
        prompt, completion = "", value(row, source["text"])
    else:
        prompt, completion = value(row, source["question"]), value(row, source["answer"])
    if (not prompt and not completion) or len(prompt) + len(completion) > MAX_CHARS:
        return None
    if source["kind"] == "mates" and (not prompt or not completion):
        return None
    raw = unicodedata.normalize("NFKC", prompt + " " + completion).casefold()
    normalized = " ".join(re.sub(r"[^\w]+", " ", raw).split())
    if len(normalized) < MIN_TEXT_CHARS or len(normalized.split()) < MIN_TEXT_WORDS:
        return None
    return {
        "prompt": prompt,
        "completion": completion,
        "source": source["repo"],
        "source_key": source["key"],
        "source_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "prefix_key": " ".join(normalized.split()[:5]),
        "type": source["kind"],
        "confirmed": source["confirmed"],
        "license": source["license"],
    }


def stream_for(source: dict[str, Any]):
    """Resolve the current immutable dataset commit and open that exact revision."""
    revision = HfApi().dataset_info(source["repo"], revision="main").sha
    args = {"path": source["repo"], "split": "train", "streaming": True, "revision": revision}
    if source["config"]:
        args["name"] = source["config"]
    return load_dataset(**args).shuffle(seed=SEED, buffer_size=10000), revision


def sample(source: dict[str, Any], global_hashes: set[str]) -> tuple[list[dict[str, Any]], str]:
    """Sample, exact-deduplicate and apply the five-token prefix rule."""
    wanted = source["quota"]
    candidates: list[dict[str, Any]] = []
    hashes: set[str] = set()
    stream, revision = stream_for(source)
    for row in stream:
        record = convert(row, source)
        if record is None or record["source_hash"] in hashes or record["source_hash"] in global_hashes:
            continue
        hashes.add(record["source_hash"])
        global_hashes.add(record["source_hash"])
        candidates.append(record)
        if len(candidates) >= wanted * READ_MULTIPLIER:
            break

    counts = Counter(item["prefix_key"] for item in candidates)
    filtered = []
    kept_prefixes: set[str] = set()
    for item in candidates:
        prefix = item["prefix_key"]
        if counts[prefix] > 10:
            if prefix in kept_prefixes:
                continue
            kept_prefixes.add(prefix)
        filtered.append(item)

    random.Random(SEED).shuffle(filtered)
    selected = filtered[:wanted]
    if len(selected) != wanted:
        raise RuntimeError(f"{source['key']}: only {len(selected)}/{wanted} rows survived; quota not silently reduced")
    for item in selected:
        item.pop("prefix_key", None)
    return selected, revision


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Write JSONL and return its row count and SHA-256."""
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
    return {"rows": len(rows), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def main() -> None:
    """Download approved source candidates and emit a provenance manifest."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/lumen-source-candidates")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.seed != SEED:
        raise SystemExit("Seed 42 is frozen by the master spec; edit the spec before changing it.")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "project": "RONIN-LUMEN V1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "status": "source_candidates_only_not_final_training_dataset",
        "final_target_rows": 25000,
        "external_target_rows": 20000,
        "owner_curated_rows_required": 5000,
        "quality_filters": {
            "minimum_normalized_characters": MIN_TEXT_CHARS,
            "minimum_normalized_words": MIN_TEXT_WORDS,
            "read_multiplier": READ_MULTIPLIER,
            "deduplication": "NFKC + casefold + punctuation/whitespace normalization",
        },
        "sources": [],
        "blocked_sources": BLOCKED,
    }

    global_hashes: set[str] = set()
    for source in SOURCES:
        LOG.info("Sampling %s: %s rows", source["key"], source["quota"])
        rows, revision = sample(source, global_hashes)
        filename = source["key"] + ".jsonl"
        result = write_jsonl(output / filename, rows)
        manifest["sources"].append({**source, "dataset_revision": revision, "file": filename, **result, "status": "downloaded_and_sampled"})
        LOG.info("Saved %s (%s rows, sha256 %s)", filename, result["rows"], result["sha256"])

    manifest["downloaded_external_rows"] = sum(item["rows"] for item in manifest["sources"])
    manifest["unresolved_external_rows"] = 1000
    manifest["final_dataset_ready"] = False
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    LOG.info("Staging complete: %s external rows. Final train_25k.jsonl remains blocked pending 1,000 licensed code rows and 5,000 owner-curated rows.",
             manifest["downloaded_external_rows"])


if __name__ == "__main__":
    main()
