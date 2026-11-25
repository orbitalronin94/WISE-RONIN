#!/usr/bin/env python3
"""Stream Spanish public-domain books into immutable 100 MiB JSONL shards.

Rows are emitted in the current trainer's prompt/completion schema. Each completion
contains at most 126 tokenizer pieces so BOS/EOS can fit in max_len=128.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_url
from tokenizers import Tokenizer
import langid


SOURCES = [
    {
        "key": "spanish_pd_books",
        "repo": "PleIAs/Spanish-PD-Books",
        "revision": "001eaf13681f483069361dd82195ce279e12ed63",
        "license": "public_domain",
        "license_url": "https://huggingface.co/datasets/PleIAs/Spanish-PD-Books",
        "share": 1.0,
    },
]

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
DNI_RE = re.compile(r"\b(?:\d{8}[A-Z]|[XYZ]\d{7}[A-Z])\b", re.I)
PHONE_RE = re.compile(r"(?<!\w)(?:\+34[\s.-]?)?(?:[6789]\d{2})[\s.-]?\d{3}[\s.-]?\d{3}(?!\w)")
ALPHA_RE = re.compile(r"[^\W\d_]", re.UNICODE)
NOISE_RE = re.compile(r"[^A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9\s.,;:!?¡¿'\"()\[\]{}%&/+\-—–…:]", re.UNICODE)


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", value).strip()


def load_denylist(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    patterns = []
    for raw in data["terms"]:
        term = normalize(raw)
        patterns.append((raw, re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")))
    return data, patterns


def find_attractors(text: str, patterns):
    value = normalize(text)
    return [raw for raw, pattern in patterns if pattern.search(value)]


def text_quality(text: str) -> tuple[bool, str]:
    text = text.strip()
    if len(text) < 800:
        return False, "too_short"
    words = re.findall(r"\b\w+\b", text, re.UNICODE)
    if len(words) < 120:
        return False, "too_few_words"
    alpha = len(ALPHA_RE.findall(text)) / max(1, len(text))
    if alpha < 0.58:
        return False, "low_alphabetic_density"
    noise = len(NOISE_RE.findall(text)) / max(1, len(text))
    if noise > 0.035:
        return False, "ocr_noise"
    if EMAIL_RE.search(text) or DNI_RE.search(text) or PHONE_RE.search(text):
        return False, "personal_identifier_pattern"
    try:
        lang = langid.classify(text[:4000])[0]
    except Exception:
        return False, "language_detection_error"
    if lang != "es":
        return False, "not_spanish"
    return True, "ok"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShardWriter:
    def __init__(self, root: Path, max_bytes: int):
        self.root = root
        self.max_bytes = max_bytes
        self.index = -1
        self.handle = None
        self.bytes = 0
        self.rows = 0
        self.shards = []
        self._open_next()

    def _open_next(self):
        if self.handle:
            self.handle.close()
        self.index += 1
        path = self.root / f"corpus-v4-{self.index:05d}.jsonl"
        self.handle = path.open("wb")
        self.bytes = 0
        self.shards.append({"file": path.name, "rows": 0, "bytes": 0})

    def write(self, row: dict):
        payload = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        if self.bytes and self.bytes + len(payload) > self.max_bytes:
            self._open_next()
        if len(payload) > self.max_bytes:
            raise ValueError("A single row exceeds the 100 MiB shard cap")
        self.handle.write(payload)
        self.bytes += len(payload)
        self.rows += 1
        self.shards[-1]["rows"] += 1
        self.shards[-1]["bytes"] += len(payload)

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None
        self.shards = [item for item in self.shards if item["rows"] > 0]
        for item in self.shards:
            item["sha256"] = sha256_file(self.root / item["file"])


def stream_parquet_source(source, seed: int):
    # Parquet schemas vary across source shards. Infer each immutable file separately
    # instead of forcing the entire repository through one stale dataset-card schema.
    api = HfApi()
    files = api.list_repo_files(
        repo_id=source["repo"], repo_type="dataset", revision=source["revision"]
    )
    parquet_files = sorted(
        name for name in files
        if name.lower().endswith(".parquet")
        and (not source.get("path_prefix") or name.startswith(source["path_prefix"]))
    )
    if not parquet_files:
        raise RuntimeError(f"No Parquet files found for {source['repo']}@{source['revision']}")
    import random
    random.Random(seed).shuffle(parquet_files)
    print(f"FILES {source['key']} parquet_files={len(parquet_files)}", flush=True)
    for file_index, name in enumerate(parquet_files):
        print(f"FILE_START {source['key']} {file_index + 1}/{len(parquet_files)} {name}", flush=True)
        url = hf_hub_url(
            repo_id=source["repo"], filename=name, repo_type="dataset",
            revision=source["revision"]
        )
        shard = load_dataset(
            "parquet", data_files={"train": [url]}, split="train", streaming=True
        ).shuffle(seed=seed + file_index, buffer_size=128)
        yield from shard

def metadata_text(item: dict) -> str:
    """Flatten all document metadata; exclude the full body to avoid duplicate scans."""
    excluded = {"text", "content", "document"}
    values = []

    def visit(value, key=""):
        if key.lower() in excluded:
            return
        if isinstance(value, str):
            if value.strip():
                values.append(value)
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
        elif isinstance(value, (list, tuple)):
            for child_value in value:
                visit(child_value, key)

    visit(item)
    return "\n".join(values)


def get_text(item):
    for key in ("text", "content", "document"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def get_document_id(item, source_key: str, index: int) -> str:
    # Newspaper files identify an issue; prefer that over publication-level identifiers.
    file_name = item.get("file_name") or item.get("file")
    if file_name is not None and str(file_name).strip():
        return f"{item.get('directory', item.get('id', source_key))}/{file_name}"
    identifier = item.get("identifier")
    if identifier is not None and str(identifier).strip():
        return str(identifier)
    for key in ("id", "publication_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    seed = json.dumps({k: item.get(k) for k in ("title", "creator", "publication_date", "start_date")}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(f"{source_key}|{index}|{seed}".encode("utf-8")).hexdigest()


def encode_chunks(text: str, tokenizer: Tokenizer, chunk_tokens: int):
    encoded = tokenizer.encode(text, add_special_tokens=False)
    offsets = [(start, end) for start, end in encoded.offsets if end > start]
    ids = [token_id for token_id, (start, end) in zip(encoded.ids, encoded.offsets) if end > start]
    if not ids or len(ids) != len(offsets):
        return
    for start in range(0, len(ids), chunk_tokens):
        end = min(start + chunk_tokens, len(ids))
        is_final_chunk = end == len(ids)
        if end - start < 32 and not is_final_chunk:
            continue
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunk = text[char_start:char_end].strip()
        if len(chunk) >= 80 or (is_final_chunk and len(chunk) >= 20):
            yield start // chunk_tokens, chunk, ids[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--denylist", type=Path, default=Path("data/corpus_v4_attractor_denylist.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/corpus-v4-openculture"))
    parser.add_argument("--target-rows", type=int, default=1_800_000)
    parser.add_argument("--chunk-tokens", type=int, default=126)
    parser.add_argument("--max-shard-bytes", type=int, default=100 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.target_rows < 1 or not 32 <= args.chunk_tokens <= 126:
        parser.error("target-rows must be positive and chunk-tokens must be 32..126")
    if args.max_shard_bytes > 100 * 1024 * 1024:
        parser.error("max-shard-bytes cannot exceed 100 MiB")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    tokenizer_sha = sha256_file(args.tokenizer)
    denylist, patterns = load_denylist(args.denylist)
    writer = ShardWriter(args.output_dir, args.max_shard_bytes)
    db = sqlite3.connect(args.output_dir / "dedup.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS chunks (sha256 TEXT PRIMARY KEY)")
    db.commit()

    total_docs = 0
    accepted_docs = 0
    total_rows = 0
    source_stats = {}
    rejects = Counter()
    chunk_seen = 0

    try:
        for source_index, source in enumerate(SOURCES):
            # The final source absorbs any shortfall from earlier quotas.
            quota = (
                args.target_rows - total_rows
                if source_index == len(SOURCES) - 1
                else round(args.target_rows * source["share"])
            )
            source_rows = 0
            source_docs = 0
            accepted_source_docs = 0
            print(f"START {source['key']} quota={quota}", flush=True)
            dataset = stream_parquet_source(source, args.seed)
            for index, item in enumerate(dataset):
                if source_rows >= quota or total_rows >= args.target_rows:
                    break
                source_docs += 1
                total_docs += 1
                if total_docs % 2_500 == 0:
                    print(
                        f"DOC_PROGRESS source={source['key']} "
                        f"source_docs={source_docs} total_docs={total_docs} "
                        f"source_rows={source_rows} total_rows={total_rows}",
                        flush=True,
                    )
                text = get_text(item)
                if not text:
                    rejects["missing_text"] += 1
                    continue
                text = unicodedata.normalize("NFC", text).replace("\x00", " ")
                text = re.sub(r"[ \t]+", " ", text)
                text = re.sub(r"\n{3,}", "\n\n", text).strip()
                hits = find_attractors(text + "\n" + metadata_text(item), patterns)
                if hits:
                    rejects["attractor:" + "|".join(hits)] += 1
                    continue
                ok, reason = text_quality(text)
                if not ok:
                    rejects[reason] += 1
                    continue

                accepted_source_docs += 1
                accepted_docs += 1
                doc_id = get_document_id(item, source["key"], index)
                title = str(item.get("title") or item.get("file_name") or "")[:500]
                creator = str(item.get("creator") or "")[:300]
                date = str(item.get("publication_date") or item.get("start_date") or "")[:80]

                for chunk_index, chunk, _token_ids in encode_chunks(text, tokenizer, args.chunk_tokens):
                    if source_rows >= quota or total_rows >= args.target_rows:
                        break
                    digest = hashlib.sha256(normalize(chunk).encode("utf-8")).hexdigest()
                    try:
                        db.execute("INSERT INTO chunks(sha256) VALUES (?)", (digest,))
                    except sqlite3.IntegrityError:
                        rejects["duplicate_chunk"] += 1
                        continue
                    row = {
                        "id": hashlib.sha256(
                            f"{source['key']}|{source['revision']}|{doc_id}|{chunk_index}".encode("utf-8")
                        ).hexdigest()[:24],
                        "prompt": "",
                        "completion": chunk,
                        "task": "raw_text_pretraining",
                        "language": "es",
                        "source": source["repo"],
                        "source_revision": source["revision"],
                        "source_document_id": doc_id,
                        "chunk_index": chunk_index,
                        "license": source["license"],
                        "license_url": source["license_url"],
                        "attribution": source.get("attribution", source["repo"]),
                        "source_url": str(item.get("url") or ""),
                        "title": title,
                        "creator": creator,
                        "publication_date": date,
                    }
                    writer.write(row)
                    source_rows += 1
                    total_rows += 1
                    chunk_seen += 1
                    if total_rows % 25_000 == 0:
                        db.commit()
                        print(f"PROGRESS rows={total_rows}/{args.target_rows} source={source['key']}", flush=True)

            db.commit()
            source_stats[source["key"]] = {
                "repo": source["repo"],
                "revision": source["revision"],
                "license": source["license"],
                "license_url": source["license_url"],
                "rows": source_rows,
                "documents_seen": source_docs,
                "documents_accepted": accepted_source_docs,
                "quota": quota,
            }
            print(f"END {source['key']} rows={source_rows}/{quota}", flush=True)

        db.commit()
    finally:
        writer.close()
        db.close()
    dedup_path = args.output_dir / "dedup.sqlite"
    if dedup_path.exists():
        dedup_path.unlink()

    shard_rows = sum(item["rows"] for item in writer.shards)
    manifest = {
        "schema": "wise-ronin-corpus-v4-openculture-manifest-v1",
        "status": "candidate_build_not_training_authorized",
        "target_rows": args.target_rows,
        "actual_rows": total_rows,
        "row_target_met": total_rows >= args.target_rows,
        "max_sequence_length": 128,
        "chunk_tokens": args.chunk_tokens,
        "max_shard_bytes": args.max_shard_bytes,
        "tokenizer_sha256": tokenizer_sha,
        "denylist_sha256": sha256_file(args.denylist),
        "denylist_terms": denylist["terms"],
        "source_stats": source_stats,
        "documents_seen": total_docs,
        "documents_accepted": accepted_docs,
        "chunks_written": chunk_seen,
        "rejections": dict(rejects.most_common()),
        "rows_in_shards": shard_rows,
        "shards": writer.shards,
        "training_authorized": False,
        "notes": [
            "Each output row uses prompt/completion, matching train/dataset.py.",
            "Each completion is limited to 126 BPE pieces for the 128-token context.",
            "The attractor denylist is applied to the full source document before chunking.",
            "Chunks are exact-deduplicated in SQLite across both source datasets.",
            "Historical newspaper OCR is quality-filtered; all rejected counts are preserved.",
            "Public-domain status and provenance are recorded per row; no training approval is implied."
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if total_rows < args.target_rows:
        raise SystemExit(f"Target not met: {total_rows} < {args.target_rows}")


if __name__ == "__main__":
    main()

# Rebuild immutable candidate after removing non-public-domain documentary sources; allowlist regression test gates this run.
