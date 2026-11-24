"""Strict JSONL training data loading and frozen BPE tokenization."""
import hashlib
import json
from pathlib import Path
import torch

class BPETokenizer:
    def __init__(self, path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install tokenizers to use the trained BPE tokenizer") from exc
        self.tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text, max_len):
        """Encode with a guaranteed terminal EOS, even when truncating."""
        if max_len < 2:
            raise ValueError("max_len must leave room for BOS and EOS")
        ids = list(self.tokenizer.encode(text).ids)
        eos_id = self.tokenizer.token_to_id("[EOS]")
        if eos_id is None:
            raise ValueError("Tokenizer must define [EOS]")
        if len(ids) > max_len:
            ids = ids[:max_len - 1]
            ids.append(eos_id)
        elif not ids or ids[-1] != eos_id:
            ids = ids[:max_len - 1]
            ids.append(eos_id)
        return ids

def load_jsonl(path):
    """Load a single JSONL file or a verified, training-authorized shard manifest."""
    path = Path(path)
    manifest_path = path / "manifest.json" if path.is_dir() else (
        path if path.name == "manifest.json" else None
    )
    if manifest_path is None:
        if not path.is_file():
            raise FileNotFoundError(f"Training corpus not found: {path}")
        sources = [(path, None)]
        manifest = None
    else:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Corpus manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("training_authorized") is not True:
            raise ValueError("Corpus manifest is not authorized for training")
        shard_entries = manifest.get("shards")
        if not isinstance(shard_entries, list) or not shard_entries:
            raise ValueError("Corpus manifest has no immutable shards")
        sources = [(manifest_path.parent / item["file"], item) for item in shard_entries]

    rows = []
    digest = hashlib.sha256()
    for shard_path, shard in sources:
        if not shard_path.is_file():
            raise FileNotFoundError(f"Training corpus shard not found: {shard_path}")
        shard_digest = hashlib.sha256()
        shard_rows = 0
        with shard_path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                encoded_line = line.encode("utf-8")
                digest.update(encoded_line)
                shard_digest.update(encoded_line)
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("split") == "holdout" or row.get("source") == "holdout_500.jsonl":
                    raise ValueError(f"Holdout contamination marker at {shard_path}:{line_no}")
                if "prompt" not in row or "completion" not in row:
                    raise ValueError(f"Missing prompt/completion at {shard_path}:{line_no}")
                if row.get("language") != "es":
                    raise ValueError(f"Non-Spanish or unverified language at {shard_path}:{line_no}")
                rows.append(row)
                shard_rows += 1
        if shard is not None:
            if shard_rows != int(shard.get("rows", -1)):
                raise ValueError(f"Shard row count mismatch: {shard_path}")
            if shard_digest.hexdigest() != shard.get("sha256"):
                raise ValueError(f"Shard SHA-256 mismatch: {shard_path}")
    if not rows:
        raise ValueError("Training JSONL is empty")
    if manifest is not None and len(rows) != int(manifest.get("total_rows", -1)):
        raise ValueError("Corpus manifest total row count does not match its shards")
    return rows, digest.hexdigest()

def encode_rows(rows, tokenizer, max_len):
    """Tokenize the immutable corpus once; preserve row order for RNG sampling."""
    return [
        tokenizer.encode(
            (str(row["prompt"]) + "\n" + str(row["completion"])).strip(),
            max_len,
        )
        for row in rows
    ]


def collate_token_ids(encoded_rows, pad_id=0):
    """Pad pre-tokenized rows without invoking the tokenizer in the train loop."""
    encoded = [ids for ids in encoded_rows if len(ids) >= 2]
    if not encoded:
        raise ValueError("Batch contains no sequences with at least two BPE tokens")
    width = max(map(len, encoded))
    batch = torch.full((len(encoded), width), pad_id, dtype=torch.long)
    for index, ids in enumerate(encoded):
        batch[index, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return batch


def load_token_cache(path, data_hash, tokenizer_hash, max_len, row_count):
    """Load a cache only when it is tied to the exact corpus/tokenizer/length."""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("format") != "ronin_token_cache_v1":
        raise ValueError("Unsupported token cache format")
    expected = {
        "data_sha256": data_hash,
        "tokenizer_sha256": tokenizer_hash,
        "max_len": int(max_len),
        "row_count": int(row_count),
    }
    actual = {key: cache.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Token cache metadata mismatch: expected={expected}, actual={actual}")
    rows = cache.get("encoded_rows")
    if not isinstance(rows, list) or len(rows) != row_count:
        raise ValueError("Token cache row count/content is invalid")
    return rows


def collate_rows(rows, tokenizer, max_len):
    """Backward-compatible encoder for callers without a frozen token cache."""
    pad_id = tokenizer.tokenizer.token_to_id("[PAD]")
    if pad_id != 0:
        raise ValueError("Tokenizer must define [PAD] at token ID 0")
    return collate_token_ids(encode_rows(rows, tokenizer, max_len), pad_id=pad_id)


def collate(rows, tokenizer, batch_size, max_len, rng):
    """Backward-compatible random sampler plus deterministic batch encoder."""
    return collate_rows(rng.choices(rows, k=batch_size), tokenizer, max_len)


def pack_token_cache(encoded_rows, max_len, pad_id=0):
    """Pack cached variable-length rows once for allocation-light batch gathers."""
    packed = torch.full((len(encoded_rows), int(max_len)), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(encoded_rows), dtype=torch.long)
    for index, row in enumerate(encoded_rows):
        ids = row[:max_len]
        length = len(ids)
        if length:
            packed[index, :length] = torch.as_tensor(ids, dtype=torch.long)
        lengths[index] = length
    return packed, lengths


def collate_cached_indices(packed_rows, lengths, indices):
    """Gather a sampled batch from a packed cache, preserving dynamic batch padding."""
    index_tensor = torch.as_tensor(indices, dtype=torch.long)
    selected_lengths = lengths.index_select(0, index_tensor)
    valid = selected_lengths >= 2
    index_tensor = index_tensor[valid]
    selected_lengths = selected_lengths[valid]
    if index_tensor.numel() == 0:
        raise ValueError("Batch contains no sequences with at least two BPE tokens")
    width = int(selected_lengths.max().item())
    return packed_rows.index_select(0, index_tensor)[:, :width]
