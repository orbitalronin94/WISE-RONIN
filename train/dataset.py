"""Strict JSONL loading and EOS-safe frozen BPE tokenization."""
import hashlib
import json
from pathlib import Path

import torch

SPECIAL_IDS = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3, "[SEP]": 4}


class BPETokenizer:
    def __init__(self, path):
        try:
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError("Install tokenizers to use the trained BPE tokenizer") from exc
        self.tokenizer = Tokenizer.from_file(str(path))
        actual = {name: self.tokenizer.token_to_id(name) for name in SPECIAL_IDS}
        if actual != SPECIAL_IDS:
            raise ValueError(f"Tokenizer special-token IDs differ: expected={SPECIAL_IDS}, actual={actual}")
        self.pad_id = SPECIAL_IDS["[PAD]"]
        self.bos_id = SPECIAL_IDS["[BOS]"]
        self.eos_id = SPECIAL_IDS["[EOS]"]
        self.sep_id = SPECIAL_IDS["[SEP]"]

    def _content_ids(self, text):
        return self.tokenizer.encode(str(text), add_special_tokens=False).ids

    def encode(self, text, max_len):
        """Encode plain text with BOS/EOS guaranteed, even when truncating."""
        if max_len < 2:
            raise ValueError("max_len must be at least 2 to preserve BOS and EOS")
        content = self._content_ids(text)[: max_len - 2]
        return [self.bos_id, *content, self.eos_id]

    def encode_prompt_completion(self, prompt, completion, max_len):
        """Keep the prompt tail and answer head while reserving BOS, SEP and EOS."""
        if max_len < 2:
            raise ValueError("max_len must be at least 2 to preserve BOS and EOS")
        prompt_ids = self._content_ids(prompt)
        completion_ids = self._content_ids(completion)
        if not completion_ids:
            raise ValueError("Completion tokenizes to an empty sequence")

        content_budget = max_len - 2
        if not prompt_ids:
            return [self.bos_id, *completion_ids[:content_budget], self.eos_id]
        if max_len < 4:
            raise ValueError("max_len must be at least 4 for prompt/SEP/completion/EOS")

        # Retain the end of long prompts: it is most likely to contain the
        # actual question and assistant prefix. Leave at least one token for
        # the answer; short prompts naturally give the answer more capacity.
        prompt_budget = min(len(prompt_ids), max(1, (content_budget - 1) // 2))
        prompt_ids = prompt_ids[-prompt_budget:]
        completion_budget = content_budget - len(prompt_ids) - 1
        if completion_budget < 1:
            prompt_ids = prompt_ids[-max(1, content_budget - 2):]
            completion_budget = content_budget - len(prompt_ids) - 1
        if completion_budget < 1:
            raise ValueError("max_len leaves no room for completion tokens")
        return [
            self.bos_id,
            *prompt_ids,
            self.sep_id,
            *completion_ids[:completion_budget],
            self.eos_id,
        ]


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Training corpus not found: {path}")
    rows = []
    digest = hashlib.sha256()
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            digest.update(line.encode("utf-8"))
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split") == "holdout" or row.get("source") == "holdout_500.jsonl":
                raise ValueError(f"Holdout contamination marker at line {line_no}")
            if "prompt" not in row or "completion" not in row:
                raise ValueError(f"Missing prompt/completion at line {line_no}")
            if not str(row["completion"]).strip():
                raise ValueError(f"Empty completion at line {line_no}")
            rows.append(row)
    if not rows:
        raise ValueError("Training JSONL is empty")
    return rows, digest.hexdigest()


def encode_rows(rows, tokenizer, max_len):
    """Tokenize the immutable corpus once; preserve row order for RNG sampling."""
    return [
        tokenizer.encode_prompt_completion(
            str(row["prompt"]), str(row["completion"]), max_len
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
    """Load only EOS-safe caches tied to the exact corpus/tokenizer/length."""
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("format") != "ronin_token_cache_v2":
        raise ValueError("Unsupported token cache format; rebuild with prepare_token_cache.py")
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
    for index, ids in enumerate(rows):
        if (not isinstance(ids, list) or len(ids) < 2 or len(ids) > max_len
                or ids[0] != SPECIAL_IDS["[BOS]"] or ids[-1] != SPECIAL_IDS["[EOS]"]
                or SPECIAL_IDS["[PAD]"] in ids):
            raise ValueError(f"Token cache row {index} violates BOS/EOS/length invariants")
    return rows


def collate_rows(rows, tokenizer, max_len):
    """Backward-compatible encoder for callers without a frozen token cache."""
    return collate_token_ids(encode_rows(rows, tokenizer, max_len), pad_id=tokenizer.pad_id)


def collate(rows, tokenizer, batch_size, max_len, rng):
    """Backward-compatible random sampler plus deterministic batch encoder."""
    return collate_rows(rng.choices(rows, k=batch_size), tokenizer, max_len)


def pack_token_cache(encoded_rows, max_len, pad_id=0):
    """Pack cached variable-length rows once for allocation-light batch gathers."""
    packed = torch.full((len(encoded_rows), int(max_len)), pad_id, dtype=torch.long)
    lengths = torch.zeros(len(encoded_rows), dtype=torch.long)
    for index, row in enumerate(encoded_rows):
        if len(row) > max_len:
            raise ValueError(f"Token cache row {index} exceeds max_len")
        ids = row
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
