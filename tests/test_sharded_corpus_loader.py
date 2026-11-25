import hashlib
import json

from train.dataset import IndexedJsonlCorpus, load_jsonl


def _write_shard(root, name, rows):
    path = root / name
    payload = "".join(json.dumps(row, ensure_ascii=False) + chr(10) for row in rows).encode("utf-8")
    path.write_bytes(payload)
    return {
        "file": name,
        "rows": len(rows),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def test_authorized_sharded_corpus_is_lazy_and_random_access(tmp_path):
    rows_a = [
        {"prompt": "", "completion": "Primera fila.", "language": "es", "source": "a"},
        {"prompt": "Pregunta", "completion": "Segunda fila.", "language": "es", "source": "a"},
    ]
    rows_b = [
        {"prompt": "", "completion": "Tercera fila.", "language": "es", "source": "b"},
    ]
    shards = [
        _write_shard(tmp_path, "part-0.jsonl", rows_a),
        _write_shard(tmp_path, "part-1.jsonl", rows_b),
    ]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "training_authorized": True,
        "total_rows": 3,
        "shards": shards,
    }), encoding="utf-8")

    rows, digest = load_jsonl(tmp_path)
    try:
        assert isinstance(rows, IndexedJsonlCorpus)
        assert len(rows) == 3
        assert rows[0]["completion"] == "Primera fila."
        assert rows[2]["completion"] == "Tercera fila."
        assert rows[-1]["source"] == "b"
        assert len(digest) == 64
    finally:
        rows.close()


def test_sharded_corpus_rejects_bad_shard_hash(tmp_path):
    shard = _write_shard(tmp_path, "part-0.jsonl", [
        {"prompt": "", "completion": "Texto.", "language": "es"},
    ])
    shard["sha256"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps({
        "training_authorized": True,
        "total_rows": 1,
        "shards": [shard],
    }), encoding="utf-8")
    try:
        load_jsonl(tmp_path)
    except ValueError as exc:
        assert "SHA-256 mismatch" in str(exc)
    else:
        raise AssertionError("Corrupt shard hash was accepted")

# Regression coverage for the v4 lazy-shard loader.



def test_sharded_corpus_rejects_bad_shard_byte_count(tmp_path):
    shard = _write_shard(tmp_path, "part-0.jsonl", [
        {"prompt": "", "completion": "Texto.", "language": "es"},
    ])
    shard["bytes"] += 1
    (tmp_path / "manifest.json").write_text(json.dumps({
        "training_authorized": True,
        "total_rows": 1,
        "shards": [shard],
    }), encoding="utf-8")
    try:
        load_jsonl(tmp_path)
    except ValueError as exc:
        assert "byte count mismatch" in str(exc)
    else:
        raise AssertionError("Corrupt shard byte count was accepted")
