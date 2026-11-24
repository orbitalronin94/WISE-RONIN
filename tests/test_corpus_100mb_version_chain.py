#!/usr/bin/env python3
"""Small deterministic integration test for the cumulative version builder."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_corpus_100mb_blocks.py"

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    base = root / "base.jsonl"
    source = root / "reviewed.jsonl"
    out = root / "out"
    base.write_text(json.dumps({"text": "A frozen v3.1 example that must never be re-imported"}) + "\n", encoding="utf-8")

    def row(text, **overrides):
        value = {
            "text": text,
            "source": "fixture/test",
            "source_revision": "a" * 40,
            "license": "MIT",
            "rights_status": "verified",
            "privacy_status": "clear",
            "quality_status": "approved",
            "review_status": "accept",
        }
        value.update(overrides)
        return value

    rows = [
        row("A frozen v3.1 example that must never be re-imported"),
        row("First independently reviewed sample " + "alpha " * 35),
        row("Second independently reviewed sample " + "bravo " * 35),
        row("Third independently reviewed sample " + "charlie " * 35),
        row("Rejected because rights are not cleared " + "delta " * 35, rights_status="pending"),
    ]
    source.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
    subprocess.run([
        sys.executable, str(BUILDER), "--input", str(source),
        "--base-jsonl", str(base), "--output-dir", str(out),
        "--parent-version", "v3.1", "--start-version", "1",
        "--max-block-bytes", "600"
    ], check=True)

    chain = json.loads((out / "lumen-corpus-v3-2-version-chain.json").read_text())
    assert [x["version"] for x in chain["versions"]] == ["v3.2.001", "v3.2.002", "v3.2.003"]
    assert [x["parent_version"] for x in chain["versions"]] == ["v3.1", "v3.2.001", "v3.2.002"]
    assert all(x["delta_bytes"] <= 600 for x in chain["versions"])
    assert chain["versions"][0]["delta_rows"] == 1
    assert chain["versions"][1]["delta_rows"] == 1
    assert chain["versions"][2]["delta_rows"] == 1
    assert chain["rejected_rows_by_reason"]["duplicate_against_previous_or_current_versions"] == 1
    assert chain["rejected_rows_by_reason"]["rights_not_verified"] == 1
    assert chain["training_authorized"] is False
    print("PASS: cumulative numbering, parent chain, block ceiling, v3.1 dedup and admission gates")
