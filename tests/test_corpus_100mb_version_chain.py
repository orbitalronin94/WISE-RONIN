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
        "--max-block-bytes", "900"
    ], check=True)

    chain = json.loads((out / "lumen-corpus-v3-2-version-chain.json").read_text())
    assert [x["version"] for x in chain["versions"]] == [f"v3.2.{i:03d}" for i in range(1, len(chain["versions"]) + 1)]
    assert [x["parent_version"] for x in chain["versions"]] == ["v3.1"] + [x["version"] for x in chain["versions"][:-1]]
    assert len(chain["versions"]) >= 2
    assert all(x["delta_bytes"] <= 900 for x in chain["versions"])
    assert sum(x["delta_rows"] for x in chain["versions"]) == 3
    assert chain["rejected_rows_by_reason"]["duplicate_against_previous_or_current_versions"] == 1
    assert chain["rejected_rows_by_reason"]["rights_not_verified"] == 1
    assert chain["training_authorized"] is False
    # Auto-approval is opt-in and requires the named policy method.
    auto_source = root / "auto.jsonl"
    auto_out = root / "auto-out"
    auto_row = row("A policy approved batch record " + "echo " * 35,
                   review_status="auto_accept", review_method="batch_policy_v1")
    auto_source.write_text(json.dumps(auto_row) + "\\n", encoding="utf-8")
    denied = subprocess.run([sys.executable, str(BUILDER), "--input", str(auto_source),
        "--base-jsonl", str(base), "--output-dir", str(auto_out)], capture_output=True, text=True)
    assert denied.returncode == 0
    denied_chain = json.loads((auto_out / "lumen-corpus-v3-2-version-chain.json").read_text())
    assert denied_chain["versions_created"] == 0
    auto_out2 = root / "auto-out-enabled"
    subprocess.run([sys.executable, str(BUILDER), "--input", str(auto_source),
        "--base-jsonl", str(base), "--output-dir", str(auto_out2),
        "--allow-auto-approved"], check=True)
    auto_chain = json.loads((auto_out2 / "lumen-corpus-v3-2-version-chain.json").read_text())
    assert auto_chain["versions_created"] == 1
    assert auto_chain["versions"][0]["delta_rows"] == 1
    print("PASS: cumulative chain, block ceiling, dedup, rights gates and opt-in batch policy")
