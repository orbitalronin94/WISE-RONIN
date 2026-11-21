import hashlib
import json

import pytest

from data.build_dataset import validate_code_rows, validate_owner_rows, validate_source_artifact


def write_source_artifact(tmp_path, approved):
    row = {
        "prompt": "Explain a nebula",
        "completion": "A cloud of gas and dust in space.",
        "source": "example/corpus",
        "source_key": "example",
        "license": "CC-BY-4.0",
    }
    source_path = tmp_path / "example.jsonl"
    raw = (json.dumps(row) + "\n").encode()
    source_path.write_bytes(raw)
    manifest = {
        "status": "source_candidates_only_not_final_training_dataset",
        "downloaded_external_rows": 1,
        "sources": [{
            "key": "example",
            "repo": "example/corpus",
            "dataset_revision": "a" * 40,
            "license": "CC-BY-4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "license_reviewed": approved,
            "rights_confirmed": approved,
            "file": "example.jsonl",
            "rows": 1,
            "quota": 1,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_source_manifest_fails_closed_without_human_rights_review(tmp_path):
    write_source_artifact(tmp_path, approved=False)
    with pytest.raises(ValueError, match="human license review"):
        validate_source_artifact(tmp_path)


def test_source_manifest_accepts_explicitly_reviewed_source(tmp_path):
    write_source_artifact(tmp_path, approved=True)
    rows, provenance = validate_source_artifact(tmp_path)
    assert len(rows) == 1
    assert provenance[0]["revision"] == "a" * 40


def test_code_rows_require_file_level_license_evidence():
    rows = [{
        "prompt": f"Explain function {i}",
        "completion": f"It returns result {i}.",
        "source_repo": "example/project",
        "source_path": f"src/add_{i}.py",
        "source_commit": "a" * 40,
        "source_hash": f"{i:064x}",
        "license": "MIT",
        "license_url": "https://opensource.org/license/mit",
        "license_file_path": "LICENSE",
        "license_file_sha256": "c" * 64,
        "attribution": "Copyright notice retained",
        "license_audited": True,
        "rights_confirmed": True,
    } for i in range(1000)]
    validate_code_rows(rows)
    rows[0]["license_audited"] = False
    with pytest.raises(ValueError, match="file-level license audit"):
        validate_code_rows(rows)


def test_owner_rows_require_explicit_rights_and_origin():
    rows = [{
        "prompt": f"¿Qué significa el concepto {i}?",
        "completion": f"Explicación original y diferenciada del concepto {i}.",
        "origin": "owner_curated",
        "license": "Owner-authorized original text",
        "rights_confirmed": True,
        "language": "es",
    } for i in range(5000)]
    validate_owner_rows(rows)
    rows[0]["rights_confirmed"] = False
    with pytest.raises(ValueError, match="rights_confirmed"):
        validate_owner_rows(rows)
