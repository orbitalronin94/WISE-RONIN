import json
import sys

import pytest

from scripts.aggregate_pending_mechanism_benchmarks import (
    EXPECTED_CANDIDATES,
    main as aggregate_main,
)


def _write_report(root, candidate, decision="PENDING"):
    folder = root / candidate
    folder.mkdir()
    (folder / "comparison.json").write_text(
        json.dumps({
            "candidate": candidate,
            "decision": decision,
            "reasons": ["screening fixture"],
            "profile_delta": {},
            "candidate_result": None,
            "control": None,
        }),
        encoding="utf-8",
    )


def test_tournament_register_accounts_for_all_twenty_and_never_authorizes_training(
    tmp_path, monkeypatch
):
    assert len(EXPECTED_CANDIDATES) == 20
    (tmp_path / "main-runtime-profile.json").write_text("{}", encoding="utf-8")
    for candidate in EXPECTED_CANDIDATES:
        _write_report(tmp_path, candidate)
    monkeypatch.setattr(sys, "argv", ["aggregate", "--root", str(tmp_path)])

    aggregate_main()

    payload = json.loads(
        (tmp_path / "approved_candidates.json").read_text(encoding="utf-8")
    )
    report = (tmp_path / "AGGREGATE.md").read_text(encoding="utf-8")
    assert len(payload["received_candidates"]) == 20
    assert payload["missing_candidates"] == []
    assert payload["training_authorized"] is False
    assert "Artefactos recibidos: **20/20**" in report
    assert "Evaluaciones contabilizadas: **20/20**" in report


def test_tournament_aggregator_rejects_duplicate_candidate_artifacts(tmp_path, monkeypatch):
    (tmp_path / "main-runtime-profile.json").write_text("{}", encoding="utf-8")
    for folder_name in ("first", "second"):
        folder = tmp_path / folder_name
        folder.mkdir()
        (folder / "comparison.json").write_text(
            json.dumps({
                "candidate": EXPECTED_CANDIDATES[0],
                "decision": "PENDING",
                "profile_delta": {},
            }),
            encoding="utf-8",
        )
    monkeypatch.setattr(sys, "argv", ["aggregate", "--root", str(tmp_path)])

    with pytest.raises(ValueError, match="duplicate candidate reports"):
        aggregate_main()
