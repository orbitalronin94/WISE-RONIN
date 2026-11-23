import ast
import json
import sys

from scripts.aggregate_neuroplasticity_benchmarks import EXPECTED_CANDIDATES, main as aggregate_main


def test_every_training_sleep_path_receives_replay_and_downscale_profile():
    tree = ast.parse(open("train/train.py", encoding="utf-8").read())
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "sueno"
    ]
    assert len(calls) >= 2
    for call in calls:
        keywords = {item.arg for item in call.keywords}
        assert {"compression_ratio", "downscale"} <= keywords


def test_aggregator_does_not_authorize_training_when_any_candidate_is_missing(tmp_path, monkeypatch):
    (tmp_path / "main-runtime-profile.json").write_text(
        json.dumps({"threads": 4, "organs": {}}), encoding="utf-8"
    )
    for name in EXPECTED_CANDIDATES[:-1]:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "comparison.json").write_text(
            json.dumps({
                "candidate": name,
                "decision": "APPROVED",
                "profile_delta": {},
                "candidate_result": {
                    "mean_loss_100_steps": 1.0,
                    "steps_per_second": 10.0,
                    "max_dead_percent": 0.1,
                },
                "control": {
                    "mean_loss_100_steps": 1.1,
                    "steps_per_second": 9.0,
                    "max_dead_percent": 0.1,
                },
            }),
            encoding="utf-8",
        )
    monkeypatch.setattr(sys, "argv", ["aggregate", "--root", str(tmp_path)])
    aggregate_main()
    result = json.loads((tmp_path / "approved_candidates.json").read_text(encoding="utf-8"))
    assert result["missing_candidates"] == [EXPECTED_CANDIDATES[-1]]
    assert result["training_authorized"] is False
