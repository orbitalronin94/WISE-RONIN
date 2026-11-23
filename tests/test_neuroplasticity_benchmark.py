import json
from pathlib import Path

from scripts.aggregate_neuroplasticity_benchmarks import EXPECTED_CANDIDATES, merge_delta
from scripts.benchmark_neuroplasticity_candidate import (
    candidate_delta,
    deep_merge,
    load_main_profile,
)


ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_matrix_contains_exactly_eighteen_unique_candidates():
    names = [f for f in EXPECTED_CANDIDATES]
    assert len(names) == 18
    assert len(set(names)) == 18
    assert set(names) == {
        "01_shared_signal_core", "02_pain_response", "03_pleasure_response",
        "04_pain_budget_gain", "05_pleasure_budget_gain", "06_ltp_threshold",
        "07_disinhibition_gain", "08_modulator_gain", "09_eligibility_window",
        "10_attention_gate", "11_tagging_capture", "12_intrinsic_threshold",
        "13_simulated_pruning", "14_tag_protected_ltd", "15_one_shot_memory",
        "16_homeostatic_scaling", "17_prioritized_replay", "18_sleep_downscaling",
    }


def test_each_candidate_has_one_declared_delta_and_unknown_names_fail():
    for name in EXPECTED_CANDIDATES:
        assert isinstance(candidate_delta(name), dict)
    try:
        candidate_delta("19_unregistered")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown candidate should be rejected")


def test_main_runtime_profile_is_the_exact_benchmark_baseline():
    profile = load_main_profile(ROOT)
    committed = json.loads((ROOT / "config/lumen_cpu_profile.json").read_text())
    assert profile == committed


def test_all_eighteen_deltas_compose_without_hidden_parameter_conflicts():
    combined = load_main_profile(ROOT)
    for name in EXPECTED_CANDIDATES:
        conflicts = merge_delta(combined, candidate_delta(name))
        assert conflicts == [], (name, conflicts)


def test_deep_merge_does_not_mutate_main_profile():
    base = {"organs": {"pain": {"response": 0.35}}, "threads": 4}
    candidate = deep_merge(base, candidate_delta("02_pain_response"))
    assert base["organs"]["pain"]["response"] == 0.35
    assert candidate["organs"]["pain"]["response"] == 0.20
    assert candidate["threads"] == 4
