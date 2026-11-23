import pytest

from model.plasticity import PlasticityProfile
from scripts.benchmark_pending_mechanism_candidate import candidate_delta


EXPECTED = {
    "01_tagging_capture",
    "02_intrinsic_threshold",
    "03_simulated_pruning",
    "04_tag_protected_ltd",
    "05_one_shot_memory",
    "06_homeostatic_scaling",
    "07_prioritized_replay",
    "08_sleep_downscaling",
    "09_tag_plus_intrinsic",
    "10_tag_plus_pruning",
    "11_tag_plus_one_shot",
    "12_tag_plus_scaling",
    "13_tag_plus_replay",
    "14_tag_plus_downscale",
    "15_intrinsic_plus_pruning",
    "16_one_shot_plus_scaling",
    "17_pruning_plus_ltd",
    "18_scaling_plus_downscale",
    "19_tag_replay_downscale",
    "20_all_pending",
}


def test_exactly_twenty_unique_candidate_profiles_are_registered():
    import ast
    from pathlib import Path

    source = Path("scripts/benchmark_pending_mechanism_candidate.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "candidate_delta"
    )
    assignment = next(
        node for node in function.body if isinstance(node, ast.Assign)
    )
    names = set(ast.literal_eval(assignment.value).keys())
    assert names == EXPECTED
    assert len(names) == 20


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_candidate_uses_a_valid_profile(name):
    delta = candidate_delta(name)
    if "plasticity" in delta:
        profile = PlasticityProfile.from_config(delta["plasticity"])
        assert profile.name == "custom"
    if "organs" in delta:
        assert delta["organs"]["homeostasis"]["synaptic_scaling"] is True
    if "replay_compression" in delta:
        assert 0.1 <= delta["replay_compression"] <= 1.0
        assert delta["sleep_interval_steps"] == 50
    if "sleep_downscale" in delta:
        assert 0.90 <= delta["sleep_downscale"] <= 1.0
        assert delta["sleep_interval_steps"] == 50


def test_all_pending_candidate_activates_all_eight_pending_capabilities():
    delta = candidate_delta("20_all_pending")
    plasticity = PlasticityProfile.from_config(delta["plasticity"])
    assert plasticity.tagging_enabled
    assert plasticity.intrinsic_threshold_enabled
    assert plasticity.simulated_pruning_enabled
    assert plasticity.ltd_enabled
    assert plasticity.one_shot_enabled
    assert delta["organs"]["homeostasis"]["synaptic_scaling"] is True
    assert delta["replay_compression"] == 0.5
    assert delta["sleep_downscale"] == 0.99
