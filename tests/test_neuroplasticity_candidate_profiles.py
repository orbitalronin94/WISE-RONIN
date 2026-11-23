import pytest

from model.plasticity import PlasticityProfile
from scripts.benchmark_neuroplasticity_candidate import candidate_delta


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("06_ltp_threshold", {"threshold_scale": 0.85}),
        ("07_disinhibition_gain", {"inhibition_scale": 0.80}),
        ("08_modulator_gain", {"modulator_gain": 1.50}),
        ("09_eligibility_window", {"eligibility_retention": 0.98}),
        ("10_attention_gate", {"attention_gain": 0.40}),
        ("11_tagging_capture", {"tagging_enabled": True}),
        ("12_intrinsic_threshold", {"intrinsic_threshold_enabled": True}),
        ("13_simulated_pruning", {"simulated_pruning_enabled": True}),
        ("14_tag_protected_ltd", {"tagging_enabled": True, "ltd_enabled": True}),
        ("15_one_shot_memory", {"one_shot_enabled": True}),
    ],
)
def test_candidate_deltas_are_valid_custom_profiles(name, expected):
    delta = candidate_delta(name)
    profile = PlasticityProfile.from_config(delta["plasticity"])
    for key, value in expected.items():
        assert getattr(profile, key) == value


def test_custom_profile_is_bounded_and_neutral_remains_identity():
    assert PlasticityProfile.from_config({"mode": "neutral"}) == PlasticityProfile()
    with pytest.raises(ValueError):
        PlasticityProfile.from_config({"mode": "neutral", "threshold_scale": 0.85})
    with pytest.raises(ValueError):
        PlasticityProfile.from_config({"mode": "custom", "threshold_scale": 0.1})
