import pytest
import torch

from model.delta_state import EstadoDelta
from model.plasticity import PlasticityProfile


def test_neutral_profile_is_exact_identity():
    assert PlasticityProfile.from_config() == PlasticityProfile()
    assert PlasticityProfile.from_config({"mode": "neutral"}) == PlasticityProfile()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("disinhibition", {"threshold_scale": 0.85}),
        ("modulator_gain", {"modulator_gain": 1.5, "eligibility_retention": 0.96}),
        ("plasticity_window", {"eligibility_retention": 0.98}),
        ("attentional_gate", {"attention_gain": 0.4}),
    ],
)
def test_named_profiles_are_bounded_and_explicit(mode, expected):
    profile = PlasticityProfile.from_config({"mode": mode})
    for key, value in expected.items():
        assert getattr(profile, key) == pytest.approx(value)


def test_profile_overrides_support_deterministic_parameter_sweeps():
    profile = PlasticityProfile.from_config({
        "mode": "disinhibition",
        "threshold_scale": 0.8,
    })
    assert profile.threshold_scale == pytest.approx(0.8)


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "unknown"},
        {"mode": "neutral", "threshold_scale": 0.9},
        {"mode": "disinhibition", "threshold_scale": 0.1},
        {"mode": "modulator_gain", "modulator_gain": 3.0},
        {"mode": "plasticity_window", "eligibility_retention": 1.0},
        {"mode": "attentional_gate", "attention_gain": float("nan")},
        {"mode": "disinhibition", "surprise_ema": 0.2},
    ],
)
def test_invalid_or_unbounded_profiles_are_rejected(config):
    with pytest.raises(ValueError):
        PlasticityProfile.from_config(config)


def test_delta_gate_consumes_threshold_and_attention_profiles():
    delta = EstadoDelta(dim=4, tau=0.5, rank=2)
    surprise = torch.tensor([[[0.45]]])
    neutral = delta._plasticity_gate(surprise, surprise.dtype)
    assert neutral.item() == 0.0

    delta.plasticity_profile = PlasticityProfile.from_config({"mode": "disinhibition"})
    disinhibited = delta._plasticity_gate(surprise, surprise.dtype)
    assert disinhibited.item() == 1.0

    delta.plasticity_profile = PlasticityProfile.from_config({"mode": "attentional_gate"})
    attended = delta._plasticity_gate(surprise, surprise.dtype)
    assert attended.item() == 1.0


def test_modulator_and_window_profiles_are_non_mutating_configuration():
    delta = EstadoDelta(dim=4, tau=0.5, rank=2)
    delta.plasticity_profile = PlasticityProfile.from_config({"mode": "modulator_gain"})
    assert delta.plasticity_profile.modulator_gain == pytest.approx(1.5)
    assert delta.plasticity_profile.eligibility_retention == pytest.approx(0.96)
    delta.plasticity_profile = PlasticityProfile.from_config({"mode": "plasticity_window"})
    assert delta.plasticity_profile.eligibility_retention == pytest.approx(0.98)
