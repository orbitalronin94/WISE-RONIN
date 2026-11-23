"""Bounded computational plasticity profiles shared by Delta layers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PlasticityProfile:
    """Four bounded controls; neutral is an exact identity profile."""

    name: str = "neutral"
    threshold_scale: float = 1.0
    modulator_gain: float = 1.0
    eligibility_retention: float = 0.95
    attention_gain: float = 0.0

    @classmethod
    def from_config(cls, config=None):
        config = config or {}
        if not isinstance(config, dict):
            raise ValueError("plasticity profile must be a JSON object")
        name = str(config.get("mode", "neutral"))
        presets = {
            "neutral": {},
            "disinhibition": {"threshold_scale": 0.85},
            "modulator_gain": {"modulator_gain": 1.5, "eligibility_retention": 0.96},
            "plasticity_window": {"eligibility_retention": 0.98},
            "attentional_gate": {"attention_gain": 0.4},
        }
        if name not in presets:
            raise ValueError(f"unknown plasticity mode: {name}")
        values = {**presets[name], **{k: v for k, v in config.items() if k != "mode"}}
        allowed = {"threshold_scale", "modulator_gain", "eligibility_retention", "attention_gain"}
        if set(values) - allowed:
            raise ValueError(f"unknown plasticity factors: {sorted(set(values) - allowed)}")
        result = cls(name=name, **{k: float(v) for k, v in values.items()})
        if not 0.5 <= result.threshold_scale <= 1.5:
            raise ValueError("threshold_scale must be in [0.5,1.5]")
        if not 0.5 <= result.modulator_gain <= 2.0:
            raise ValueError("modulator_gain must be in [0.5,2.0]")
        if not 0.90 <= result.eligibility_retention <= 0.99:
            raise ValueError("eligibility_retention must be in [0.90,0.99]")
        if not 0.0 <= result.attention_gain <= 1.0:
            raise ValueError("attention_gain must be in [0,1]")
        if name == "neutral" and result != cls():
            raise ValueError("neutral profile must remain the exact identity")
        return result
