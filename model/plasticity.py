"""Bounded computational plasticity profiles shared by Delta layers."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PlasticityProfile:
    """Bounded computational modes; neutral is an exact identity profile."""

    name: str = "neutral"
    threshold_scale: float = 1.0
    inhibition_scale: float = 1.0
    modulator_gain: float = 1.0
    eligibility_retention: float = 0.95
    attention_gain: float = 0.0
    tagging_enabled: bool = False
    intrinsic_threshold_enabled: bool = False
    simulated_pruning_enabled: bool = False
    one_shot_enabled: bool = False
    ltd_enabled: bool = False

    @classmethod
    def from_config(cls, config=None):
        config = config or {}
        if not isinstance(config, dict):
            raise ValueError("plasticity profile must be a JSON object")
        name = str(config.get("mode", "neutral"))
        presets = {
            "neutral": {},
            "custom": {},
            "disinhibition": {"threshold_scale": 0.85, "inhibition_scale": 0.8},
            "modulator_gain": {"modulator_gain": 1.5, "eligibility_retention": 0.96},
            "plasticity_window": {"eligibility_retention": 0.98, "tagging_enabled": True},
            "attentional_gate": {"attention_gain": 0.4},
        }
        if name not in presets:
            raise ValueError(f"unknown plasticity mode: {name}")
        values = {**presets[name], **{k: v for k, v in config.items() if k != "mode"}}
        allowed = {"threshold_scale", "inhibition_scale", "modulator_gain", "eligibility_retention", "attention_gain", "tagging_enabled", "intrinsic_threshold_enabled", "simulated_pruning_enabled", "one_shot_enabled", "ltd_enabled"}
        if set(values) - allowed:
            raise ValueError(f"unknown plasticity factors: {sorted(set(values) - allowed)}")
        boolean_keys = {"tagging_enabled", "intrinsic_threshold_enabled", "simulated_pruning_enabled", "one_shot_enabled", "ltd_enabled"}
        if any(k in values and not isinstance(values[k], bool) for k in boolean_keys):
            raise ValueError("plasticity mechanism flags must be booleans")
        numeric = {k: float(v) for k, v in values.items() if k not in boolean_keys}
        flags = {k: values[k] for k in boolean_keys if k in values}
        result = cls(name=name, **numeric, **flags)
        if not 0.5 <= result.threshold_scale <= 1.5:
            raise ValueError("threshold_scale must be in [0.5,1.5]")
        if not 0.5 <= result.inhibition_scale <= 1.5:
            raise ValueError("inhibition_scale must be in [0.5,1.5]")
        if not 0.5 <= result.modulator_gain <= 2.0:
            raise ValueError("modulator_gain must be in [0.5,2.0]")
        if not 0.90 <= result.eligibility_retention <= 0.99:
            raise ValueError("eligibility_retention must be in [0.90,0.99]")
        if not 0.0 <= result.attention_gain <= 1.0:
            raise ValueError("attention_gain must be in [0,1]")
        if name == "neutral" and result != cls():
            raise ValueError("neutral profile must remain the exact identity")
        return result
