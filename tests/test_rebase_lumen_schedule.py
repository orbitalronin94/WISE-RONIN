import copy

import pytest

from scripts.rebase_lumen_schedule import rebase_state


def sample_state():
    return {
        "step": 3000,
        "training_config": {
            "batch": 8, "max_len": 128, "lr": 2e-5, "warmup": 100,
            "total_steps": 4000, "weight_decay": 0.01,
            "scheduler_mode": "cosine_transition_v1", "transition_start": 2000,
            "transition_end": 4000, "warmup_transition": 20, "grad_clip": 0.5,
        },
        "scheduler_transition": {
            "kind": "cosine_transition_v1", "start_step": 2000, "end_step": 4000,
            "old_lr": 0.0, "new_lr": 2e-5, "warmup_steps": 20, "current_step": 3000,
        },
        "scheduler": {},
        "optimizer": {"param_groups": [{"lr": 1.015865963834808e-5}]},
        "model": {"weight": [1, 2, 3]},
        "rng": {"state": [4, 5]},
    }


def test_rebase_preserves_weights_optimizer_rate_and_rng():
    original = sample_state()
    before = copy.deepcopy(original)
    result = rebase_state(original, end_step=6000, warmup_steps=20)
    assert result["model"] == before["model"]
    assert result["rng"] == before["rng"]
    assert result["optimizer"]["param_groups"] == before["optimizer"]["param_groups"]
    assert result["scheduler_transition"]["current_step"] == 3000
    assert result["scheduler_transition"]["start_step"] == 3000
    assert result["scheduler_transition"]["end_step"] == 6000
    assert result["training_config"]["total_steps"] == 6000


def test_rebase_rejects_nonpositive_lr():
    state = sample_state()
    state["optimizer"]["param_groups"][0]["lr"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        rebase_state(state, end_step=6000)


def test_rebase_rejects_too_short_horizon():
    with pytest.raises(ValueError, match="room"):
        rebase_state(sample_state(), end_step=3010, warmup_steps=20)
