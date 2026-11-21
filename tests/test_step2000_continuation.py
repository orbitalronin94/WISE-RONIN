import copy

import pytest
import torch

from scripts.prepare_step2000_continuation import (
    EXPECTED,
    PARENT_STEP,
    END_STEP,
    continuation_config,
    expected_parent_config,
    rebase_state,
)


def make_parent(kind):
    spec = EXPECTED[kind]
    old_schedule = {
        "kind": "cosine_transition_v1",
        "start_step": 1000,
        "end_step": 2000,
        "old_lr": 1.469619736075401e-4,
        "new_lr": spec["lr"],
        "warmup_steps": 20,
        "current_step": 2000,
    }
    return {
        "step": 2000,
        "architecture_id": spec["architecture_id"],
        "params": spec["params"],
        "seed": 42,
        "tokens_seen": 1_670_175,
        "data_sha256": "data-hash",
        "tokenizer_sha256": "tokenizer-hash",
        "training_config": expected_parent_config(kind),
        "optimizer": {
            "state": {
                0: {
                    "exp_avg": torch.tensor([0.25, -0.5]),
                    "exp_avg_sq": torch.tensor([0.1, 0.2]),
                }
            },
            "param_groups": [{"lr": 0.0, "betas": (0.9, 0.999)}],
        },
        "scheduler": copy.deepcopy(old_schedule),
        "scheduler_transition": copy.deepcopy(old_schedule),
        "rng": {"torch_cpu": torch.tensor([1, 2, 3], dtype=torch.uint8)},
        "model": {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        **({"phase": 1, "pilot_all_rows": True, "v32_ze": False} if kind == "lumen" else {}),
    }


@pytest.mark.parametrize("kind", ["lumen", "transformer"])
def test_rebase_preserves_weights_optimizer_and_rng(kind):
    parent = make_parent(kind)
    weights_before = parent["model"]["weight"].clone()
    moments_before = parent["optimizer"]["state"][0]["exp_avg"].clone()
    rng_before = parent["rng"]["torch_cpu"].clone()

    result = rebase_state(parent, kind, "data-hash", "tokenizer-hash", "source-sha")

    assert result["step"] == PARENT_STEP
    assert torch.equal(result["model"]["weight"], weights_before)
    assert torch.equal(result["optimizer"]["state"][0]["exp_avg"], moments_before)
    assert torch.equal(result["rng"]["torch_cpu"], rng_before)
    assert result["training_config"] == continuation_config(kind)
    assert result["scheduler"] == {
        "kind": "cosine_transition_v1",
        "start_step": 2000,
        "end_step": END_STEP,
        "old_lr": 0.0,
        "new_lr": EXPECTED[kind]["lr"],
        "warmup_steps": 20,
        "current_step": 2000,
    }
    assert result["continuation_parent"]["source_checkpoint_sha256"] == "source-sha"


def test_rebase_refuses_wrong_parent_step():
    parent = make_parent("lumen")
    parent["step"] = 1999
    with pytest.raises(ValueError, match="expected parent step"):
        rebase_state(parent, "lumen", "data-hash", "tokenizer-hash", "source-sha")


def test_rebase_refuses_data_or_tokenizer_mismatch():
    parent = make_parent("transformer")
    with pytest.raises(ValueError, match="corpus/tokenizer hash mismatch"):
        rebase_state(parent, "transformer", "wrong-data", "tokenizer-hash", "source-sha")
