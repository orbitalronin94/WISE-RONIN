import random

import torch

from model.homeostasis import Homeostasis
from train.transformer_baseline import capture_rng, restore_rng


def test_dead_neurons_are_measured_after_relu_and_topk():
    organ = Homeostasis(dim=8, m_target=0.05, eta=0.001, k_top=2)
    organ.train()

    # Positive pre-gate inputs do not count as functional activity when all
    # values are suppressed by the learned threshold after Top-K selection.
    organ.bias.data.fill_(2.0)
    output = organ(torch.ones(2, 4, 8))

    assert organ.last_input_positive == 1.0
    # The sparse gate emits no update, but the residual information stream
    # must remain intact instead of being erased by Top-K.
    assert torch.equal(output, torch.ones_like(output))
    assert organ.last_gate_positive == 0.0
    assert organ.neuronas_muertas() == 1.0


def test_effective_activity_ema_tracks_units_that_survive_topk():
    organ = Homeostasis(dim=8, m_target=0.05, eta=0.001, k_top=2)
    organ.train()
    x = torch.zeros(2, 4, 8)
    x[..., 0] = 2.0
    x[..., 1] = 1.0

    output = organ(x)

    assert torch.count_nonzero(output[..., 0]).item() == 8
    assert organ.effective_m[0].item() > 0
    assert organ.effective_m[1].item() > 0
    assert torch.count_nonzero(organ.effective_m[2:]).item() == 0
    assert organ.neuronas_muertas() == 0.75


def test_transformer_resume_restores_sampler_and_torch_rng():
    local_rng = random.Random(42)
    torch.manual_seed(42)
    state = capture_rng(local_rng)

    expected_python = random.random()
    expected_torch = torch.rand(3)
    expected_local = local_rng.random()

    random.random()
    torch.rand(7)
    local_rng.random()

    restore_rng(state, local_rng)

    assert random.random() == expected_python
    assert torch.equal(torch.rand(3), expected_torch)
    assert local_rng.random() == expected_local

def test_transformer_checkpoint_persists_optimizer_scheduler_and_rng(tmp_path):
    from types import SimpleNamespace

    from train.transformer_baseline import make_scheduler, save_checkpoint

    model = torch.nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = make_scheduler(optimizer, warmup=2, total=20)
    local_rng = random.Random(123)
    args = SimpleNamespace(
        seed=123, batch=2, max_len=8, lr=1e-3, warmup=2, total_steps=20
    )
    path = tmp_path / "transformer" / "step_5.pt"

    save_checkpoint(
        path, model, optimizer, scheduler, 5, args,
        "corpus-hash", "tokenizer-hash", local_rng, 77,
        sum(parameter.numel() for parameter in model.parameters()),
    )
    state = torch.load(path, map_location="cpu", weights_only=False)

    assert state["step"] == 5
    assert state["tokens_seen"] == 77
    assert state["architecture_id"]
    assert "optimizer" in state
    assert "scheduler" in state
    assert {"python", "torch_cpu", "local_sampler"} <= state["rng"].keys()


def test_v30_to_v31_migration_resets_only_homeostasis_state():
    from train.train import migrate_legacy_homeostasis

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.homeo = torch.nn.ModuleList([
                Homeostasis(dim=8, m_target=0.05, eta=0.001, k_top=2),
                Homeostasis(dim=8, m_target=0.05, eta=0.001, k_top=2),
            ])
            self.other = torch.nn.Parameter(torch.tensor([3.0]))

    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = sum(parameter.square().sum() for parameter in model.parameters())
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        for organ in model.homeo:
            organ.bias.fill_(2.0)
            organ.m.fill_(0.4)
            organ.effective_m.fill_(0.02)

    other_before = model.other.detach().clone()
    other_state_before = {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in optimizer.state[model.other].items()
    }

    migrate_legacy_homeostasis(model, optimizer)

    for organ in model.homeo:
        assert torch.count_nonzero(organ.bias).item() == 0
        assert torch.allclose(organ.m, torch.full_like(organ.m, organ.m_target))
        assert torch.count_nonzero(organ.effective_m).item() == 0
        bias_state = optimizer.state[organ.bias]
        assert torch.count_nonzero(bias_state["exp_avg"]).item() == 0
        assert torch.count_nonzero(bias_state["exp_avg_sq"]).item() == 0

    assert torch.equal(model.other.detach(), other_before)
    for key, before in other_state_before.items():
        after = optimizer.state[model.other][key]
        if isinstance(before, torch.Tensor):
            assert torch.equal(after, before)
        else:
            assert after == before


def test_v31_homeostasis_migration_restores_gate_updates():
    from train.train import migrate_legacy_homeostasis

    organ = Homeostasis(dim=8, m_target=0.05, eta=0.001, k_top=2)
    model = torch.nn.Module()
    model.homeo = torch.nn.ModuleList([organ])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Simulate the legacy threshold that suppressed all positive updates.
    with torch.no_grad():
        organ.bias.fill_(2.0)
        organ.m.fill_(0.4)
        organ.effective_m.zero_()
    migrate_legacy_homeostasis(model, optimizer)

    organ.train()
    x = torch.ones(2, 4, 8)
    organ(x)
    assert organ.last_gate_positive > 0.0
    assert organ.neuronas_muertas() < 1.0


def test_bounded_multiplicative_homeostatic_scaling_tracks_effective_activity():
    organ = Homeostasis(dim=8, m_target=0.1, eta=0.01, k_top=2)
    organ.synaptic_scaling_enabled = True
    organ.train()
    before = organ.synaptic_scale.clone()
    organ(torch.randn(2, 4, 8))
    assert torch.isfinite(organ.synaptic_scale).all()
    assert torch.all(organ.synaptic_scale >= 0.8)
    assert torch.all(organ.synaptic_scale <= 1.2)
    assert not torch.equal(organ.synaptic_scale, before)
