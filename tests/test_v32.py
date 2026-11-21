import torch
from model.config import CONFIG
from model.lumen import RoninLumen
from model.memory import MemoriaElastica
from model.ze import DualModelZe
from train.loss import perdida_total


def test_v32_dual_state_forward_and_free_energy_backward():
    model = RoninLumen(v32_ze=True)
    tokens = torch.randint(4, CONFIG.vocab_size, (2, 16))
    output = model(tokens, reset_state=True)
    assert output["logits"].shape == (2, 16, CONFIG.vocab_size)
    assert output["ze"]["F_A"].shape == (2, 16)
    assert output["ze"]["F_B"].shape == (2, 16)
    assert torch.isfinite(output["free_energy_complexity"])
    loss, metrics = perdida_total(output, tokens, model, use_free_energy=True)
    assert torch.isfinite(loss)
    assert torch.isfinite(metrics["F_complexity"])
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in model.parameters())


def test_ze_operator_removes_context_component_with_bounded_change():
    ze = DualModelZe()
    state = torch.randn(2, 128, 128)
    direction = torch.randn(2, 128)
    before = state.clone()
    removed = ze.operator_borrado(state, direction)
    assert removed >= 0
    assert torch.isfinite(state).all()
    assert (state - before).norm() <= 0.25 * before.norm() + 1e-5


def test_ze_detects_ten_consecutive_conflict_tokens():
    ze = DualModelZe(threshold=-1.0, streak_limit=10)
    hidden = torch.randn(2, 12, 256)
    pe = torch.ones(2, 12, 1)
    _, metrics = ze(hidden, pe, reset_state=True)
    assert metrics["erase_trigger"][:, 9].all()


def test_memory_ternary_gate_only_large_pe_allocates_new_slots():
    memory = MemoriaElastica(dim=16, hidden=4, slots=8)
    x = torch.randn(2, 4, 16)
    energy = torch.ones(2, 4, 1)
    medium = torch.full((2, 4, 1), 0.5)
    memory(x, medium, energy)
    assert memory.occupancy() == 0.0
    large = torch.full((2, 4, 1), 0.9)
    memory(x, large, energy)
    assert memory.occupancy() > 0.0
    assert memory.occupancy() <= 1.0
