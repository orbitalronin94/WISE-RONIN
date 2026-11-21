import torch

from model.config import CONFIG
from model.delta_state import EstadoDelta


def test_pilot_surprise_gate_allows_cold_start_delta_writes():
    # The first pilot measured mean prediction error around 0.003 at step 1.
    # A 0.15 threshold therefore suppressed all recurrent writes from zero state.
    assert CONFIG.surprise_threshold < 0.003
    torch.manual_seed(42)
    layer = EstadoDelta(dim=16, tau=CONFIG.surprise_threshold)
    x = torch.randn(2, 8, 16) * 0.02
    # Match Presupuesto.forward(): per-token allocation has shape [batch, steps, 1].
    energy = torch.ones(2, 8, 1)

    output, surprise = layer(x, energy, reset_state=True)

    assert torch.isfinite(surprise).all()
    assert (surprise >= CONFIG.surprise_threshold).any()
    assert torch.count_nonzero(layer.state).item() > 0
    assert torch.count_nonzero(output).item() > 0


def test_lumen_delta_stack_preserves_homeostatic_activity_on_cold_start():
    from model.lumen import RoninLumen

    torch.manual_seed(42)
    model = RoninLumen()
    model.train()
    tokens = torch.randint(4, CONFIG.vocab_size, (2, 8))
    output = model(tokens, reset_state=True)

    assert torch.isfinite(output["logits"]).all()
    activity = [organ.last_input_positive for organ in model.homeo]
    assert all(rate > 0.0 for rate in activity), (
        "A Delta layer erased the representation before homeostasis: "
        f"per-layer positive input rates={activity}"
    )
    assert all(organ.last_output_positive > 0.0 for organ in model.homeo), (
        "Homeostasis produced no positive activations in at least one layer: "
        f"{[organ.last_output_positive for organ in model.homeo]}"
    )



def test_delta_state_bounds_recurrent_writes_under_extreme_input_magnitude():
    torch.manual_seed(7)
    layer = EstadoDelta(dim=16, tau=0.001)
    x = torch.randn(2, 8, 16) * 1e8
    energy = torch.ones(2, 8, 1)

    output, surprise = layer(x, energy, reset_state=True)
    output.square().mean().backward()

    assert torch.isfinite(surprise).all()
    assert torch.isfinite(layer.state).all()
    assert torch.isfinite(output).all()
    # Decay >= .90 and rank-one writes of norm <= 1 imply a geometric
    # Frobenius-norm ceiling of 10 from a zero state.
    assert layer.state.norm(dim=(-2, -1)).max().item() <= 10.001
    assert torch.all(output.norm(dim=-1) <= 0.5001 * x.norm(dim=-1))
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )
