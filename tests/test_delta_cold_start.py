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



def test_delta_low_rank_state_reduces_memory_and_keeps_finite_gradients():
    torch.manual_seed(11)
    layer = EstadoDelta(dim=64, tau=0.001, rank=16)
    x = torch.randn(2, 12, 64)
    energy = torch.ones(2, 12, 1)
    output, surprise = layer(x, energy, reset_state=True)
    (output.square().mean() + surprise.mean()).backward()
    assert layer.state.shape == (2, 16, 64)
    assert layer.state.numel() == 2 * 16 * 64
    assert torch.isfinite(output).all()
    assert torch.isfinite(surprise).all()
    assert torch.isfinite(layer.memory_projection).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters())


def test_full_rank_projection_is_orthonormal():
    torch.manual_seed(12)
    layer = EstadoDelta(dim=16, rank=16)
    identity = torch.eye(16)
    assert torch.allclose(
        layer.memory_projection.T @ layer.memory_projection,
        identity, atol=1e-5, rtol=1e-5,
    )



def test_parallel_delta_scan_matches_sequential_reference_and_gradients():
    import copy

    torch.manual_seed(29)
    reference = EstadoDelta(dim=32, tau=0.001, rank=12, parallel_refinements=6)
    parallel = copy.deepcopy(reference)
    reference.parallel_scan = False
    parallel.parallel_scan = True

    x_ref = (torch.randn(2, 24, 32) * 0.1).requires_grad_(True)
    x_fast = x_ref.detach().clone().requires_grad_(True)
    energy = torch.ones(2, 24, 1)

    out_ref, surprise_ref = reference(x_ref, energy, reset_state=True)
    out_fast, surprise_fast = parallel(x_fast, energy, reset_state=True)

    assert torch.allclose(out_fast, out_ref, atol=2e-5, rtol=2e-4)
    assert torch.allclose(surprise_fast, surprise_ref, atol=2e-5, rtol=2e-4)
    assert torch.allclose(parallel.state, reference.state, atol=2e-5, rtol=2e-4)

    loss_ref = out_ref.square().mean() + surprise_ref.mean()
    loss_fast = out_fast.square().mean() + surprise_fast.mean()
    loss_ref.backward()
    loss_fast.backward()

    assert torch.allclose(x_fast.grad, x_ref.grad, atol=2e-5, rtol=2e-3)
    for (name_ref, parameter_ref), (name_fast, parameter_fast) in zip(
        reference.named_parameters(), parallel.named_parameters()
    ):
        assert name_ref == name_fast
        if parameter_ref.grad is not None:
            assert parameter_fast.grad is not None
            assert torch.allclose(
                parameter_fast.grad, parameter_ref.grad, atol=2e-5, rtol=2e-3
            ), name_ref
