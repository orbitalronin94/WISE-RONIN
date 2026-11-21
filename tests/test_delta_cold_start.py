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
