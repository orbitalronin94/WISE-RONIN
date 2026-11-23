import torch

from model.chunking import AdaptiveChunking
from model.delta_state import EstadoDelta


def test_adaptive_chunking_uses_shorter_chunks_for_higher_surprise():
    module = AdaptiveChunking(dim=16, min_chunk=4, max_chunk=16)
    low = torch.full((2, 32, 1), 0.0001)
    high = torch.full((2, 32, 1), 0.1)
    assert module.choose_chunk_size(high) < module.choose_chunk_size(low)


def test_adaptive_chunking_preserves_shape_and_bounds_residual():
    torch.manual_seed(7)
    module = AdaptiveChunking(dim=16)
    x = torch.randn(2, 32, 16)
    surprise = torch.rand(2, 32, 1)
    output = module(x, surprise)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert module.min_chunk <= module.last_chunk_size <= module.max_chunk
    assert module.last_chunk_count >= 1
    assert torch.all((output - x).norm(dim=-1) <= 0.025 * x.norm(dim=-1) + 1e-6)


def test_delta_eligibility_trace_persists_across_transient_reset():
    torch.manual_seed(11)
    delta = EstadoDelta(dim=16, rank=4, parallel_refinements=2)
    x = torch.randn(2, 8, 16)
    energy = torch.ones(2, 8, 1)
    delta(x, energy, reset_state=True)
    trace = delta.eligibility_trace.clone()
    assert torch.isfinite(trace).all()
    assert trace.norm() > 0
    delta.set_eligibility_modulator(0.8)
    assert 0 < float(delta.eligibility_modulator) <= 0.25
    delta.clear_state()
    assert delta.state.numel() == 0
    assert torch.equal(delta.eligibility_trace, trace)


def test_delta_eligibility_modulator_is_bounded_and_signed():
    delta = EstadoDelta(dim=16, rank=4, parallel_refinements=1)
    delta.set_eligibility_modulator(9.0)
    assert float(delta.eligibility_modulator) == 0.25
    delta.set_eligibility_modulator(-9.0)
    assert float(delta.eligibility_modulator) == -0.25


def test_lumen_builds_one_shared_surprise_uncertainty_context():
    from model.config import CONFIG
    from model.lumen import RoninLumen
    from model.neuroplasticity import SignalContext

    torch.manual_seed(31)
    model = RoninLumen()
    output = model(torch.randint(4, CONFIG.vocab_size, (2, 8)), reset_state=True)
    signals = output["signals"]
    assert isinstance(signals, SignalContext)
    assert signals.surprise.shape == output["surprise"].shape
    assert torch.equal(signals.surprise, output["surprise"].detach())
    assert torch.all(signals.uncertainty >= 0)
    assert torch.all(signals.uncertainty <= 1)
