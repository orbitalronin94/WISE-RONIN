import torch

from model.memory import MemoriaElastica
from train.sleep import sueno


class _SleepHarness:
    def __init__(self):
        self.memoria = MemoriaElastica(dim=8, hidden=4, slots=8)

    def reset_states(self):
        return None


def test_sleep_uses_prioritized_compressed_replay_and_bounded_downscaling():
    model = _SleepHarness()
    with torch.no_grad():
        model.memoria.lenta[:6].copy_(torch.randn(6, 8))
        model.memoria.contador_lenta[:6].fill_(1)
        model.memoria.usage_ema[:6].copy_(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    before = model.memoria.mlp[0].weight.detach().clone()
    loss = sueno(model, device="cpu", compression_ratio=0.5, downscale=0.99)
    assert loss >= 0
    assert torch.isfinite(model.memoria.mlp[0].weight).all()
    assert not torch.equal(model.memoria.mlp[0].weight, before)
