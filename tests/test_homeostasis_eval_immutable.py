import torch
from model.homeostasis import Homeostasis


def test_homeostasis_does_not_adapt_during_eval():
    layer = Homeostasis(dim=16, k_top=4)
    layer.train()
    layer(torch.randn(2, 5, 16))
    layer.eval()
    before_m = layer.m.clone()
    before_bias = layer.bias.detach().clone()
    layer(torch.randn(2, 7, 16) * 4)
    torch.testing.assert_close(layer.m, before_m)
    torch.testing.assert_close(layer.bias.detach(), before_bias)


def test_homeostasis_adapts_during_training():
    layer = Homeostasis(dim=16, k_top=4)
    layer.train()
    before = layer.m.clone()
    layer(torch.randn(2, 5, 16) + 1.0)
    assert not torch.equal(layer.m, before)
