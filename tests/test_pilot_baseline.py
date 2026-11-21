import torch

from model.config import CONFIG
from model.lumen import RoninLumen
from train.transformer_baseline import SmallTransformerLM


def test_transformer_baseline_is_parameter_matched_to_lumen():
    lumen = RoninLumen()
    baseline = SmallTransformerLM(CONFIG.vocab_size, 128)
    lumen_count = sum(p.numel() for p in lumen.parameters())
    baseline_count = sum(p.numel() for p in baseline.parameters())
    assert baseline_count == 2_295_168
    assert abs(baseline_count - lumen_count) / lumen_count < 0.01


def test_transformer_baseline_causal_attention():
    torch.manual_seed(7)
    model = SmallTransformerLM(CONFIG.vocab_size, 128).eval()
    a = torch.tensor([[2, 17, 18, 19]])
    b = torch.tensor([[2, 17, 18, 29]])
    with torch.no_grad():
        out_a = model(a)
        out_b = model(b)
    torch.testing.assert_close(out_a[:, :3], out_b[:, :3], rtol=0, atol=1e-6)


def test_transformer_baseline_forward_shape():
    model = SmallTransformerLM(CONFIG.vocab_size, 128)
    tokens = torch.randint(0, CONFIG.vocab_size, (2, 24))
    assert model(tokens).shape == (2, 24, CONFIG.vocab_size)
