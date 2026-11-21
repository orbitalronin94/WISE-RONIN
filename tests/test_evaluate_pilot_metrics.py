import torch

from model.config import CONFIG
from model.lumen import RoninLumen
from scripts.evaluate_pilot import generation_metrics


def test_generation_metrics_capture_repetition_eos_and_full_text():
    text = "a a a b b b"
    metrics = generation_metrics(text, eos_terminated=True)
    assert metrics["rep_bigram"] == 0.4
    assert metrics["rep_trigram"] == 0.0
    assert metrics["first_loop_at"] == 0
    assert metrics["diversity"] == 2 / 6
    assert metrics["has_eos"] == 1
    assert metrics["len_tokens"] == 6
    assert metrics["text"] == text


def test_lumen_explicit_reset_clears_only_recurrent_activity():
    model = RoninLumen()
    with torch.no_grad():
        for layer in model.delta_layers:
            layer.state = torch.ones(1, CONFIG.model_dim, CONFIG.model_dim)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        biases_before = [organ.bias.detach().clone() for organ in model.homeo]
        model.reset_state()
        assert all(layer.state.numel() == 0 for layer in model.delta_layers)
        assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
        assert all(torch.equal(a, organ.bias) for a, organ in zip(biases_before, model.homeo))
