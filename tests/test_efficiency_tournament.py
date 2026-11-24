import copy
from types import SimpleNamespace

import torch

from model.embedding import VivoEmbedding
from model.homeostasis import Homeostasis
from model.memory import MemoriaElastica
from train.loss import perdida_total


def test_vectorized_memory_writes_match_legacy_when_empty_slots_are_sufficient():
    torch.manual_seed(101)
    reference = MemoriaElastica(slots=8)
    candidate = copy.deepcopy(reference)
    candidate.vectorized_memory_writes = True
    reference.train()
    candidate.train()
    x = torch.randn(3, 12, 256)
    surprise = torch.full((3, 12, 1), 1.0)
    energy = torch.ones(3, 12, 1)

    out_ref = reference(x, surprise, energy)
    out_new = candidate(x, surprise, energy)

    torch.testing.assert_close(out_new, out_ref, atol=1e-6, rtol=1e-6)
    for name in ("lenta", "contador_lenta", "uso_slot", "usage_ema", "write_pointer"):
        torch.testing.assert_close(getattr(candidate, name), getattr(reference, name), atol=1e-6, rtol=1e-6)


def test_vectorized_memory_write_uses_legacy_fallback_when_bank_is_saturated():
    torch.manual_seed(102)
    reference = MemoriaElastica(slots=4)
    candidate = copy.deepcopy(reference)
    candidate.vectorized_memory_writes = True
    with torch.no_grad():
        reference.contador_lenta.fill_(1)
        candidate.contador_lenta.fill_(1)
        reference.uso_slot.copy_(torch.tensor([1., 2., 3., 4.]))
        candidate.uso_slot.copy_(reference.uso_slot)
        reference.usage_ema.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        candidate.usage_ema.copy_(reference.usage_ema)
    reference.train()
    candidate.train()
    x = torch.randn(2, 8, 256)
    surprise = torch.full((2, 8, 1), 1.0)
    energy = torch.ones(2, 8, 1)
    out_ref = reference(x, surprise, energy)
    out_new = candidate(x, surprise, energy)
    torch.testing.assert_close(out_new, out_ref, atol=1e-6, rtol=1e-6)
    for name in ("lenta", "contador_lenta", "uso_slot", "usage_ema", "write_pointer"):
        torch.testing.assert_close(getattr(candidate, name), getattr(reference, name), atol=1e-6, rtol=1e-6)


def test_batched_decay_accounting_matches_repeated_decay():
    torch.manual_seed(103)
    reference = MemoriaElastica(slots=4)
    candidate = copy.deepcopy(reference)
    candidate.batched_decay_accounting = True
    for module in (reference, candidate):
        with torch.no_grad():
            module.contador_lenta[0] = 1
            module.lenta[0].fill_(2.0)
            module.usage_ema[0] = 0.0
            module.tokens_since_decay.fill_(500)
        module.train()
    x = torch.zeros(1, 600, 256)
    surprise = torch.zeros(1, 600, 1)
    energy = torch.ones(1, 600, 1)
    reference(x, surprise, energy)
    candidate(x, surprise, energy)
    torch.testing.assert_close(candidate.lenta, reference.lenta, atol=1e-6, rtol=1e-6)
    assert candidate.tokens_since_decay.item() == reference.tokens_since_decay.item()


def test_embedding_usage_sampling_does_not_change_embedding_output():
    torch.manual_seed(104)
    reference = VivoEmbedding(vocab=32, dim=8)
    candidate = copy.deepcopy(reference)
    candidate.usage_interval = 3
    tokens = torch.tensor([[1, 2, 2, 4], [3, 3, 5, 6]])
    out_ref = reference(tokens)
    out_new = candidate(tokens)
    torch.testing.assert_close(out_new, out_ref)
    assert candidate.uso.sum().item() == tokens.numel()
    candidate(tokens)
    candidate(tokens)
    assert candidate.uso.sum().item() == tokens.numel()


def test_homeostasis_input_activity_ema_can_be_skipped_without_changing_output():
    torch.manual_seed(105)
    reference = Homeostasis(dim=16, k_top=4)
    candidate = copy.deepcopy(reference)
    candidate.track_input_activity = False
    x_ref = torch.randn(2, 8, 16, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    out_ref = reference(x_ref)
    out_new = candidate(x_new)
    torch.testing.assert_close(out_new, out_ref)
    assert torch.count_nonzero(candidate.m).item() == 0
    assert torch.count_nonzero(reference.m).item() > 0
    out_ref.sum().backward()
    out_new.sum().backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad)


def test_shared_loss_normalization_matches_loss_and_logits_gradients():
    torch.manual_seed(106)
    logits_ref = torch.randn(2, 12, 32, requires_grad=True)
    logits_new = logits_ref.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 32, (2, 12))
    output_ref = {
        "logits": logits_ref, "router": torch.rand(2, 12, 2),
        "activations": torch.randn(2, 12, 8), "hidden": torch.randn(2, 12, 8),
    }
    output_new = {**output_ref, "logits": logits_new}
    model = SimpleNamespace(emb=None, predict_latent=None)
    loss_ref, metrics_ref = perdida_total(output_ref, targets, model, use_jepa=False, shared_loss_normalization=False)
    loss_new, metrics_new = perdida_total(output_new, targets, model, use_jepa=False, shared_loss_normalization=True)
    torch.testing.assert_close(loss_new, loss_ref, atol=2e-6, rtol=2e-6)
    for key in metrics_ref:
        torch.testing.assert_close(metrics_new[key], metrics_ref[key], atol=2e-6, rtol=2e-6)
    loss_ref.backward()
    loss_new.backward()
    torch.testing.assert_close(logits_new.grad, logits_ref.grad, atol=3e-6, rtol=3e-5)


def test_local_learning_produces_gradients_for_each_delta_organ():
    from model.lumen import RoninLumen
    torch.manual_seed(107)
    model = RoninLumen()
    model.local_learning_enabled = True
    model.train()
    tokens = torch.randint(4, model.config.vocab_size, (2, 16))
    output = model(tokens, reset_state=True, collect_diagnostics=False)
    assert len(output["local_hidden"]) == len(model.delta_layers)
    loss, _ = perdida_total(output, tokens, model, use_jepa=True, use_rep=False)
    loss.backward()
    for layer in model.delta_layers:
        assert layer.proj_v.weight.grad is not None
        assert torch.isfinite(layer.proj_v.weight.grad).all()
        assert layer.proj_v.weight.grad.norm() > 0
