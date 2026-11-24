import torch

from train.loss import perdida_total


class _Model:
    pass


def _output(logits):
    batch, steps, _ = logits.shape
    return {
        "logits": logits,
        "hidden": torch.zeros(batch, steps, 4, dtype=logits.dtype),
        "activations": torch.randn(batch, steps, 4, dtype=logits.dtype),
        "router": torch.full((batch, steps, 2), 0.5, dtype=logits.dtype),
    }


def test_shared_normalization_preserves_total_loss_and_logits_gradients():
    torch.manual_seed(19)
    base = torch.randn(2, 9, 13, dtype=torch.float64)
    targets = torch.randint(0, 13, (2, 9))
    logits_reference = base.clone().requires_grad_(True)
    logits_shared = base.clone().requires_grad_(True)

    reference, reference_metrics = perdida_total(
        _output(logits_reference), targets, _Model(),
        use_jepa=False, use_rep=True, shared_loss_normalization=False,
    )
    shared, shared_metrics = perdida_total(
        _output(logits_shared), targets, _Model(),
        use_jepa=False, use_rep=True, shared_loss_normalization=True,
    )

    reference.backward()
    shared.backward()

    assert torch.allclose(shared, reference, rtol=1e-10, atol=1e-12)
    assert torch.allclose(logits_shared.grad, logits_reference.grad, rtol=1e-10, atol=1e-12)
    for key in ("L_token", "L_rep", "L_div", "L_router"):
        assert torch.allclose(shared_metrics[key], reference_metrics[key], rtol=1e-10, atol=1e-12)


def test_shared_normalization_flag_is_inert_without_repetition_term():
    torch.manual_seed(23)
    base = torch.randn(2, 7, 11)
    targets = torch.randint(0, 11, (2, 7))
    a = base.clone().requires_grad_(True)
    b = base.clone().requires_grad_(True)

    loss_a, _ = perdida_total(
        _output(a), targets, _Model(), use_jepa=False, use_rep=False,
    )
    loss_b, _ = perdida_total(
        _output(b), targets, _Model(), use_jepa=False, use_rep=False,
        shared_loss_normalization=True,
    )
    loss_a.backward()
    loss_b.backward()

    assert torch.equal(loss_a, loss_b)
    assert torch.equal(a.grad, b.grad)


def test_shared_normalization_handles_short_sequences_without_repetition():
    torch.manual_seed(29)
    base = torch.randn(2, 2, 7)
    targets = torch.randint(0, 7, (2, 2))
    a = base.clone().requires_grad_(True)
    b = base.clone().requires_grad_(True)

    loss_a, _ = perdida_total(
        _output(a), targets, _Model(), use_jepa=False, use_rep=True,
    )
    loss_b, _ = perdida_total(
        _output(b), targets, _Model(), use_jepa=False, use_rep=True,
        shared_loss_normalization=True,
    )
    loss_a.backward()
    loss_b.backward()

    assert torch.equal(loss_a, loss_b)
    assert torch.equal(a.grad, b.grad)
