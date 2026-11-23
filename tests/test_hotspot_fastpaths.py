import copy
import torch
from model.delta_state import EstadoDelta
from model.homeostasis import Homeostasis

def _compare_grads(left, right, atol=2e-6, rtol=2e-5):
    for (name_l, param_l), (name_r, param_r) in zip(left.named_parameters(), right.named_parameters()):
        assert name_l == name_r
        if param_l.grad is None or param_r.grad is None:
            assert param_l.grad is None and param_r.grad is None
        else:
            torch.testing.assert_close(param_l.grad, param_r.grad, atol=atol, rtol=rtol)

def test_delta_sequence_normalization_matches_reference_outputs_and_gradients():
    torch.manual_seed(7)
    reference = EstadoDelta(dim=32)
    candidate = copy.deepcopy(reference)
    candidate.pre_normalize_sequence = True
    x_ref = torch.randn(3, 17, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    energy = torch.rand(3, 17, 1)
    out_ref, surprise_ref = reference(x_ref, energy, reset_state=True)
    out_new, surprise_new = candidate(x_new, energy, reset_state=True)
    torch.testing.assert_close(out_new, out_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(surprise_new, surprise_ref, atol=2e-6, rtol=2e-5)
    (out_ref.square().mean() + surprise_ref.mean()).backward()
    (out_new.square().mean() + surprise_new.mean()).backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=3e-6, rtol=3e-5)
    _compare_grads(reference, candidate, atol=3e-6, rtol=3e-5)

def test_homeostasis_fused_scatter_matches_reference_outputs_gradients_and_ema():
    torch.manual_seed(11)
    reference = Homeostasis(dim=32, k_top=8)
    candidate = copy.deepcopy(reference)
    candidate.fused_scatter_add = True
    x_ref = torch.randn(4, 19, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    out_ref = reference(x_ref)
    out_new = candidate(x_new)
    torch.testing.assert_close(out_new, out_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate.effective_m, reference.effective_m, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(candidate.bias, reference.bias, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(candidate.last_gate_activity_per_unit, reference.last_gate_activity_per_unit, atol=1e-7, rtol=1e-6)
    out_ref.square().mean().backward()
    out_new.square().mean().backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    _compare_grads(reference, candidate, atol=1e-6, rtol=1e-6)


def test_delta_write_update_reuse_matches_outputs_state_and_gradients():
    torch.manual_seed(19)
    reference = EstadoDelta(dim=32, parallel_refinements=2)
    candidate = copy.deepcopy(reference)
    candidate.reuse_write_update = True
    x_ref = torch.randn(2, 13, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    energy = torch.rand(2, 13, 1)
    out_ref, surprise_ref = reference(x_ref, energy, reset_state=True)
    out_new, surprise_new = candidate(x_new, energy, reset_state=True)
    torch.testing.assert_close(out_new, out_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(surprise_new, surprise_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(candidate.state, reference.state, atol=2e-6, rtol=2e-5)
    (out_ref.square().mean() + surprise_ref.mean()).backward()
    (out_new.square().mean() + surprise_new.mean()).backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=3e-6, rtol=3e-5)
    _compare_grads(reference, candidate, atol=3e-6, rtol=3e-5)


def test_delta_shared_causal_masks_match_outputs_state_and_gradients():
    torch.manual_seed(23)
    reference = EstadoDelta(dim=32, parallel_refinements=2)
    candidate = copy.deepcopy(reference)
    candidate.share_causal_masks = True
    x_ref = torch.randn(2, 11, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    energy = torch.rand(2, 11, 1)
    out_ref, surprise_ref = reference(x_ref, energy, reset_state=True)
    out_new, surprise_new = candidate(x_new, energy, reset_state=True)
    torch.testing.assert_close(out_new, out_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(surprise_new, surprise_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(candidate.state, reference.state, atol=2e-6, rtol=2e-5)
    (out_ref.square().mean() + surprise_ref.mean()).backward()
    (out_new.square().mean() + surprise_new.mean()).backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=3e-6, rtol=3e-5)
    _compare_grads(reference, candidate, atol=3e-6, rtol=3e-5)


def test_delta_on_demand_trace_diagnostics_preserves_computation():
    torch.manual_seed(29)
    reference = EstadoDelta(dim=32, parallel_refinements=2)
    candidate = copy.deepcopy(reference)
    candidate.trace_norm_diagnostics_on_demand = True
    x = torch.randn(2, 9, 32)
    energy = torch.rand(2, 9, 1)
    out_ref, surprise_ref = reference(x, energy, reset_state=True, collect_diagnostics=False)
    out_new, surprise_new = candidate(x, energy, reset_state=True, collect_diagnostics=False)
    torch.testing.assert_close(out_new, out_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(surprise_new, surprise_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(candidate.state, reference.state, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(candidate.eligibility_trace, reference.eligibility_trace, atol=2e-6, rtol=2e-5)


def test_delta_always_apply_eligibility_matches_conditional_path():
    torch.manual_seed(31)
    reference = EstadoDelta(dim=32, parallel_refinements=2)
    candidate = copy.deepcopy(reference)
    reference.always_apply_eligibility = False
    candidate.always_apply_eligibility = True
    with torch.no_grad():
        trace = torch.randn_like(reference.eligibility_trace) * 0.01
        reference.eligibility_trace.copy_(trace)
        candidate.eligibility_trace.copy_(trace)
        reference.eligibility_modulator.fill_(0.2)
        candidate.eligibility_modulator.fill_(0.2)
    x_ref = torch.randn(2, 13, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    energy = torch.rand(2, 13, 1)
    out_ref, surprise_ref = reference(x_ref, energy, reset_state=True)
    out_new, surprise_new = candidate(x_new, energy, reset_state=True)
    torch.testing.assert_close(out_new, out_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(surprise_new, surprise_ref, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(candidate.state, reference.state, atol=2e-6, rtol=2e-5)
    (out_ref.square().mean() + surprise_ref.mean()).backward()
    (out_new.square().mean() + surprise_new.mean()).backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=3e-6, rtol=3e-5)
    _compare_grads(reference, candidate, atol=3e-6, rtol=3e-5)


def test_repetition_loss_shared_log_softmax_matches_reference_and_gradients():
    from train.loss import perdida_total

    torch.manual_seed(37)
    logits_ref = torch.randn(2, 7, 31, requires_grad=True)
    logits_new = logits_ref.detach().clone().requires_grad_(True)
    targets = torch.randint(0, 31, (2, 7))
    activations = torch.randn(2, 7, 8)
    router = torch.rand(2, 7, 1)
    output_ref = {"logits": logits_ref, "activations": activations, "router": router}
    output_new = {"logits": logits_new, "activations": activations, "router": router}

    loss_new, metrics_new = perdida_total(
        output_new, targets, None, use_jepa=False, use_rep=True
    )

    logits_0 = logits_ref[:, :-2]
    logits_1 = logits_ref[:, 1:-1]
    logits_2 = logits_ref[:, 2:]
    reference_log_triple = (
        torch.logsumexp(logits_0 + logits_1 + logits_2, dim=-1)
        - torch.logsumexp(logits_0, dim=-1)
        - torch.logsumexp(logits_1, dim=-1)
        - torch.logsumexp(logits_2, dim=-1)
    )
    loss_ref = reference_log_triple.exp().mean()
    torch.testing.assert_close(metrics_new["L_rep"], loss_ref.detach(), atol=2e-7, rtol=2e-6)
    loss_new.backward()
    loss_ref.backward()
    torch.testing.assert_close(logits_new.grad, logits_ref.grad, atol=2e-7, rtol=2e-6)
