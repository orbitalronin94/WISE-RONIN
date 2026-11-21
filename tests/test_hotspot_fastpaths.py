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
