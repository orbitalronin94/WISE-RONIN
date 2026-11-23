import copy

import torch

from model.chunking import AdaptiveChunking
from model.delta_state import EstadoDelta
from model.homeostasis import Homeostasis
from model.memory import MemoriaElastica


def _compare_grads(left, right, atol=3e-6, rtol=3e-5):
    for (name_l, param_l), (name_r, param_r) in zip(
        left.named_parameters(), right.named_parameters()
    ):
        assert name_l == name_r
        if param_l.grad is None or param_r.grad is None:
            assert param_l.grad is None and param_r.grad is None
        else:
            torch.testing.assert_close(param_l.grad, param_r.grad, atol=atol, rtol=rtol)


def _delta_pair():
    torch.manual_seed(29)
    reference = EstadoDelta(dim=32, rank=12, parallel_refinements=2)
    candidate = copy.deepcopy(reference)
    candidate.reuse_write_update = True
    candidate.always_apply_eligibility = True
    candidate.trace_norm_diagnostics_on_demand = True
    candidate.share_causal_masks = True
    # Exercise a nonzero delayed signal and nonempty eligibility trace.
    with torch.no_grad():
        trace = torch.randn_like(reference.eligibility_trace) * 0.01
        reference.eligibility_trace.copy_(trace)
        candidate.eligibility_trace.copy_(trace)
        reference.set_eligibility_modulator(0.35)
        candidate.set_eligibility_modulator(0.35)
    return reference, candidate


def test_delta_simplifications_preserve_outputs_states_and_gradients():
    reference, candidate = _delta_pair()
    x_ref = torch.randn(2, 13, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    energy = torch.rand(2, 13, 1)
    out_ref, surprise_ref = reference(x_ref, energy, reset_state=True, collect_diagnostics=True)
    out_new, surprise_new = candidate(x_new, energy, reset_state=True, collect_diagnostics=True)
    torch.testing.assert_close(out_new, out_ref, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(surprise_new, surprise_ref, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(candidate.state, reference.state, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(candidate.eligibility_trace, reference.eligibility_trace, atol=3e-6, rtol=3e-5)
    assert candidate.last_eligibility_norm == reference.last_eligibility_norm
    (out_ref.square().mean() + surprise_ref.mean()).backward()
    (out_new.square().mean() + surprise_new.mean()).backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=3e-6, rtol=3e-5)
    _compare_grads(reference, candidate)


def test_delta_on_demand_diagnostics_preserve_requested_samples():
    reference, candidate = _delta_pair()
    x = torch.randn(2, 9, 32)
    energy = torch.ones(2, 9, 1)
    reference(x, energy, reset_state=True, collect_diagnostics=False)
    candidate(x, energy, reset_state=True, collect_diagnostics=False)
    # Both expose the most recent sampled value on a non-diagnostic call.
    assert candidate.last_eligibility_norm == 0.0
    assert reference.last_eligibility_norm > 0.0
    reference(x, energy, reset_state=True, collect_diagnostics=True)
    candidate(x, energy, reset_state=True, collect_diagnostics=True)
    assert abs(candidate.last_eligibility_norm - reference.last_eligibility_norm) < 1e-7


def test_homeostasis_reused_activity_reduction_preserves_function():
    torch.manual_seed(31)
    reference = Homeostasis(dim=32, k_top=8)
    candidate = copy.deepcopy(reference)
    candidate.reuse_input_activity = True
    x_ref = torch.randn(3, 11, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    out_ref = reference(x_ref, collect_diagnostics=True)
    out_new = candidate(x_new, collect_diagnostics=True)
    torch.testing.assert_close(out_new, out_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate.m, reference.m, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(candidate.effective_m, reference.effective_m, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(candidate.bias, reference.bias, atol=1e-7, rtol=1e-6)
    assert abs(candidate.last_input_positive - reference.last_input_positive) < 1e-7
    out_ref.square().mean().backward()
    out_new.square().mean().backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    _compare_grads(reference, candidate, atol=1e-6, rtol=1e-6)


def test_memory_bmm_read_preserves_outputs_and_gradients():
    torch.manual_seed(37)
    reference = MemoriaElastica(dim=32, hidden=12, slots=9)
    candidate = copy.deepcopy(reference)
    candidate.bmm_read = True
    with torch.no_grad():
        reference.contador_lenta[:5].fill_(1)
        reference.lenta[:5].normal_()
        candidate.contador_lenta.copy_(reference.contador_lenta)
        candidate.lenta.copy_(reference.lenta)
    reference.eval()
    candidate.eval()
    x_ref = torch.randn(2, 7, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    surprise = torch.full((2, 7, 1), 0.5)
    energy = torch.rand(2, 7, 1)
    out_ref = reference(x_ref, surprise, energy)
    out_new = candidate(x_new, surprise, energy)
    torch.testing.assert_close(out_new, out_ref, atol=2e-6, rtol=2e-5)
    out_ref.square().mean().backward()
    out_new.square().mean().backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=2e-6, rtol=2e-5)
    _compare_grads(reference, candidate, atol=2e-6, rtol=2e-5)


def test_chunk_nearest_gather_matches_interpolate_outputs_and_gradients():
    torch.manual_seed(41)
    reference = AdaptiveChunking(dim=32, min_chunk=2, max_chunk=8)
    candidate = copy.deepcopy(reference)
    candidate.nearest_gather = True
    x_ref = torch.randn(2, 17, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    surprise = torch.rand(2, 17, 1)
    out_ref = reference(x_ref, surprise)
    out_new = candidate(x_new, surprise)
    torch.testing.assert_close(out_new, out_ref, atol=1e-6, rtol=1e-6)
    out_ref.square().mean().backward()
    out_new.square().mean().backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=2e-6, rtol=2e-5)

def test_homeostasis_combined_scatter_and_activity_reuse_preserve_function():
    torch.manual_seed(43)
    reference = Homeostasis(dim=32, k_top=8)
    candidate = copy.deepcopy(reference)
    candidate.fused_scatter_add = True
    candidate.reuse_input_activity = True
    x_ref = torch.randn(3, 11, 32, requires_grad=True)
    x_new = x_ref.detach().clone().requires_grad_(True)
    out_ref = reference(x_ref, collect_diagnostics=True)
    out_new = candidate(x_new, collect_diagnostics=True)
    torch.testing.assert_close(out_new, out_ref, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(candidate.m, reference.m, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(candidate.effective_m, reference.effective_m, atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(candidate.bias, reference.bias, atol=1e-7, rtol=1e-6)
    assert abs(candidate.last_input_positive - reference.last_input_positive) < 1e-7
    out_ref.square().mean().backward()
    out_new.square().mean().backward()
    torch.testing.assert_close(x_new.grad, x_ref.grad, atol=1e-6, rtol=1e-6)
    _compare_grads(reference, candidate, atol=1e-6, rtol=1e-6)
