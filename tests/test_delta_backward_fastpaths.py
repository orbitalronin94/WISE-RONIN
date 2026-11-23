import copy

import torch

from model.delta_state import EstadoDelta


def test_delta_exact_fastpaths_preserve_outputs_state_and_gradients():
    torch.manual_seed(2026)
    reference = EstadoDelta(dim=16, tau=0.01, rank=8, parallel_refinements=3)
    optimized = copy.deepcopy(reference)
    optimized.trace_summary_no_grad = True
    optimized.trace_norm_diagnostics = False
    optimized.state_update_bmm = True
    optimized.summary_update_bmm = True

    x_ref = torch.randn(2, 9, 16, requires_grad=True)
    e_ref = torch.rand(2, 9, 1, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    e_opt = e_ref.detach().clone().requires_grad_(True)

    out_ref, surprise_ref = reference(x_ref, e_ref, reset_state=True, collect_diagnostics=True)
    out_opt, surprise_opt = optimized(x_opt, e_opt, reset_state=True, collect_diagnostics=False)

    torch.testing.assert_close(out_opt, out_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(surprise_opt, surprise_ref, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(optimized.state, reference.state, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        optimized.eligibility_trace, reference.eligibility_trace,
        rtol=1e-5, atol=1e-6,
    )

    (out_ref.square().mean() + surprise_ref.square().mean()).backward()
    (out_opt.square().mean() + surprise_opt.square().mean()).backward()
    torch.testing.assert_close(x_opt.grad, x_ref.grad, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(e_opt.grad, e_ref.grad, rtol=2e-5, atol=2e-6)

    ref_params = dict(reference.named_parameters())
    opt_params = dict(optimized.named_parameters())
    assert ref_params.keys() == opt_params.keys()
    for name in ref_params:
        if ref_params[name].grad is None:
            assert opt_params[name].grad is None, name
        else:
            torch.testing.assert_close(
                opt_params[name].grad, ref_params[name].grad,
                rtol=2e-5, atol=2e-6, msg=name,
            )
