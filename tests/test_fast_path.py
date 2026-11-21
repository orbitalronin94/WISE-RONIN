import pytest
import torch

from model.delta_state import EstadoDelta
from train.dataset import (
    collate_token_ids, collate_cached_indices, load_token_cache, pack_token_cache,
)


def test_pre_tokenized_collation_and_cache_validation(tmp_path):
    rows = [[1, 2, 3], [4, 5], [6]]
    batch = collate_token_ids(rows)
    assert batch.tolist() == [[1, 2, 3], [4, 5, 0]]
    path = tmp_path / "tokens.pt"
    torch.save({
        "format": "ronin_token_cache_v1",
        "data_sha256": "data",
        "tokenizer_sha256": "tokenizer",
        "max_len": 128,
        "row_count": 3,
        "encoded_rows": rows,
    }, path)
    assert load_token_cache(path, "data", "tokenizer", 128, 3) == rows
    with pytest.raises(ValueError, match="metadata mismatch"):
        load_token_cache(path, "wrong-data", "tokenizer", 128, 3)


def test_delta_v31_fast_path_preserves_outputs_and_gradients():
    torch.manual_seed(7)
    reference = EstadoDelta(dim=8, tau=0.001)
    optimized = EstadoDelta(dim=8, tau=0.001)
    optimized.load_state_dict(reference.state_dict())
    x_ref = torch.randn(2, 6, 8, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    energy = torch.full((2, 6, 1), 0.4)

    out_ref, surprise_ref = reference(x_ref, energy, reset_state=True, compute_complexity=True)
    out_opt, surprise_opt = optimized(x_opt, energy, reset_state=True, compute_complexity=False)

    torch.testing.assert_close(out_opt, out_ref)
    torch.testing.assert_close(surprise_opt, surprise_ref)
    assert optimized.last_complexity is None
    assert reference.last_complexity is not None

    out_ref.square().mean().backward()
    out_opt.square().mean().backward()
    torch.testing.assert_close(x_opt.grad, x_ref.grad)
    for (name_ref, param_ref), (name_opt, param_opt) in zip(
        reference.named_parameters(), optimized.named_parameters()
    ):
        assert name_ref == name_opt
        if param_ref.grad is not None:
            torch.testing.assert_close(param_opt.grad, param_ref.grad)


def test_packed_token_cache_matches_reference_collation():
    rows = [[1, 2, 3], [4, 5], [6], [7, 8, 9, 10]]
    packed, lengths = pack_token_cache(rows, max_len=3)
    for indices in ([0, 1, 2], [3, 0], [2, 1, 2]):
        expected = collate_token_ids([rows[index][:3] for index in indices])
        actual = collate_cached_indices(packed, lengths, indices)
        torch.testing.assert_close(actual, expected)


def test_delta_batched_projection_matches_tokenwise_reference():
    torch.manual_seed(19)
    reference = EstadoDelta(dim=8, tau=0.001)
    optimized = EstadoDelta(dim=8, tau=0.001)
    optimized.load_state_dict(reference.state_dict())
    x_ref = torch.randn(2, 7, 8, requires_grad=True)
    x_opt = x_ref.detach().clone().requires_grad_(True)
    energy = torch.full((2, 7, 1), 0.4)

    state = torch.zeros(2, 8, 8)
    ref_out, ref_surprise = [], []
    for position in range(x_ref.shape[1]):
        xt = x_ref[:, position]
        k = reference.proj_k(xt)
        v = reference.proj_v(xt)
        q = reference.proj_q(xt)
        pred = torch.einsum("bi,bij->bj", k, state)
        err = v - pred
        surprise = err.norm(dim=-1, keepdim=True) / (8 ** 0.5)
        decay = torch.sigmoid(reference.a * surprise + reference.b).clamp(0.90, 0.9999)
        write_lr = surprise.clamp(0.1, 1.0)
        gate = (surprise >= reference.tau.clamp_min(0.0)).to(x_ref.dtype)
        state = state * decay.unsqueeze(-1)
        state = state + torch.einsum("bi,bj->bij", k, err) * (
            write_lr * gate * energy[:, position]
        ).unsqueeze(-1)
        ref_out.append(torch.einsum("bi,bij->bj", q, state))
        ref_surprise.append(surprise)
    expected = torch.stack(ref_out, dim=1)
    expected_surprise = torch.stack(ref_surprise, dim=1)

    actual, actual_surprise = optimized(
        x_opt, energy, reset_state=True, compute_complexity=False,
        collect_diagnostics=False,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_surprise, expected_surprise, rtol=1e-5, atol=1e-6)

    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(x_opt.grad, x_ref.grad, rtol=1e-5, atol=1e-6)
    for (name_ref, param_ref), (name_opt, param_opt) in zip(
        reference.named_parameters(), optimized.named_parameters()
    ):
        assert name_ref == name_opt
        if param_ref.grad is not None:
            torch.testing.assert_close(param_opt.grad, param_ref.grad, rtol=1e-5, atol=1e-6)
