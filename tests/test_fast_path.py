import pytest
import torch

from model.delta_state import EstadoDelta
from train.dataset import collate_token_ids, load_token_cache


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


def test_embedding_usage_histogram_matches_token_counts():
    from model.embedding import VivoEmbedding

    embedding = VivoEmbedding(vocab=8, dim=4)
    tokens = torch.tensor([[1, 1, 2], [0, 2, 3]])
    embedding(tokens)
    expected = torch.tensor([1, 2, 2, 1, 0, 0, 0, 0], dtype=torch.float32)
    torch.testing.assert_close(embedding.uso, expected)
    assert embedding.pasos_desde_poda.item() == 1
