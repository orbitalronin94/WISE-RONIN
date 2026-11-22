import pytest
import torch

from model.delta_state import EstadoDelta
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from train.dataset import BPETokenizer, collate_token_ids, load_token_cache


def test_pre_tokenized_collation_and_cache_validation(tmp_path):
    rows = [[2, 6, 3], [2, 5, 3], [2, 3]]
    batch = collate_token_ids(rows)
    assert batch.tolist() == [[2, 6, 3], [2, 5, 3], [2, 3, 0]]
    path = tmp_path / "tokens.pt"
    torch.save({
        "format": "ronin_token_cache_v2",
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



def _tiny_bpe(tmp_path):
    special = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[SEP]"]
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        min_frequency=1,
        special_tokens=special,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        [
            "Pregunta: ¿Qué significa memoria asociativa?",
            "Contexto: El modelo procesa secuencias en español.",
            "Respuesta: La memoria conserva información útil.",
        ] * 20,
        trainer=trainer,
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    return BPETokenizer(path)


def test_long_prompt_and_completion_keep_bos_sep_eos(tmp_path):
    tokenizer = _tiny_bpe(tmp_path)
    prompt = "Contexto: " + "información contextual importante " * 80 + "Pregunta: ¿Qué debe responder?"
    completion = "Respuesta: " + "La respuesta conserva una explicación verificable. " * 80
    ids = tokenizer.encode_prompt_completion(prompt, completion, max_len=32)

    assert len(ids) <= 32
    assert ids[0] == 2
    assert ids[-1] == 3
    sep_index = ids.index(4)
    assert sep_index > 1
    assert len(ids) - sep_index > 2
    assert 0 not in ids


def test_empty_prompt_language_model_text_still_keeps_eos(tmp_path):
    tokenizer = _tiny_bpe(tmp_path)
    ids = tokenizer.encode_prompt_completion("", "Texto español de continuación. " * 80, max_len=16)
    assert len(ids) == 16
    assert ids[0] == 2
    assert ids[-1] == 3
    assert 4 not in ids


def test_plain_encode_never_truncates_eos(tmp_path):
    tokenizer = _tiny_bpe(tmp_path)
    ids = tokenizer.encode("Una secuencia muy larga. " * 80, max_len=12)
    assert len(ids) == 12
    assert ids[0] == 2 and ids[-1] == 3


def test_legacy_token_cache_is_rejected(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save({"format": "ronin_token_cache_v1"}, path)
    with pytest.raises(ValueError, match="Unsupported token cache format"):
        load_token_cache(path, "data", "tokenizer", 128, 1)
