import torch

from scripts.validate_step1000_parent import tensor_hash


def test_tensor_hash_supports_scalar_and_non_scalar_tensors():
    state = {
        "scalar_float": torch.tensor(1.25),
        "scalar_int": torch.tensor(7, dtype=torch.int64),
        "vector": torch.tensor([1.0, 2.0]),
    }
    digest = tensor_hash(state)
    assert len(digest) == 64
    assert digest == tensor_hash(state)
    changed = dict(state)
    changed["scalar_float"] = torch.tensor(1.5)
    assert tensor_hash(changed) != digest
