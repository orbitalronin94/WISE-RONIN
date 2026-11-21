import torch
from model.config import CONFIG
from model.lumen import RoninLumen

def test_forward_shapes_and_cpu_backward():
    model=RoninLumen()
    x=torch.randint(4,CONFIG.vocab_size,(2,16))
    out=model(x,reset_state=True)
    assert out["logits"].shape==(2,16,CONFIG.vocab_size)
    assert out["router"].shape==(2,16,2)
    assert out["activations"].shape==(2,16,CONFIG.model_dim)
    assert out["energy"].shape==(2,16,3)
    loss=torch.nn.functional.cross_entropy(out["logits"][:,:-1].reshape(-1,CONFIG.vocab_size),x[:,1:].reshape(-1))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

def test_parameter_budget_and_tied_embeddings():
    model=RoninLumen()
    assert sum(p.numel() for p in model.parameters())<=6_000_000
    assert model.head_out.weight.data_ptr()==model.emb.emb.weight.data_ptr()

def test_delta_state_reset_is_empty():
    model=RoninLumen()
    model(torch.randint(4,CONFIG.vocab_size,(1,8)))
    assert model.delta.state.numel()>0
    model.reset_states()
    assert model.delta.state.numel()==0
