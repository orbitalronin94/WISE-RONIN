import torch
from model.memory import MemoriaElastica


def test_slow_memory_does_not_mutate_during_eval():
    memory = MemoriaElastica(dim=8, hidden=4, slots=4)
    memory.registrar_nueva(torch.ones(8))
    memory.eval()
    before = {
        "lenta": memory.lenta.clone(),
        "contador": memory.contador_lenta.clone(),
        "uso": memory.uso_slot.clone(),
        "ema": memory.usage_ema.clone(),
        "pointer": memory.write_pointer.clone(),
        "decay": memory.tokens_since_decay.clone(),
    }
    x = torch.randn(2, 5, 8)
    surprise = torch.ones(2, 5, 1) * 10.0
    energy = torch.ones(2, 5, 1)
    memory(x, surprise, energy)
    torch.testing.assert_close(memory.lenta, before["lenta"])
    torch.testing.assert_close(memory.contador_lenta, before["contador"])
    torch.testing.assert_close(memory.uso_slot, before["uso"])
    torch.testing.assert_close(memory.usage_ema, before["ema"])
    torch.testing.assert_close(memory.write_pointer, before["pointer"])
    torch.testing.assert_close(memory.tokens_since_decay, before["decay"])


def test_slow_memory_can_still_write_during_training():
    memory = MemoriaElastica(dim=8, hidden=4, slots=4)
    memory.train()
    x = torch.randn(1, 5, 8)
    surprise = torch.ones(1, 5, 1) * 10.0
    energy = torch.ones(1, 5, 1)
    memory(x, surprise, energy)
    assert memory.contador_lenta.sum().item() > 0
