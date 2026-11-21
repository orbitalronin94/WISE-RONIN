"""Offline consolidation sleep; only elastic-memory parameters are optimized."""
import logging
import torch
from torch.nn import functional as F

LOG = logging.getLogger("lumen.sleep")

def sueno(model, pasos=50, lr=1e-4, device="cpu"):
    memory = model.memoria
    valid = memory.contador_lenta > 0
    if not valid.any():
        LOG.warning("Sleep skipped: slow memory has no consolidated records")
        model.reset_states()
        memory.reset_elastica()
        return 0.0
    target_bank = memory.lenta[valid].detach()
    optimizer = torch.optim.AdamW(memory.mlp.parameters(), lr=lr)
    total = 0.0
    for _ in range(pasos):
        indices = torch.randint(0, target_bank.size(0), (8, 16), device=device)
        target = target_bank[indices]
        noise = torch.randn_like(target)
        pred = memory.mlp(noise)
        loss = F.mse_loss(pred, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
    model.reset_states()
    memory.reset_elastica()
    return total / max(1, pasos)
