"""Sleep-time replay over consolidated records with a deterministic 1/3/7 spacing."""
import logging
import torch
from torch.nn import functional as F

LOG = logging.getLogger("lumen.sleep")


def sueno(model, pasos=None, lr=1e-4, device="cpu"):
    memory = model.memoria
    valid = memory.contador_lenta > 0
    if not valid.any():
        LOG.warning("Sleep skipped: slow memory has no consolidated records")
        model.reset_states()
        memory.reset_elastica()
        return 0.0

    targets = memory.lenta[valid].detach().to(device)
    optimizer = torch.optim.AdamW(memory.mlp.parameters(), lr=lr)
    # Cumulative replay offsets: first replay at 1, then +3, then +7 ticks.
    replay_ticks = {1, 4, 11}
    total, updates = 0.0, 0
    for tick in range(1, 12):
        if tick not in replay_ticks:
            continue
        indices = torch.arange(targets.size(0), device=device)
        target = targets[indices]
        noise = torch.randn_like(target)
        prediction = memory.mlp(noise)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())
        updates += 1

    model.reset_states()
    memory.reset_elastica()
    return total / max(1, updates)
