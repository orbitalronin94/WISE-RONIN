"""Sleep-time replay over consolidated records with a deterministic 1/3/7 spacing."""
import logging
import torch
from torch.nn import functional as F

LOG = logging.getLogger("lumen.sleep")


def sueno(model, pasos=None, lr=1e-4, device="cpu", compression_ratio=1.0, downscale=1.0):
    """Replay consolidated slow memories.

    With pasos=None, preserve the original three-update 1/3/7 replay schedule.
    With pasos=N, perform exactly N optimizer updates over the same prioritized
    consolidated-memory targets. This makes longer offline sleep experiments
    explicit without changing ordinary training-time sleep behavior.
    """
    memory = model.memoria
    valid = memory.contador_lenta > 0
    if not valid.any():
        LOG.warning("Sleep skipped: slow memory has no consolidated records")
        model.reset_states()
        memory.reset_elastica()
        return 0.0

    if not 0.1 <= float(compression_ratio) <= 1.0:
        raise ValueError("replay compression ratio must be in [0.1,1]")
    if not 0.90 <= float(downscale) <= 1.0:
        raise ValueError("sleep downscale must be in [0.90,1]")
    if pasos is not None and int(pasos) < 1:
        raise ValueError("pasos must be a positive integer")

    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    count = max(1, int(valid_indices.numel() * float(compression_ratio)))
    priorities = memory.usage_ema[valid_indices]
    chosen = torch.topk(priorities, k=min(count, priorities.numel()), sorted=True).indices
    selected_indices = valid_indices[chosen]
    targets = memory.lenta[selected_indices].detach().to(device)
    optimizer = torch.optim.AdamW(memory.mlp.parameters(), lr=lr)

    # Legacy sleep retains the cumulative 1/3/7 spacing. Extended sleep uses
    # the exact requested number of replay optimizer updates.
    update_count = 3 if pasos is None else int(pasos)
    total = 0.0
    for update in range(update_count):
        target = targets
        noise = torch.randn_like(target)
        prediction = memory.mlp(noise)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += float(loss.detach())

    if downscale < 1.0:
        with torch.no_grad():
            for parameter in memory.mlp.parameters():
                if parameter.ndim >= 2:
                    parameter.mul_(float(downscale))
    model.reset_states()
    memory.reset_elastica()
    return total / max(1, update_count)
