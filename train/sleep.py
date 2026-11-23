"""Sleep-time replay over consolidated records with a deterministic 1/3/7 spacing."""
import logging
import torch
from torch.nn import functional as F

LOG = logging.getLogger("lumen.sleep")


def sueno(model, pasos=None, lr=1e-4, device="cpu", compression_ratio=1.0, downscale=1.0):
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
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    count = max(1, int(valid_indices.numel() * float(compression_ratio)))
    # Prioritized compressed replay: retain the most-used consolidated slots.
    priorities = memory.usage_ema[valid_indices]
    chosen = torch.topk(priorities, k=min(count, priorities.numel()), sorted=True).indices
    selected_indices = valid_indices[chosen]
    targets = memory.lenta[selected_indices].detach().to(device)
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

    if downscale < 1.0:
        # Bounded global downscaling applies only to replay-MLP weight matrices;
        # the optimizer is local to this sleep phase and is discarded afterwards.
        with torch.no_grad():
            for parameter in memory.mlp.parameters():
                if parameter.ndim >= 2:
                    parameter.mul_(float(downscale))
    model.reset_states()
    memory.reset_elastica()
    return total / max(1, updates)
