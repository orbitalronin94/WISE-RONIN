"""Positive reinforcement: bounded response to sustained check-level improvement."""
import torch
from torch import nn


class Recompensa(nn.Module):
    def __init__(self, ema=0.8, deadband=0.005, full_scale=0.08,
                 response=0.35, decay=0.80):
        super().__init__()
        if not 0.0 <= ema < 1.0 or deadband < 0 or full_scale <= deadband:
            raise ValueError("invalid EMA/deadband/full_scale")
        if not 0.0 < response <= 1.0 or not 0.0 <= decay < 1.0:
            raise ValueError("invalid response/decay")
        self.ema, self.deadband, self.full_scale = float(ema), float(deadband), float(full_scale)
        self.response, self.decay = float(response), float(decay)
        self.register_buffer("loss_ema", torch.tensor(float("nan")))
        self.register_buffer("pleasure", torch.tensor(0.0))
        self.register_buffer("last_improvement", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, check_loss):
        current = torch.as_tensor(check_loss, dtype=self.pleasure.dtype, device=self.pleasure.device)
        if current.numel() != 1 or not torch.isfinite(current) or current.item() < 0:
            raise ValueError("pleasure requires one finite non-negative check loss")
        if not torch.isfinite(self.loss_ema):
            self.loss_ema.copy_(current)
            self.last_improvement.zero_()
            return self.state()
        reference = self.loss_ema.clamp_min(1e-8)
        improvement = (reference - current) / reference
        self.last_improvement.copy_(improvement)
        raw = ((improvement - self.deadband) / (self.full_scale - self.deadband)).clamp(0.0, 1.0)
        self.pleasure.mul_(self.decay).add_(raw, alpha=self.response).clamp_(0.0, 1.0)
        self.loss_ema.mul_(self.ema).add_(current, alpha=1.0 - self.ema)
        return self.state()

    @torch.no_grad()
    def update_signal(self, improvement):
        """Consume the trainer's shared normalized improvement signal; no private EMA."""
        value = torch.as_tensor(improvement, dtype=self.pleasure.dtype, device=self.pleasure.device)
        if value.numel() != 1 or not torch.isfinite(value):
            raise ValueError("pleasure shared signal must be one finite value")
        self.last_improvement.copy_(value)
        raw = ((value - self.deadband) / (self.full_scale - self.deadband)).clamp(0.0, 1.0)
        self.pleasure.mul_(self.decay).add_(raw, alpha=self.response).clamp_(0.0, 1.0)
        return self.state()

    @torch.no_grad()
    def state(self):
        value = float(self.pleasure.item())
        return {"pleasure": value, "plasticity_scale": 1.0 + 0.10 * value,
                "gate_scale": 1.0 + 0.05 * value,
                "consolidation_drive": value,
                "improvement": float(self.last_improvement.item())}
