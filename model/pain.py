"""Nociception: bounded protection signal driven by sustained loss regressions.

This organ is independent from glucose. It observes check-level loss (not
minibatch loss), keeps its own EMA state, and exposes bounded coefficients.
It does not change the optimizer or loss.
"""
import torch
from torch import nn


class Nociception(nn.Module):
    def __init__(self, ema=0.8, deadband=0.005, full_scale=0.08,
                 response=0.35, decay=0.85):
        super().__init__()
        if not 0.0 <= ema < 1.0:
            raise ValueError("ema must be in [0, 1)")
        if deadband < 0 or full_scale <= deadband:
            raise ValueError("full_scale must exceed non-negative deadband")
        if not 0.0 < response <= 1.0 or not 0.0 <= decay < 1.0:
            raise ValueError("invalid response or decay")
        self.ema, self.deadband, self.full_scale = float(ema), float(deadband), float(full_scale)
        self.response, self.decay = float(response), float(decay)
        self.register_buffer("loss_ema", torch.tensor(float("nan")))
        self.register_buffer("pain", torch.tensor(0.0))
        self.register_buffer("last_regression", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, check_loss):
        current = torch.as_tensor(check_loss, dtype=self.pain.dtype, device=self.pain.device)
        if current.numel() != 1 or not torch.isfinite(current) or current.item() < 0:
            raise ValueError("nociception requires one finite non-negative check loss")
        if not torch.isfinite(self.loss_ema):
            self.loss_ema.copy_(current)
            self.last_regression.zero_()
            return self.state()
        reference = self.loss_ema.clamp_min(1e-8)
        regression = (current - reference) / reference
        self.last_regression.copy_(regression)
        raw = ((regression - self.deadband) / (self.full_scale - self.deadband)).clamp(0.0, 1.0)
        self.pain.mul_(self.decay).add_(raw, alpha=self.response).clamp_(0.0, 1.0)
        self.loss_ema.mul_(self.ema).add_(current, alpha=1.0 - self.ema)
        return self.state()

    @torch.no_grad()
    def state(self):
        pain = float(self.pain.item())
        return {
            "pain": pain,
            "plasticity_scale": 1.0 - 0.15 * pain,
            "gate_scale": 1.0 - 0.10 * pain,
            "consolidation_brake": pain,
            "regression": float(self.last_regression.item()),
        }
