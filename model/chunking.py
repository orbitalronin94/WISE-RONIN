"""Adaptive sequence chunking driven by prediction surprise.

The controller changes only the granularity of a bounded residual summary.
It never removes an organ or replaces the token-level information stream.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveChunking(nn.Module):
    def __init__(self, dim=256, min_chunk=4, max_chunk=16, gain=0.025,
                 surprise_scale=0.01):
        super().__init__()
        if dim < 1 or min_chunk < 1 or max_chunk < min_chunk:
            raise ValueError("invalid adaptive chunking dimensions")
        if gain < 0 or surprise_scale <= 0:
            raise ValueError("gain must be non-negative and surprise_scale positive")
        self.dim = int(dim)
        self.min_chunk = int(min_chunk)
        self.max_chunk = int(max_chunk)
        self.gain = float(gain)
        self.surprise_scale = float(surprise_scale)
        self.last_chunk_size = self.max_chunk
        self.last_chunk_count = 0

    def choose_chunk_size(self, surprise):
        score = float(surprise.detach().float().mean().item())
        volatility = score / (score + self.surprise_scale)
        span = self.max_chunk - self.min_chunk
        # More surprise -> shorter chunks and more frequent boundaries.
        return max(self.min_chunk, min(
            self.max_chunk,
            int(round(self.max_chunk - volatility * span)),
        ))

    def forward(self, x, surprise):
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError("chunking expects [batch, time, dim]")
        if surprise.shape[:2] != x.shape[:2]:
            raise ValueError("surprise must match batch/time dimensions")
        steps = x.shape[1]
        chunk_size = self.choose_chunk_size(surprise)
        chunks = max(1, math.ceil(steps / chunk_size))
        self.last_chunk_size = chunk_size
        self.last_chunk_count = chunks

        pooled = F.adaptive_avg_pool1d(x.transpose(1, 2), chunks)
        context = F.interpolate(pooled, size=steps, mode="nearest").transpose(1, 2)
        # Parameter-free similarity gate: keep the matched-size comparison fair.
        similarity = (x * context).sum(dim=-1, keepdim=True) / math.sqrt(self.dim)
        gate = torch.sigmoid(similarity)
        update = self.gain * gate * context
        # Keep the new path bounded relative to the original representation.
        update_norm = update.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        max_norm = self.gain * x.norm(dim=-1, keepdim=True)
        update = update * (max_norm / update_norm).clamp(max=1.0)
        return x + update
