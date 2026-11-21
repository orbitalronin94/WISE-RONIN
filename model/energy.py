"""Per-token energy allocation across delta, memory and output heads."""
import torch
from torch import nn

class Presupuesto(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Linear(dim, 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        e = torch.softmax(self.proj(x), dim=-1)
        return e[..., 0:1], e[..., 1:2], e[..., 2:3], e

    @staticmethod
    def entropy(weights):
        return -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(-1).mean()
