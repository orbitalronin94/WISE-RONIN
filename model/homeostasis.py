"""Homeostatic activation control and per-token top-k sparsity."""
import torch
from torch import nn

class Homeostasis(nn.Module):
    def __init__(self, dim=256, m_target=0.05, eta=0.001, k_top=32):
        super().__init__()
        self.dim, self.m_target, self.eta, self.k_top = dim, m_target, eta, k_top
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("m", torch.zeros(dim))

    def forward(self, x):
        # Homeostatic adaptation is plasticity: update only during training.
        # Evaluation must use frozen statistics and thresholds across prompts.
        if self.training:
            with torch.no_grad():
                active = (x > 0).float().mean(dim=(0, 1))
                self.m.mul_(0.99).add_(active, alpha=0.01)
                # Negative feedback: frequently active units receive a larger subtractive threshold.
                self.bias.add_(self.eta * (self.m - self.m_target))
        y = x - self.bias
        k = min(self.k_top, self.dim)
        values, indices = torch.topk(y, k=k, dim=-1)
        mask = torch.zeros_like(y).scatter_(-1, indices, 1.0)
        return torch.relu(y) * mask

    def neuronas_muertas(self):
        return (self.m < 0.001).float().mean().item()
