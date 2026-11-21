"""Homeostatic activation control and per-token top-k sparsity."""
import torch
from torch import nn

class Homeostasis(nn.Module):
    def __init__(self, dim=256, m_target=0.05, eta=0.001, k_top=32):
        super().__init__()
        self.dim, self.m_target, self.eta, self.k_top = dim, m_target, eta, k_top
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("m", torch.zeros(dim))
        # EMA of units that survive the actual ReLU + Top-K output gate.
        self.register_buffer("effective_m", torch.zeros(dim))
        self.last_input_positive = 0.0
        self.last_output_positive = 0.0
        self.last_gate_positive = 0.0
        self.last_output_norm = 0.0

    def forward(self, x):
        self.last_input_positive = float((x.detach() > 0).float().mean().item())
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
        gated_update = torch.relu(y) * mask
        # Top-K sparsifies the homeostatic update, not the information stream.
        # Returning only gated_update erased most dimensions at every layer.
        output = x + gated_update
        self.last_gate_positive = float((gated_update.detach() > 0).float().mean().item())
        self.last_output_positive = float((output.detach() > 0).float().mean().item())
        self.last_output_norm = float(output.detach().norm().item() / max(1, (output.shape[0] * output.shape[1]) ** 0.5))
        if self.training:
            with torch.no_grad():
                effective_active = (gated_update > 0).float().mean(dim=(0, 1))
                self.effective_m.mul_(0.99).add_(effective_active, alpha=0.01)
        return output

    def neuronas_muertas(self):
        """Fraction of units with no sustained post-ReLU/Top-K activity."""
        return (self.effective_m < 0.001).float().mean().item()
