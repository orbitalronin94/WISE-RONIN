"""Homeostatic activation control and per-token top-k sparsity."""
import torch
from torch import nn


class Homeostasis(nn.Module):
    def __init__(self, dim=256, m_target=0.05, eta=0.001, k_top=32):
        super().__init__()
        self.dim, self.m_target, self.eta, self.k_top = dim, m_target, eta, k_top
        self.bias = nn.Parameter(torch.zeros(dim))
        # Input activity is retained as a diagnostic, not as the control signal.
        self.register_buffer("m", torch.zeros(dim))
        # EMA of units that survive the actual ReLU + Top-K output gate.
        self.register_buffer("effective_m", torch.zeros(dim))
        self.last_input_positive = 0.0
        self.last_output_positive = 0.0
        self.last_gate_positive = 0.0
        self.last_output_norm = 0.0
        self.last_gate_activity_per_unit = None

    def forward(self, x, collect_diagnostics=True):
        if collect_diagnostics:
            self.last_input_positive = float((x.detach() > 0).float().mean().item())
        if self.training:
            with torch.no_grad():
                input_active = (x > 0).float().mean(dim=(0, 1))
                self.m.mul_(0.99).add_(input_active, alpha=0.01)

        y = x - self.bias
        k = min(self.k_top, self.dim)
        _, indices = torch.topk(y, k=k, dim=-1)
        mask = torch.zeros_like(y).scatter_(-1, indices, 1.0)
        gated_update = torch.relu(y) * mask
        # Top-K sparsifies the homeostatic update, not the information stream.
        # Bound residual gain to 10% of the incoming token norm; the previous
        # unscaled addition could nearly double selected coordinates per layer.
        update_norm = gated_update.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        max_update_norm = 0.1 * x.norm(dim=-1, keepdim=True)
        gated_update = gated_update * (max_update_norm / update_norm).clamp(max=1.0)
        output = x + gated_update

        if self.training:
            with torch.no_grad():
                # Close the loop around the activity that actually survives
                # ReLU + Top-K. Input positivity is not a valid proxy: the
                # residual stream can remain active while every gated update
                # is suppressed, which otherwise drives the threshold upward
                # indefinitely and kills the sparse update path.
                effective_active = (gated_update > 0).float().mean(dim=(0, 1))
                self.effective_m.mul_(0.99).add_(effective_active, alpha=0.01)
                self.bias.add_(self.eta * (self.effective_m - self.m_target))
        else:
            effective_active = (gated_update.detach() > 0).float().mean(dim=(0, 1))
        # The health window needs per-unit activity each step, but scalar
        # summaries are only materialized on requested diagnostic steps.
        self.last_gate_activity_per_unit = effective_active.detach()
        if collect_diagnostics:
            self.last_gate_positive = float(effective_active.mean().item())
            self.last_output_positive = float((output.detach() > 0).float().mean().item())
            self.last_output_norm = float(
                output.detach().norm().item()
                / max(1, (output.shape[0] * output.shape[1]) ** 0.5)
            )
        return output

    def neuronas_muertas(self):
        """Fraction of units with no sustained post-ReLU/Top-K activity."""
        return (self.effective_m < 0.001).float().mean().item()
