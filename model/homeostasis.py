"""Homeostatic activation control and per-token top-k sparsity."""
import torch
from torch import nn


class Homeostasis(nn.Module):
    def __init__(self, dim=256, m_target=0.05, eta=0.001, k_top=32):
        super().__init__()
        self.dim, self.m_target, self.eta, self.k_top = dim, m_target, eta, k_top
        # Optional fused residual scatter path; parity-checked against the reference path.
        self.fused_scatter_add = False
        self.synaptic_scaling_enabled = False
        self.bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer("synaptic_scale", torch.ones(dim))
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
        # Event-driven Top-K path: only gather and apply ReLU to selected
        # coordinates. The previous dense ReLU + dense mask multiplication
        # performed elementwise work over all dim coordinates even though
        # only k_top coordinates can contribute to the update.
        _, indices = torch.topk(y, k=k, dim=-1, sorted=False)
        selected = torch.gather(y, dim=-1, index=indices)
        selected_update = torch.relu(selected)
        if self.synaptic_scaling_enabled:
            selected_update = selected_update * self.synaptic_scale[indices]
        # Top-K sparsifies the homeostatic update, not the information stream.
        # Bound residual gain to 10% of the incoming token norm.
        if self.fused_scatter_add:
            update_norm = selected_update.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            max_update_norm = 0.1 * x.norm(dim=-1, keepdim=True)
            scaled_selected = selected_update * (max_update_norm / update_norm).clamp(max=1.0)
            output = x.scatter_add(-1, indices, scaled_selected)
            # Exact per-unit activity counts without materializing a dense gated-update tensor.
            effective_active = torch.bincount(
                indices.reshape(-1),
                weights=(scaled_selected.detach() > 0).to(x.dtype).reshape(-1),
                minlength=self.dim,
            ).to(x.dtype).div_(x.shape[0] * x.shape[1])
        else:
            gated_update = torch.zeros_like(y).scatter_(-1, indices, selected_update)
            update_norm = gated_update.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            max_update_norm = 0.1 * x.norm(dim=-1, keepdim=True)
            gated_update = gated_update * (max_update_norm / update_norm).clamp(max=1.0)
            output = x + gated_update
            effective_active = (gated_update.detach() > 0).float().mean(dim=(0, 1))

        if self.training:
            with torch.no_grad():
                # Close the loop around activity that survives ReLU + Top-K.
                self.effective_m.mul_(0.99).add_(effective_active, alpha=0.01)
                self.bias.add_(self.eta * (self.effective_m - self.m_target))
                if self.synaptic_scaling_enabled:
                    # Bounded per-channel multiplicative gain scaling. This is
                    # an engineering analogue, not literal biological synapses.
                    self.synaptic_scale.mul_(
                        1.0 + self.eta * (self.m_target - effective_active)
                    ).clamp_(0.8, 1.2)
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
