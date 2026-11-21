"""Surprise-gated recurrent Delta memory with per-token state-complexity diagnostics."""
import torch
from torch import nn
from torch.nn import functional as F


class EstadoDelta(nn.Module):
    def __init__(self, dim=256, tau=0.15):
        super().__init__()
        self.dim = dim
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.a = nn.Parameter(torch.tensor(-1.0))
        self.b = nn.Parameter(torch.tensor(2.0))
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        self.register_buffer("state", torch.empty(0), persistent=True)
        self.last_complexity = None
        self.last_write_rate = 0.0
        self.last_state_norm = 0.0
        self.last_output_norm = 0.0

    def reset(self, batch, device, dtype):
        self.state = torch.zeros(batch, self.dim, self.dim, device=device, dtype=dtype)

    def forward(self, x, e_delta, reset_state=False, compute_complexity=False, collect_diagnostics=True):
        batch, steps, _ = x.shape
        if (reset_state or self.state.numel() == 0
                or self.state.shape[0] != batch or self.state.device != x.device):
            self.reset(batch, x.device, x.dtype)
        S = self.state.to(dtype=x.dtype)
        # K/V/Q projections are token-independent; fuse them into one sequence-wide GEMM.
        projected = F.linear(x, torch.cat(
            (self.proj_k.weight, self.proj_v.weight, self.proj_q.weight), dim=0
        ))
        k_all, v_all, q_all = projected.chunk(3, dim=-1)
        outs, surprises = [], []
        complexities, write_rates = [], []
        for t in range(steps):
            if compute_complexity:
                previous_log_prob = torch.log_softmax(S.float(), dim=-1)
            k, v, q = k_all[:, t], v_all[:, t], q_all[:, t]
            # Unit-norm keys prevent recurrent reads scaling with upstream magnitude.
            k_unit = F.normalize(k, dim=-1, eps=1e-6)
            pred = torch.bmm(k_unit.unsqueeze(1), S).squeeze(1)
            err = v - pred
            surprise = err.norm(dim=-1, keepdim=True) / (self.dim ** 0.5)
            decay = torch.sigmoid(self.a * surprise + self.b).clamp(0.90, 0.9999)
            # A normalized rank-one write has Frobenius norm <= 1. tanh and
            # bounded energy keep each write finite; decay >= .90 bounds the
            # recurrent state by a geometric series instead of raw activation scale.
            err_unit = F.normalize(err, dim=-1, eps=1e-6)
            write_lr = torch.tanh(surprise) * e_delta[:, t].clamp(0.0, 1.0)
            gate = (surprise >= self.tau.clamp_min(0.0)).to(x.dtype)
            if collect_diagnostics:
                write_rates.append(gate.mean().detach())
            S = S * decay.unsqueeze(-1)
            update = torch.bmm(k_unit.unsqueeze(2), err_unit.unsqueeze(1))
            S = S + update * (write_lr * gate).unsqueeze(-1)
            if compute_complexity:
                current_log_prob = torch.log_softmax(S.float(), dim=-1)
                current_prob = current_log_prob.exp()
                kl = (current_prob * (current_log_prob - previous_log_prob)).sum(dim=-1).mean()
                complexities.append(kl.to(x.dtype))
            q_unit = F.normalize(q, dim=-1, eps=1e-6)
            delta_out = torch.bmm(q_unit.unsqueeze(1), S).squeeze(1)
            # Keep associative retrieval subordinate to the information stream.
            max_out_norm = 0.5 * x[:, t].norm(dim=-1, keepdim=True)
            out_norm = delta_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            delta_out = delta_out * (max_out_norm / out_norm).clamp(max=1.0)
            outs.append(delta_out)
            surprises.append(surprise)
        self.state = S
        self.last_complexity = torch.stack(complexities).mean() if compute_complexity else None
        output = torch.stack(outs, dim=1)
        if collect_diagnostics:
            self.last_write_rate = float(torch.stack(write_rates).mean().item())
            self.last_state_norm = float(S.detach().norm().item() / max(1, batch ** 0.5))
            self.last_output_norm = float(output.detach().norm().item() / max(1, (batch * steps) ** 0.5))
        return output, torch.stack(surprises, dim=1)

    def clear_state(self):
        self.state = torch.empty(0, device=self.state.device if self.state.numel() else None)
