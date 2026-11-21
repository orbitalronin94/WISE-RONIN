"""Surprise-gated recurrent Delta memory with per-token state-complexity diagnostics."""
import torch
from torch import nn


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

    def forward(self, x, e_delta, reset_state=False):
        batch, steps, _ = x.shape
        if (reset_state or self.state.numel() == 0
                or self.state.shape[0] != batch or self.state.device != x.device):
            self.reset(batch, x.device, x.dtype)
        S = self.state.to(dtype=x.dtype)
        outs, surprises, complexities, write_rates = [], [], [], []
        for t in range(steps):
            previous_log_prob = torch.log_softmax(S.float(), dim=-1)
            xt = x[:, t]
            k, v, q = self.proj_k(xt), self.proj_v(xt), self.proj_q(xt)
            pred = torch.einsum("bi,bij->bj", k, S)
            err = v - pred
            surprise = err.norm(dim=-1, keepdim=True) / (self.dim ** 0.5)
            decay = torch.sigmoid(self.a * surprise + self.b).clamp(0.90, 0.9999)
            write_lr = surprise.clamp(0.1, 1.0)
            gate = (surprise >= self.tau.clamp_min(0.0)).to(x.dtype)
            write_rates.append(gate.mean().detach())
            S = S * decay.unsqueeze(-1)
            update = torch.einsum("bi,bj->bij", k, err)
            S = S + update * (write_lr * gate * e_delta[:, t]).unsqueeze(-1)
            current_log_prob = torch.log_softmax(S.float(), dim=-1)
            current_prob = current_log_prob.exp()
            kl = (current_prob * (current_log_prob - previous_log_prob)).sum(dim=-1).mean()
            complexities.append(kl.to(x.dtype))
            outs.append(torch.einsum("bi,bij->bj", q, S))
            surprises.append(surprise)
        self.state = S
        self.last_complexity = torch.stack(complexities).mean()
        output = torch.stack(outs, dim=1)
        self.last_write_rate = float(torch.stack(write_rates).mean().item())
        self.last_state_norm = float(S.detach().norm().item() / max(1, batch ** 0.5))
        self.last_output_norm = float(output.detach().norm().item() / max(1, (batch * steps) ** 0.5))
        return output, torch.stack(surprises, dim=1)

    def clear_state(self):
        self.state = torch.empty(0, device=self.state.device if self.state.numel() else None)
