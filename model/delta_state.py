"""Low-rank surprise-gated recurrent Delta memory with a parallel fixed-point scan."""
import torch
from torch import nn
from torch.nn import functional as F


class EstadoDelta(nn.Module):
    def __init__(self, dim=256, tau=0.15, rank=64, parallel_refinements=6):
        super().__init__()
        self.dim = dim
        self.rank = min(int(rank), dim)
        if self.rank < 1:
            raise ValueError("Delta memory rank must be >= 1")
        self.parallel_refinements = max(1, int(parallel_refinements))
        self.parallel_scan = True
        self.pre_normalize_sequence = False
        self.proj_k = nn.Linear(dim, dim, bias=False)
        self.proj_v = nn.Linear(dim, dim, bias=False)
        self.proj_q = nn.Linear(dim, dim, bias=False)
        self.a = nn.Parameter(torch.tensor(-1.0))
        self.b = nn.Parameter(torch.tensor(2.0))
        self.tau = nn.Parameter(torch.tensor(float(tau)))
        sketch = torch.randn(dim, self.rank)
        projection, _ = torch.linalg.qr(sketch, mode="reduced")
        self.register_buffer("memory_projection", projection, persistent=True)
        self.register_buffer("state", torch.empty(0), persistent=True)
        self.last_complexity = None
        self.last_write_rate = 0.0
        self.last_state_norm = 0.0
        self.last_output_norm = 0.0

    def reset(self, batch, device, dtype):
        self.state = torch.zeros(batch, self.rank, self.dim, device=device, dtype=dtype)

    def _project_sequence(self, x):
        projected = F.linear(x, torch.cat(
            (self.proj_k.weight, self.proj_v.weight, self.proj_q.weight), dim=0
        ))
        k_all, v_all, q_all = projected.chunk(3, dim=-1)
        if self.pre_normalize_sequence:
            k_all = F.normalize(k_all, dim=-1, eps=1e-6)
            q_all = F.normalize(q_all, dim=-1, eps=1e-6)
        projection = self.memory_projection.to(dtype=x.dtype)
        k_sketch = F.normalize(
            F.normalize(k_all, dim=-1, eps=1e-6) @ projection,
            dim=-1, eps=1e-6,
        )
        q_sketch = F.normalize(
            F.normalize(q_all, dim=-1, eps=1e-6) @ projection,
            dim=-1, eps=1e-6,
        )
        return k_sketch, v_all, q_sketch

    def _parallel_fixed_point(self, x, e_delta, k, v, q, state,
                               collect_diagnostics):
        """Batched causal solve for the surprise-gated rank-one recurrence."""
        batch, steps, _ = k.shape
        key_gram = torch.bmm(k, k.transpose(1, 2))
        query_key = torch.bmm(q, k.transpose(1, 2))
        causal = torch.ones(steps, steps, device=x.device, dtype=x.dtype).tril()
        strict_causal = torch.ones_like(causal).tril(-1)

        decay = torch.ones(batch, steps, device=x.device, dtype=x.dtype)
        write_vectors = torch.zeros_like(v)
        surprise = None
        gate = None
        for _ in range(self.parallel_refinements):
            prefix = torch.cumprod(decay, dim=1).clamp_min(1e-12)
            prefix_previous = torch.cat(
                (torch.ones_like(prefix[:, :1]), prefix[:, :-1]), dim=1
            )
            ratios = prefix_previous.unsqueeze(-1) / prefix.unsqueeze(1)
            recurrent_weights = key_gram * ratios * strict_causal
            prediction = (
                prefix_previous.unsqueeze(-1) * torch.bmm(k, state)
                + torch.bmm(recurrent_weights, write_vectors)
            )
            error = v - prediction
            surprise = error.norm(dim=-1, keepdim=True) / (self.dim ** 0.5)
            decay = torch.sigmoid(self.a * surprise + self.b).squeeze(-1).clamp(0.90, 0.9999)
            error_unit = F.normalize(error, dim=-1, eps=1e-6)
            write_lr = torch.tanh(surprise) * e_delta.clamp(0.0, 1.0)
            gate = (surprise >= self.tau.clamp_min(0.0)).to(x.dtype)
            write_vectors = error_unit * (write_lr * gate)

        prefix = torch.cumprod(decay, dim=1).clamp_min(1e-12)
        ratios = prefix.unsqueeze(-1) / prefix.unsqueeze(1)
        read_weights = query_key * ratios * causal
        output = (
            prefix.unsqueeze(-1) * torch.bmm(q, state)
            + torch.bmm(read_weights, write_vectors)
        )
        final_ratios = prefix[:, -1:].div(prefix).unsqueeze(-1)
        weighted_keys = k * final_ratios
        final_state = (
            state * prefix[:, -1].view(batch, 1, 1)
            + torch.einsum("btr,btd->brd", weighted_keys, write_vectors)
        )
        max_output_norm = 0.5 * x.norm(dim=-1, keepdim=True)
        output_norm = output.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        output = output * (max_output_norm / output_norm).clamp(max=1.0)

        if collect_diagnostics:
            self.last_write_rate = float(gate.mean().detach().item())
            self.last_state_norm = float(
                final_state.detach().norm().item() / max(1, batch ** 0.5)
            )
            self.last_output_norm = float(
                output.detach().norm().item() / max(1, (batch * steps) ** 0.5)
            )
        return output, surprise, final_state

    def forward(self, x, e_delta, reset_state=False, compute_complexity=False,
                collect_diagnostics=True):
        batch, steps, _ = x.shape
        if (reset_state or self.state.numel() == 0
                or self.state.shape[0] != batch or self.state.device != x.device):
            self.reset(batch, x.device, x.dtype)
        state = self.state.to(dtype=x.dtype)
        k_all, v_all, q_all = self._project_sequence(x)

        if self.parallel_scan and not compute_complexity:
            output, surprise, final_state = self._parallel_fixed_point(
                x, e_delta, k_all, v_all, q_all, state, collect_diagnostics
            )
            self.state = final_state
            self.last_complexity = None
            return output, surprise

        outputs, surprises = [], []
        complexities, write_rates = [], []
        current_state = state
        for t in range(steps):
            if compute_complexity:
                previous_log_prob = torch.log_softmax(current_state.float(), dim=-1)
            k, v, q = k_all[:, t], v_all[:, t], q_all[:, t]
            prediction = torch.bmm(k.unsqueeze(1), current_state).squeeze(1)
            error = v - prediction
            surprise_t = error.norm(dim=-1, keepdim=True) / (self.dim ** 0.5)
            decay = torch.sigmoid(self.a * surprise_t + self.b).clamp(0.90, 0.9999)
            error_unit = F.normalize(error, dim=-1, eps=1e-6)
            write_lr = torch.tanh(surprise_t) * e_delta[:, t].clamp(0.0, 1.0)
            gate = (surprise_t >= self.tau.clamp_min(0.0)).to(x.dtype)
            if collect_diagnostics:
                write_rates.append(gate.mean().detach())
            current_state = current_state * decay.unsqueeze(-1)
            update = torch.bmm(k.unsqueeze(2), error_unit.unsqueeze(1))
            current_state = current_state + update * (write_lr * gate).unsqueeze(-1)
            if compute_complexity:
                current_log_prob = torch.log_softmax(current_state.float(), dim=-1)
                current_prob = current_log_prob.exp()
                kl = (current_prob * (current_log_prob - previous_log_prob)).sum(dim=-1).mean()
                complexities.append(kl.to(x.dtype))
            delta_out = torch.bmm(q.unsqueeze(1), current_state).squeeze(1)
            max_out_norm = 0.5 * x[:, t].norm(dim=-1, keepdim=True)
            out_norm = delta_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            outputs.append(delta_out * (max_out_norm / out_norm).clamp(max=1.0))
            surprises.append(surprise_t)

        self.state = current_state
        self.last_complexity = torch.stack(complexities).mean() if compute_complexity else None
        output = torch.stack(outputs, dim=1)
        surprise = torch.stack(surprises, dim=1)
        if collect_diagnostics:
            self.last_write_rate = float(torch.stack(write_rates).mean().item())
            self.last_state_norm = float(current_state.detach().norm().item() / max(1, batch ** 0.5))
            self.last_output_norm = float(output.detach().norm().item() / max(1, (batch * steps) ** 0.5))
        return output, surprise

    def clear_state(self):
        self.state = torch.empty(0, device=self.state.device if self.state.numel() else None)
