"""Low-rank surprise-gated recurrent Delta memory with a parallel fixed-point scan."""
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F
from .plasticity import PlasticityProfile
from .neuroplasticity import IntrinsicThreshold, SimulatedSynapticPruner, SynapticTaggingCapture


@lru_cache(maxsize=32)
def _cached_causal_masks(steps, device, dtype):
    """Reuse immutable causal masks across the four identical Delta organs."""
    causal = torch.ones(steps, steps, device=device, dtype=dtype).tril()
    return causal, causal.tril(-1)


class EstadoDelta(nn.Module):
    def __init__(self, dim=256, tau=0.15, rank=64, parallel_refinements=6):
        super().__init__()
        self.dim = dim
        self.rank = min(int(rank), dim)
        if self.rank < 1:
            raise ValueError("Delta memory rank must be >= 1")
        self.parallel_refinements = max(1, int(parallel_refinements))
        self.plasticity_profile = PlasticityProfile()
        self.tagging = SynapticTaggingCapture((self.rank, self.dim), decay=0.95, threshold=0.005)
        self.intrinsic_threshold = IntrinsicThreshold(initial=max(0.01, min(0.8, float(tau))), target=0.05, rate=0.0005)
        self.simulated_pruner = SimulatedSynapticPruner((self.rank, self.dim))
        self.parallel_scan = True
        self.pre_normalize_sequence = False
        # Exact-equivalence simplifications toggled only by benchmark profiles.
        self.reuse_write_update = False
        self.always_apply_eligibility = False
        self.trace_norm_diagnostics_on_demand = False
        self.share_causal_masks = False
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
        # Eligibility is a bounded cross-batch trace of key/write coincidences.
        # It is retained across transient Delta resets and modulated by delayed
        # improvement/regression feedback from the affect organs.
        self.register_buffer("eligibility_trace", torch.zeros(self.rank, self.dim), persistent=True)
        self.register_buffer("eligibility_modulator", torch.zeros(()), persistent=True)
        self.last_eligibility_norm = 0.0
        self.last_complexity = None
        self.last_write_rate = 0.0
        self.last_state_norm = 0.0
        self.last_output_norm = 0.0

    def reset(self, batch, device, dtype):
        self.state = torch.zeros(batch, self.rank, self.dim, device=device, dtype=dtype)

    @torch.no_grad()
    def set_eligibility_modulator(self, signal):
        value = torch.as_tensor(
            signal, dtype=self.eligibility_modulator.dtype,
            device=self.eligibility_modulator.device,
        ).clamp(-1.0, 1.0)
        self.eligibility_modulator.copy_(0.25 * value)

    def _plasticity_gate(self, surprise, dtype):
        """Apply bounded threshold and attention modulation to one shared surprise tensor."""
        profile = self.plasticity_profile
        attended = surprise * (1.0 + profile.attention_gain * surprise.detach().clamp(0.0, 1.0))
        threshold = self.tau.clamp_min(0.0)
        if profile.intrinsic_threshold_enabled:
            threshold = self.intrinsic_threshold.threshold.to(device=surprise.device, dtype=surprise.dtype)
        return (attended >= threshold * profile.threshold_scale * profile.inhibition_scale).to(dtype)

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
        if self.share_causal_masks:
            causal, strict_causal = _cached_causal_masks(steps, x.device, x.dtype)
        else:
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
            gate = self._plasticity_gate(surprise, x.dtype)
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
        if self.reuse_write_update:
            batch_write_update = torch.einsum("btr,btd->brd", weighted_keys, write_vectors)
            write_summary = batch_write_update.mean(dim=0) / max(1, steps)
        else:
            batch_write_update = None
            write_summary = torch.einsum(
                "btr,btd->brd", weighted_keys, write_vectors
            ).mean(dim=0) / max(1, steps)
        with torch.no_grad():
            trace_update = write_summary.detach().to(
                device=self.eligibility_trace.device,
                dtype=self.eligibility_trace.dtype,
            )
            trace_norm = trace_update.norm().clamp_min(1e-8)
            trace_update = trace_update * (0.05 / trace_norm).clamp(max=1.0)
            if self.plasticity_profile.simulated_pruning_enabled:
                mask = self.simulated_pruner.observe(
                    trace_update.abs(), self.eligibility_trace.abs() + trace_update.abs()
                )
                trace_update = trace_update * mask
            retention = self.plasticity_profile.eligibility_retention
            self.eligibility_trace.mul_(retention).add_(trace_update, alpha=1.0 - retention)
            if self.plasticity_profile.tagging_enabled:
                resource = float(self.eligibility_modulator.abs().mul(4.0).clamp(0.0, 1.0).item())
                self.tagging.update(trace_update, resource)
            if self.plasticity_profile.intrinsic_threshold_enabled:
                self.intrinsic_threshold.update(gate.detach())
            if self.plasticity_profile.ltd_enabled and self.plasticity_profile.tagging_enabled:
                negative_reward = (-self.eligibility_modulator).clamp(0.0, 0.25)
                protected_tags = self.tagging.captured.clamp(0.0, 1.0)
                self.eligibility_trace.mul_(1.0 - 0.10 * negative_reward * protected_tags)
            if not self.trace_norm_diagnostics_on_demand or collect_diagnostics:
                self.last_eligibility_norm = float(self.eligibility_trace.norm().item())

        if batch_write_update is None:
            batch_write_update = torch.einsum("btr,btd->brd", weighted_keys, write_vectors)
        final_state = state * prefix[:, -1].view(batch, 1, 1) + batch_write_update
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
        # Apply the previous batch's eligible trace only when delayed feedback
        # has assigned it value. The trace itself remains bounded and detached.
        modulation = self.eligibility_modulator.to(device=x.device, dtype=x.dtype) * self.plasticity_profile.modulator_gain
        if self.always_apply_eligibility or bool(modulation.detach().abs().item() > 0):
            trace = self.eligibility_trace.to(device=x.device, dtype=x.dtype)
            if self.plasticity_profile.tagging_enabled:
                captured = self.tagging.captured.to(device=x.device, dtype=x.dtype)
                trace = trace * captured
            state = state + modulation * trace.unsqueeze(0)
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
            gate = self._plasticity_gate(surprise_t, x.dtype)
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
