"""Ternary PE-gated elastic editing and append-only slow-memory creation."""
import torch
from torch import nn
from .neuroplasticity import OneShotGate


class MemoriaElastica(nn.Module):
    def __init__(self, dim=256, hidden=64, slots=128,
                 small_pe=0.3, large_pe=0.8):
        super().__init__()
        self.dim = dim
        self.small_pe = float(small_pe)
        self.large_pe = float(large_pe)
        self.one_shot_enabled = False
        self.one_shot_gate = OneShotGate(base=0.05, maximum=0.20, gain=1.0)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.register_buffer("lenta", torch.zeros(slots, dim))
        self.register_buffer("contador_lenta", torch.zeros(slots))
        self.register_buffer("uso_slot", torch.zeros(slots))
        self.register_buffer("usage_ema", torch.zeros(slots))
        self.register_buffer("write_pointer", torch.zeros((), dtype=torch.long))
        self.register_buffer("tokens_since_decay", torch.zeros((), dtype=torch.long))

    def occupancy(self):
        return float((self.contador_lenta > 0).float().mean().item())

    @torch.no_grad()
    def registrar_nueva(self, vector):
        """Write into an empty slot; if full, replace the least-used slot."""
        vector = vector.detach().reshape(-1, self.dim).mean(0)
        if not torch.isfinite(vector).all():
            return False
        norm = torch.linalg.vector_norm(vector.to(torch.float64))
        if norm <= 1e-8:
            return False
        # Prevent one anomalous activation from poisoning long-lived memory.
        vector = vector * (10.0 / norm).clamp(max=1.0).to(vector.dtype)
        empty = torch.nonzero(self.contador_lenta == 0, as_tuple=False).flatten()
        if empty.numel():
            slot = int(empty[0].item())
        else:
            slot = int(self.usage_ema.argmin().item())
        self.lenta[slot].copy_(vector)
        self.contador_lenta[slot].add_(1)
        self.uso_slot[slot].add_(1)
        self.usage_ema[slot].add_(1.0)
        self.write_pointer.fill_((slot + 1) % self.lenta.shape[0])
        return True

    def forward(self, x, sorpresa, e_mem, signals=None):
        pe = (signals.surprise if signals is not None else sorpresa).detach().squeeze(-1)
        small = pe < self.small_pe
        fast_rate = self.one_shot_gate(signals.uncertainty.squeeze(-1) if signals is not None else pe / (1.0 + pe.abs())) if self.one_shot_enabled else torch.zeros_like(pe)
        # High uncertainty can open a bounded local write path; the global
        # optimizer learning rate is never modified.
        large = pe > self.large_pe * (1.0 - fast_rate)

        elastic = self.mlp(x)
        if self.training and elastic.requires_grad:
            elastic.register_hook(lambda grad: grad * 0.01)
        elastic = elastic * small.unsqueeze(-1).to(elastic.dtype)

        # Memory writes are a training-time plasticity operation. Evaluation
        # must retrieve from the frozen learned bank without cross-prompt mutation.
        if self.training:
            with torch.no_grad():
                for batch_index in range(x.shape[0]):
                    selected = x[batch_index][large[batch_index]]
                    if selected.numel():
                        self.registrar_nueva(selected)

        valid = self.contador_lenta > 0
        if not valid.any():
            slow = torch.zeros_like(x)
        else:
            bank = self.lenta[valid]
            weights = torch.softmax(
                torch.einsum("btd,sd->bts", x, bank) / (self.dim ** 0.5),
                dim=-1,
            )
            slow = torch.einsum("bts,sd->btd", weights, bank)
            if self.training:
                with torch.no_grad():
                    usage = torch.zeros_like(self.usage_ema)
                    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
                    usage[valid_indices] = weights.detach().mean((0, 1))
                    self.usage_ema.mul_(0.99).add_(usage, alpha=0.01)
                    self.tokens_since_decay.add_(x.shape[1])
                    while self.tokens_since_decay.item() >= 512:
                        unused = (self.contador_lenta > 0) & (self.usage_ema < 0.001)
                        self.lenta[unused] = self.lenta[unused] * 0.99
                        self.tokens_since_decay.sub_(512)
        return (0.7 * elastic + 0.3 * slow) * e_mem

    @torch.no_grad()
    def reset_elastica(self):
        # The elastic component is parameterized by the MLP; it has no separate mutable bank.
        return None
