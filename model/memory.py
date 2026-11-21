"""Elastic associative memory and a bounded consolidated slow-memory bank."""
import torch
from torch import nn

class MemoriaElastica(nn.Module):
    def __init__(self, dim=256, hidden=64, slots=128):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.register_buffer("lenta", torch.zeros(slots, dim))
        self.register_buffer("contador_lenta", torch.zeros(slots))
        self.register_buffer("sorpresa_acum", torch.zeros(()))
        self.register_buffer("contador_patron", torch.zeros(slots))
        self.register_buffer("uso_slot", torch.zeros(slots))

    def forward(self, x, sorpresa, e_mem):
        with torch.no_grad():
            self.sorpresa_acum.add_(sorpresa.detach().sum())
        elastic = self.mlp(x)
        if self.lenta.abs().sum().item() == 0:
            slow = torch.zeros_like(x)
        else:
            weights = torch.softmax(torch.einsum("btd,sd->bts", x, self.lenta) / (self.dim ** 0.5), dim=-1)
            slow = torch.einsum("bts,sd->btd", weights, self.lenta)
        return (0.7 * elastic + 0.3 * slow) * e_mem

    @torch.no_grad()
    def consolidar(self, pattern, repeats=3):
        pattern = pattern.detach().reshape(-1, self.dim).mean(0)
        if not torch.isfinite(pattern).all() or pattern.norm() == 0:
            return False
        similarities = torch.mv(self.lenta, pattern) / (self.lenta.norm(dim=-1) * pattern.norm() + 1e-8)
        slot = int(similarities.argmax().item()) if self.uso_slot.any() else int(self.uso_slot.argmin().item())
        if self.uso_slot[slot] > 0 and similarities[slot] > 0.9:
            self.contador_patron[slot].add_(1)
            if self.contador_patron[slot] >= repeats:
                self.lenta[slot].lerp_(pattern, 0.5)
                self.contador_lenta[slot].add_(1)
                self.contador_patron[slot].zero_()
                return True
        else:
            slot = int(self.uso_slot.argmin().item())
            self.lenta[slot].copy_(pattern)
            self.uso_slot[slot].add_(1)
            self.contador_patron[slot].fill_(1)
        return False

    @torch.no_grad()
    def reset_elastica(self):
        self.sorpresa_acum.zero_()

    @torch.no_grad()
    def decay_unused(self):
        self.lenta.mul_(0.99)
