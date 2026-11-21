"""Opt-in V3.2 dual-state controller and bounded context-decorrelation operator."""
import torch
from torch import nn
from .delta_state import EstadoDelta

class DualModelZe(nn.Module):
    """M_A is the existing causal stack; this module adds a compact M_B proxy."""
    def __init__(self, dim=256, counterfactual_dim=128, threshold=0.25, streak_limit=10):
        super().__init__()
        self.to_b = nn.Sequential(nn.Linear(dim, counterfactual_dim), nn.Tanh())
        self.delta_b = EstadoDelta(counterfactual_dim, tau=0.15)
        self.from_b = nn.Linear(counterfactual_dim, dim, bias=False)
        self.threshold = float(threshold)
        self.streak_limit = int(streak_limit)
        self.register_buffer("conflict_streak", torch.zeros((), dtype=torch.long))
        self.register_buffer("localization_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("erase_count", torch.zeros((), dtype=torch.long))

    def forward(self, causal_hidden, causal_pe, reset_state=True):
        b_input = self.to_b(causal_hidden)
        b_hidden, b_pe = self.delta_b(b_input, torch.ones_like(b_input[..., :1]),
                                      reset_state=reset_state)
        b_shared = self.from_b(b_hidden)
        fa = causal_pe.mean(dim=-1)
        fb = b_pe.mean(dim=-1)
        delta_f = (fa - fb).abs()
        # Error-magnitude conflict plus disagreement between the two latent predictions.
        latent_disagreement = (causal_hidden - b_shared).norm(dim=-1) / (causal_hidden.shape[-1] ** 0.5)
        conflict = delta_f + 0.1 * latent_disagreement
        high = conflict > self.threshold
        with torch.no_grad():
            if high.any():
                self.conflict_streak.add_(1)
            else:
                self.conflict_streak.zero_()
        # On low conflict, constructive interference; on high conflict, localize to lower PE.
        blend = torch.sigmoid((self.threshold - conflict).unsqueeze(-1) * 4.0)
        choose_b = (fb < fa).unsqueeze(-1)
        localized = torch.where(choose_b, b_shared, causal_hidden)
        result = torch.where(high.unsqueeze(-1), localized,
                             blend * causal_hidden + (1.0 - blend) * b_shared)
        with torch.no_grad():
            self.localization_count.add_(high.sum())
        return result, {
            "F_A": fa, "F_B": fb, "delta_F": delta_f,
            "latent_disagreement": latent_disagreement,
            "conflict": conflict, "high_conflict": high,
        }

    @torch.no_grad()
    def operator_borrado(self, delta_state, context_direction):
        """Remove only the rank-one component aligned with the estimated context direction."""
        if context_direction.numel() == 0:
            return 0.0
        e = context_direction.detach().mean(dim=tuple(range(context_direction.ndim - 1)))
        norm_sq = e.dot(e).clamp_min(1e-8)
        before = delta_state.norm()
        projected = torch.matmul(delta_state, e)
        cleaned = delta_state - projected.unsqueeze(-1) * e / norm_sq
        # Bound the intervention: do not remove more than 25% of the original state norm.
        change = cleaned - delta_state
        max_change = 0.25 * before
        change_norm = change.norm().clamp_min(1e-8)
        scale = torch.clamp(max_change / change_norm, max=1.0)
        delta_state.copy_(delta_state + scale * change)
        self.erase_count.add_(1)
        return float((scale * change).norm().item())
