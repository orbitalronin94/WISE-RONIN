"""Opt-in V3.2 dual-state controller and bounded context-decorrelation operator."""
import torch
from torch import nn
from .delta_state import EstadoDelta


class DualModelZe(nn.Module):
    """M_A is the causal stack; M_B is a compact learned counterfactual proxy."""
    def __init__(self, dim=256, counterfactual_dim=128, threshold=0.25, streak_limit=10):
        super().__init__()
        self.to_b = nn.Sequential(nn.Linear(dim, counterfactual_dim), nn.Tanh())
        self.delta_b = EstadoDelta(counterfactual_dim, tau=0.15)
        self.from_b = nn.Linear(counterfactual_dim, dim, bias=False)
        self.threshold = float(threshold)
        self.streak_limit = int(streak_limit)
        self.register_buffer("conflict_streak", torch.zeros(0, dtype=torch.long))
        self.register_buffer("localization_count", torch.zeros((), dtype=torch.long))
        self.register_buffer("erase_count", torch.zeros((), dtype=torch.long))

    def forward(self, causal_hidden, causal_pe, reset_state=True):
        if reset_state:
            with torch.no_grad():
                self.conflict_streak = torch.zeros(
                    causal_hidden.shape[0], dtype=torch.long, device=causal_hidden.device
                )
        b_input = self.to_b(causal_hidden)
        b_hidden, b_pe = self.delta_b(
            b_input, torch.ones_like(b_input[..., :1]), reset_state=reset_state
        )
        b_shared = self.from_b(b_hidden)
        fa = causal_pe.mean(dim=-1)
        fb = b_pe.mean(dim=-1)
        delta_f = (fa - fb).abs()
        latent_disagreement = (
            (causal_hidden - b_shared).norm(dim=-1) / (causal_hidden.shape[-1] ** 0.5)
        )
        conflict = delta_f + 0.1 * latent_disagreement
        high = conflict > self.threshold

        with torch.no_grad():
            if self.conflict_streak.numel() != high.shape[0]:
                self.conflict_streak = torch.zeros(
                    high.shape[0], dtype=torch.long, device=high.device
                )
            trigger = torch.zeros_like(high)
            for t in range(high.shape[1]):
                self.conflict_streak = torch.where(
                    high[:, t], self.conflict_streak + 1,
                    torch.zeros_like(self.conflict_streak),
                )
                trigger[:, t] = self.conflict_streak >= self.streak_limit
                self.conflict_streak = torch.where(
                    trigger[:, t], torch.zeros_like(self.conflict_streak),
                    self.conflict_streak,
                )

        blend = torch.sigmoid((self.threshold - conflict).unsqueeze(-1) * 4.0)
        choose_b = (fb < fa).unsqueeze(-1)
        localized = torch.where(choose_b, b_shared, causal_hidden)
        result = torch.where(
            high.unsqueeze(-1), localized,
            blend * causal_hidden + (1.0 - blend) * b_shared,
        )
        with torch.no_grad():
            self.localization_count.add_(high.sum())
        return result, {
            "F_A": fa, "F_B": fb, "delta_F": delta_f,
            "latent_disagreement": latent_disagreement,
            "conflict": conflict, "high_conflict": high,
            "erase_trigger": trigger,
            "context_direction": causal_hidden - b_shared,
        }

    @torch.no_grad()
    def operator_borrado(self, delta_state, context_direction):
        """Remove a bounded rank-one context component from [B,D,D] Delta states."""
        if context_direction.numel() == 0:
            return 0.0
        e = context_direction.detach()
        if e.ndim == 1:
            e = e.unsqueeze(0).expand(delta_state.shape[0], -1)
        norm_sq = e.square().sum(dim=-1, keepdim=True).clamp_min(1e-8)
        before = delta_state.norm(dim=(-2, -1), keepdim=True)
        projected = torch.bmm(delta_state, e.unsqueeze(-1)).squeeze(-1)
        cleaned = (
            delta_state
            - projected.unsqueeze(-1) * e.unsqueeze(1) / norm_sq.unsqueeze(-1)
        )
        change = cleaned - delta_state
        change_norm = change.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        scale = torch.clamp((0.25 * before) / change_norm, max=1.0)
        delta_state.add_(scale * change)
        self.erase_count.add_(1)
        return float((scale * change).norm().item())
