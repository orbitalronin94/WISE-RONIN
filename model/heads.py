"""KAN-like spline operator head, language MLP and learned router."""
import torch
from torch import nn

class SplineKAN(nn.Module):
    def __init__(self, dim_in=256, n_splines=32, n_puntos=8):
        super().__init__()
        self.n_splines, self.n_puntos = n_splines, n_puntos
        self.proj_in = nn.Linear(dim_in, n_splines)
        self.puntos = nn.Parameter(torch.randn(n_splines, n_puntos) * 0.1)
        self.proj_out = nn.Linear(n_splines, dim_in)

    def forward(self, x):
        pos = torch.sigmoid(self.proj_in(x)) * (self.n_puntos - 1)
        idx0 = pos.floor().long().clamp(0, self.n_puntos - 2)
        frac = pos - idx0.to(pos.dtype)
        expanded = self.puntos.unsqueeze(0).unsqueeze(0).expand(*idx0.shape, -1)
        p0 = torch.gather(expanded, -1, idx0.unsqueeze(-1)).squeeze(-1)
        p1 = torch.gather(expanded, -1, (idx0 + 1).unsqueeze(-1)).squeeze(-1)
        return self.proj_out(p0 * (1.0 - frac) + p1 * frac)

class Heads(nn.Module):
    def __init__(self, dim=256, hidden=512, n_splines=32, n_puntos=8):
        super().__init__()
        self.router = nn.Linear(dim, 2)
        self.kan = SplineKAN(dim, n_splines=n_splines, n_puntos=n_puntos)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x, e_head):
        probs = torch.softmax(self.router(x), dim=-1)
        out = probs[..., 0:1] * self.kan(x) + probs[..., 1:2] * self.mlp(x)
        return out * e_head, probs
