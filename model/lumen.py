"""Minimal M1 model shell; V1 organs are deliberately not implemented yet."""
import torch
from torch import nn
from torch.nn import functional as F
from .config import CONFIG, LumenConfig

class RoninLumen(nn.Module):
    """Embed token IDs and return vocabulary logits for the M1 smoke test."""
    def __init__(self, config: LumenConfig = CONFIG):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.projection = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.norm = nn.LayerNorm(config.model_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(token_ids)
        hidden = self.norm(hidden + F.gelu(self.projection(hidden)))
        return F.linear(hidden, self.embedding.weight)
