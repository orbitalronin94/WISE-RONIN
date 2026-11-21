"""Live vocabulary usage accounting with non-destructive prune proposals."""
import torch
from torch import nn

class VivoEmbedding(nn.Module):
    def __init__(self, vocab=4096, dim=256, max_prune_fraction=0.05):
        super().__init__()
        self.vocab, self.dim = vocab, dim
        self.max_prune_fraction = max_prune_fraction
        self.emb = nn.Embedding(vocab, dim)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        self.register_buffer("uso", torch.zeros(vocab, dtype=torch.float32))
        self.register_buffer("pasos_desde_poda", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        with torch.no_grad():
            counts = torch.bincount(x.reshape(-1), minlength=self.vocab)
            self.uso.add_(counts.to(self.uso.dtype))
            self.pasos_desde_poda.add_(1)
        return self.emb(x)

    @torch.no_grad()
    def podar(self, max_porcentaje=None):
        """Return dead-token candidates; never invalidate the frozen tokenizer in-place."""
        if self.pasos_desde_poda.item() < 1000:
            return []
        fraction = self.max_prune_fraction if max_porcentaje is None else min(float(max_porcentaje), self.max_prune_fraction)
        candidates = torch.nonzero(self.uso[4:] == 0, as_tuple=False).flatten() + 4
        n = min(int(self.vocab * fraction), candidates.numel())
        proposals = candidates[:n].cpu().tolist()
        self.uso.zero_()
        self.pasos_desde_poda.zero_()
        return proposals
