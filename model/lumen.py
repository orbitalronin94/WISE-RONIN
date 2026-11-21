"""RONIN-LUMEN recurrent CPU architecture: eight explicitly separated organs."""
import torch
from torch import nn
from .config import CONFIG, LumenConfig
from .embedding import VivoEmbedding
from .energy import Presupuesto
from .delta_state import EstadoDelta
from .homeostasis import Homeostasis
from .memory import MemoriaElastica
from .heads import Heads

class RoninLumen(nn.Module):
    def __init__(self, config: LumenConfig = CONFIG):
        super().__init__()
        self.config = config
        self.emb = VivoEmbedding(config.vocab_size, config.model_dim)
        self.presupuesto = Presupuesto(config.model_dim)
        self.delta_layers = nn.ModuleList([EstadoDelta(config.model_dim, config.surprise_threshold) for _ in range(config.delta_layers)])
        self.homeo = nn.ModuleList([
            Homeostasis(config.model_dim, config.homeostasis_target, config.homeostasis_eta,
                        config.top_k_low if i < 2 else config.top_k_high)
            for i in range(config.delta_layers)
        ])
        self.memoria = MemoriaElastica(config.model_dim)
        self.heads = Heads(config.model_dim)
        self.head_out = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.head_out.weight = self.emb.emb.weight
        self.jepa_predictor = nn.Sequential(nn.Linear(config.model_dim, config.model_dim), nn.GELU(), nn.Linear(config.model_dim, config.model_dim))

    @property
    def delta(self):
        return self.delta_layers[0]

    def reset_states(self):
        for layer in self.delta_layers:
            layer.clear_state()

    def forward(self, token_ids, reset_state=True):
        h = self.emb(token_ids)
        e_delta, e_mem, e_head, energy = self.presupuesto(h)
        surprises = []
        for delta, homeo in zip(self.delta_layers, self.homeo):
            h, surprise = delta(h, e_delta, reset_state=reset_state)
            h = homeo(h)
            surprises.append(surprise)
        surprise = torch.stack(surprises).mean(0)
        mem = self.memoria(h, surprise, e_mem)
        hidden = h + mem
        headed, router = self.heads(hidden, e_head)
        logits = self.head_out(headed)
        return {"logits": logits, "router": router, "activations": h,
                "surprise": surprise, "energy": energy, "hidden": hidden}

    def predict_latent(self, hidden):
        return self.jepa_predictor(hidden)
