"""RONIN-LUMEN recurrent CPU architecture with optional V3.2 ZE extension."""
import torch
from torch import nn
from .config import CONFIG, LumenConfig
from .embedding import VivoEmbedding
from .energy import Presupuesto
from .delta_state import EstadoDelta
from .homeostasis import Homeostasis
from .memory import MemoriaElastica
from .heads import Heads
from .ze import DualModelZe
from .pain import Nociception
from .pleasure import Recompensa
from .chunking import AdaptiveChunking
from .neuroplasticity import SignalContext


class RoninLumen(nn.Module):
    def __init__(self, config: LumenConfig = CONFIG, v32_ze: bool = False):
        super().__init__()
        self.config = config
        self.v32_ze = bool(v32_ze)
        self.ze = DualModelZe(config.model_dim, counterfactual_dim=config.ze_dim) if self.v32_ze else None
        self.emb = VivoEmbedding(config.vocab_size, config.model_dim)
        self.presupuesto = Presupuesto(config.model_dim)
        self.pain = Nociception()
        self.pleasure = Recompensa()
        self.delta_layers = nn.ModuleList([
            EstadoDelta(config.model_dim, config.surprise_threshold, config.delta_rank,
                        config.delta_parallel_refinements)
            for _ in range(config.delta_layers)
        ])
        self.homeo = nn.ModuleList([
            Homeostasis(config.model_dim, config.homeostasis_target,
                        config.homeostasis_eta,
                        config.top_k_low if i < 2 else config.top_k_high)
            for i in range(config.delta_layers)
        ])
        # A shared bounded controller keeps the parameter overhead small.
        self.chunker = AdaptiveChunking(config.model_dim)
        self.memoria = MemoriaElastica(config.model_dim, hidden=config.memory_hidden)
        self.heads = Heads(config.model_dim, hidden=config.head_hidden,
                               n_splines=config.kan_splines, n_puntos=config.kan_points)
        self.head_out = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        self.head_out.weight = self.emb.emb.weight
        self.jepa_predictor = nn.Sequential(
            nn.Linear(config.model_dim, config.model_dim), nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )

    @property
    def delta(self):
        return self.delta_layers[0]

    def reset_states(self):
        for layer in self.delta_layers:
            layer.clear_state()
        if self.ze is not None:
            self.ze.delta_b.clear_state()
            self.ze.conflict_streak = torch.zeros(
                0, dtype=torch.long, device=self.ze.conflict_streak.device
            )

    def reset_state(self):
        """Public singular alias: clear only transient recurrent activity."""
        self.reset_states()

    @torch.no_grad()
    def set_learning_modulator(self, signal):
        """Send delayed improvement/regression feedback to every Delta trace."""
        for delta in self.delta_layers:
            delta.set_eligibility_modulator(signal)

    def forward(self, token_ids, reset_state=True, collect_diagnostics=True):
        h = self.emb(token_ids)
        e_delta, e_mem, e_head, energy = self.presupuesto(h)
        surprises = []
        chunk_sizes = []
        complexity_terms = []
        for delta, homeo in zip(self.delta_layers, self.homeo):
            # Delta returns an associative-memory update, not a replacement
            # representation. Preserve the information stream through a
            # residual path so a cold/low-norm memory read cannot zero the stack.
            delta_update, surprise = delta(
                h, e_delta, reset_state=reset_state,
                compute_complexity=self.v32_ze,
                collect_diagnostics=collect_diagnostics,
            )
            h = self.chunker(h, surprise)
            chunk_sizes.append(self.chunker.last_chunk_size)
            h = homeo(h + delta_update, collect_diagnostics=collect_diagnostics)
            surprises.append(surprise)
            if self.v32_ze:
                complexity_terms.append(delta.last_complexity)
        surprise = torch.stack(surprises).mean(0)
        signals = SignalContext.from_surprise(surprise)
        free_energy_complexity = (
            torch.stack(complexity_terms).mean()
            if complexity_terms else h.new_zeros(())
        )

        ze_metrics = None
        if self.ze is not None:
            h, ze_metrics = self.ze(h, surprise, reset_state=reset_state)
            triggered = ze_metrics["erase_trigger"].any(dim=1)
            if triggered.any():
                direction = ze_metrics["context_direction"].mean(dim=1)
                direction = direction * triggered.unsqueeze(-1)
                for delta_layer in self.delta_layers:
                    if delta_layer.state.numel():
                        self.ze.operator_borrado(delta_layer.state, direction)

        mem = self.memoria(h, surprise, e_mem, signals=signals)
        hidden = h + mem
        headed, router = self.heads(hidden, e_head)
        logits = self.head_out(headed)
        return {
            "logits": logits, "router": router, "activations": h,
            "surprise": surprise, "signals": signals, "energy": energy, "hidden": hidden,
            "free_energy_complexity": free_energy_complexity,
            "ze": ze_metrics,
            "chunk_sizes": tuple(chunk_sizes),
            "eligibility_norms": tuple(delta.last_eligibility_norm for delta in self.delta_layers),
        }

    def predict_latent(self, hidden):
        return self.jepa_predictor(hidden)
