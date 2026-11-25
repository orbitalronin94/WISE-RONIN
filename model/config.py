"""Single source of truth for RONIN-LUMEN V1 configuration."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class LumenConfig:
    model_dim: int = 256
    memory_hidden: int = 64
    head_hidden: int = 512
    kan_splines: int = 32
    kan_points: int = 8
    ze_dim: int = 128
    delta_layers: int = 4
    delta_rank: int = 64
    delta_parallel_refinements: int = 6
    vocab_size: int = 4096
    max_vocab_size: int = 8192
    top_k_low: int = 16
    top_k_high: int = 32
    homeostasis_target: float = 0.05
    homeostasis_eta: float = 0.001
    # Pilot logs show mean prediction error ~0.003 at steps 1-100.
    # The previous 0.15 gate prevented every recurrent Delta write at cold start.
    surprise_threshold: float = 0.001
    elastic_memory_threshold: float = 2.0
    sleep_interval_steps: int = 500
    sleep_steps: int = 50
    learning_rate: float = 1.5e-4
    warmup_steps: int = 100
    batch_size: int = 8
    sequence_length: int = 64
    total_steps: int = 10000
    seed: int = 42
    weight_decay: float = 0.01

    def to_dict(self) -> dict:
        """Return the immutable configuration as a serializable dictionary."""
        return asdict(self)

CONFIG = LumenConfig()
