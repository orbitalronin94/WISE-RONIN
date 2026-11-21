"""Single source of truth for RONIN-LUMEN V1 configuration."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class LumenConfig:
    model_dim: int = 256
    delta_layers: int = 4
    vocab_size: int = 4096
    max_vocab_size: int = 8192
    top_k_low: int = 16
    top_k_high: int = 32
    homeostasis_target: float = 0.05
    homeostasis_eta: float = 0.001
    surprise_threshold: float = 0.15
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
