#!/usr/bin/env python3
"""Instantiate and verify the matched-size LUMEN-WISE x5 and Transformer pair."""
from __future__ import annotations
import json
from pathlib import Path
from model.config import LumenConfig
from model.lumen import RoninLumen
from train.transformer_baseline import SmallTransformerLM

ROOT = Path(__file__).resolve().parents[1]

def count(model):
    return sum(parameter.numel() for parameter in model.parameters())

def main():
    lumen_cfg = LumenConfig(**json.loads((ROOT / "config/lumen_wise_x5.json").read_text()))
    transformer_cfg = json.loads((ROOT / "config/transformer_wise_x5.json").read_text())
    lumen = RoninLumen(config=lumen_cfg, v32_ze=True)
    lumen_base = RoninLumen(config=LumenConfig(), v32_ze=True)
    transformer = SmallTransformerLM(
        transformer_cfg["vocab_size"], transformer_cfg["max_sequence_length"],
        transformer_cfg["model_dim"], transformer_cfg["heads"], transformer_cfg["ff_width"],
    )
    transformer_base = SmallTransformerLM(4096, 128, 192, 4, 576)
    lumen_n, lumen_base_n = count(lumen), count(lumen_base)
    transformer_n, transformer_base_n = count(transformer), count(transformer_base)
    lumen_ratio = lumen_n / lumen_base_n
    transformer_ratio = transformer_n / transformer_base_n
    pair_delta = abs(lumen_n - transformer_n) / max(lumen_n, transformer_n)
    result = {
        "lumen": {"params": lumen_n, "baseline_params": lumen_base_n, "scale": lumen_ratio,
                  "model_dim": lumen_cfg.model_dim, "delta_rank": lumen_cfg.delta_rank,
                  "ze_dim": lumen_cfg.ze_dim, "v32_ze": True},
        "transformer": {"params": transformer_n, "baseline_params": transformer_base_n,
                        "scale": transformer_ratio, "model_dim": transformer_cfg["model_dim"],
                        "heads": transformer_cfg["heads"], "ff_width": transformer_cfg["ff_width"]},
        "pair_parameter_delta": pair_delta,
        "shared_protocol": {"steps": 25000, "batch": 8, "sequence_length": 128,
                            "seed": 42, "learning_rate": 0.00015, "warmup_steps": 100,
                            "weight_decay": 0.01, "vocab_size": 4096},
    }
    print(json.dumps(result, indent=2))
    assert 4.9 <= lumen_ratio <= 5.15, lumen_ratio
    assert 4.9 <= transformer_ratio <= 5.15, transformer_ratio
    assert pair_delta <= 0.05, pair_delta
    assert lumen_cfg.vocab_size == transformer_cfg["vocab_size"]
    assert lumen_cfg.total_steps == transformer_cfg["total_steps"]
    assert lumen_cfg.batch_size == transformer_cfg["batch_size"]
    assert lumen_cfg.seed == transformer_cfg["seed"]

if __name__ == "__main__":
    main()
