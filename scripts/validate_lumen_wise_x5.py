#!/usr/bin/env python3
"""Instantiate and verify the matched-size LUMEN-WISE x5 and Transformer pair."""
from __future__ import annotations
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from model.config import LumenConfig
from model.lumen import RoninLumen
from train.transformer_baseline import SmallTransformerLM

ROOT = Path(__file__).resolve().parents[1]

def count(model):
    return sum(parameter.numel() for parameter in model.parameters())

def main():
    lumen_cfg = LumenConfig(**json.loads((ROOT / "config/lumen_wise_x5.json").read_text()))
    transformer_cfg = json.loads((ROOT / "config/transformer_wise_x5.json").read_text())
    runtime_cfg = json.loads((ROOT / "config/lumen_cpu_profile.json").read_text())
    expected_lumen = {
        "vocab_size": 4096, "sequence_length": 128, "batch_size": 8,
        "total_steps": 25000, "learning_rate": 0.00015,
        "warmup_steps": 100, "seed": 42, "weight_decay": 0.01,
    }
    expected_transformer = {
        "vocab_size": 4096, "max_sequence_length": 128, "model_dim": 488,
        "heads": 8, "ff_width": 1464, "batch_size": 8, "total_steps": 25000,
        "learning_rate": 0.00015, "warmup_steps": 100, "seed": 42,
        "weight_decay": 0.01, "tied_embeddings": True, "layers": 4,
    }
    for key, value in expected_lumen.items():
        assert getattr(lumen_cfg, key) == value, (key, getattr(lumen_cfg, key), value)
    for key, value in expected_transformer.items():
        assert transformer_cfg.get(key) == value, (key, transformer_cfg.get(key), value)
    for key, value in {"threads": 4, "interop_threads": 2, "mkldnn_enabled": False,
                       "foreach": False, "cpu_affinity": True}.items():
        assert runtime_cfg.get(key) == value, (key, runtime_cfg.get(key), value)
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

    # Synthetic CPU forward/backward smoke tests; no corpus or training run.
    torch.manual_seed(42)
    token_ids = torch.randint(4, lumen_cfg.vocab_size, (1, 8))
    lumen.train()
    lumen.reset_states()
    lumen_logits = lumen(token_ids, reset_state=True, collect_diagnostics=False)["logits"]
    lumen_loss = F.cross_entropy(lumen_logits[:, :-1].reshape(-1, lumen_cfg.vocab_size),
                                 token_ids[:, 1:].reshape(-1))
    lumen_loss.backward()
    assert torch.isfinite(lumen_loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in lumen.parameters())

    transformer.train()
    transformer_logits = transformer(token_ids)
    transformer_loss = F.cross_entropy(transformer_logits[:, :-1].reshape(-1, 4096),
                                       token_ids[:, 1:].reshape(-1))
    transformer_loss.backward()
    assert torch.isfinite(transformer_loss)
    assert 5.0 < float(transformer_loss.detach()) < 12.0, float(transformer_loss.detach())
    assert 5.0 < float(lumen_loss.detach()) < 12.0, float(lumen_loss.detach())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in transformer.parameters())
    print(json.dumps({"synthetic_forward_backward": {
        "lumen_loss": float(lumen_loss.detach()), "transformer_loss": float(transformer_loss.detach()),
        "device": "cpu", "sequence_length": 8
    }}, indent=2))

if __name__ == "__main__":
    main()
