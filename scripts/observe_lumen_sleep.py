#!/usr/bin/env python3
"""Run the real LUMEN sleep routine on a copy of a promoted checkpoint.

The parent checkpoint is never overwritten. The report contains greedy samples
from the same prompts before and after sleep; prompts are diagnostic, not a holdout.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.lumen import RoninLumen
from train.sleep import sueno

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


PROMPTS = [
    "¿Qué has aprendido hasta ahora y cómo lo explicarías?",
    "Explica qué es la memoria y para qué sirve.",
    "¿Qué relación hay entre aprender y descansar?",
    "¿Qué haces cuando no sabes la respuesta?",
    "Explica una contradicción con un ejemplo.",
    "Escribe una reflexión breve sobre el conocimiento.",
]


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 72):
    bos = tokenizer.token_to_id("[BOS]")
    eos = tokenizer.token_to_id("[EOS]")
    if bos is None or eos is None:
        raise RuntimeError("Tokenizer is missing [BOS] or [EOS].")
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    sequence = [bos] + ids[-127:]
    generated = []
    model.reset_state()
    model.eval()
    with torch.no_grad():
        for step in range(max_new_tokens):
            context = sequence[-128:] if step == 0 else [sequence[-1]]
            batch = torch.tensor([context], dtype=torch.long)
            output = model(batch, reset_state=(step == 0), collect_diagnostics=False)
            token = int(output["logits"][0, -1].argmax().item())
            if token == eos:
                break
            generated.append(token)
            sequence.append(token)
    return {
        "text": tokenizer.decode(generated, skip_special_tokens=True),
        "generated_tokens": len(generated),
        "eos_terminated": len(generated) < max_new_tokens,
    }


def main():
    checkpoint_path = Path("artifacts/checkpoint/checkpoints/lumen/step_229700.pt")
    tokenizer_path = Path("artifacts/inputs/artifacts/tokenizer.json")
    output_path = Path("artifacts/sleep-observation/step_229700_sleep_observation.json")
    if not checkpoint_path.is_file() or not tokenizer_path.is_file():
        raise FileNotFoundError("Checkpoint or frozen tokenizer artifact is missing.")

    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state.get("step") != 229700:
        raise RuntimeError(f"Expected step 229700, got {state.get('step')}")
    tokenizer_sha = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
    if tokenizer_sha != state.get("tokenizer_sha256"):
        raise RuntimeError("Tokenizer SHA-256 does not match checkpoint provenance.")

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    model = RoninLumen(v32_ze=bool(state.get("v32_ze", False))).cpu()
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    memory = model.memoria
    occupied_before = int((memory.contador_lenta > 0).sum().item())
    if occupied_before == 0:
        raise RuntimeError("The promoted checkpoint has no consolidated slow-memory records; real sleep would be a no-op.")

    before_mlp = {k: v.detach().clone() for k, v in memory.mlp.state_dict().items()}
    before_samples = [
        {"prompt": prompt, **generate(model, tokenizer, prompt)}
        for prompt in PROMPTS
    ]

    torch.manual_seed(20261123)
    started = time.perf_counter()
    sleep_loss = float(sueno(model, device="cpu"))
    sleep_seconds = time.perf_counter() - started
    after_mlp = memory.mlp.state_dict()
    changed_mlp = [
        key for key, value in after_mlp.items()
        if not torch.equal(before_mlp[key], value)
    ]
    occupied_after = int((memory.contador_lenta > 0).sum().item())

    after_samples = [
        {"prompt": prompt, **generate(model, tokenizer, prompt)}
        for prompt in PROMPTS
    ]

    report = {
        "schema_version": 1,
        "experiment": "lumen_real_sleep_observation",
        "checkpoint_step": int(state["step"]),
        "checkpoint_sha256": checkpoint_sha,
        "tokenizer_sha256": tokenizer_sha,
        "parent_checkpoint_modified": False,
        "sleep": {
            "routine": "train.sleep.sueno",
            "replay_loss": sleep_loss,
            "duration_seconds": round(sleep_seconds, 6),
            "consolidated_records_before": occupied_before,
            "consolidated_records_after": occupied_after,
            "memory_mlp_parameters_changed": changed_mlp,
            "updates": 3,
            "seed": 20261123,
        },
        "sampling": {
            "method": "greedy",
            "max_new_tokens": 72,
            "prompts_are_diagnostic_not_holdout": True,
            "before_sleep": before_samples,
            "after_sleep": after_samples,
        },
        "interpretation_guard": (
            "These are generated samples before and after the repository's actual sleep routine. "
            "They are not introspection or evidence of consciousness, and this experiment does not "
            "update or promote any training checkpoint."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
