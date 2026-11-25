#!/usr/bin/env python3
"""One-update synthetic train/resume smoke test for both x5 trainers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[1]
VOCAB_SIZE = 4096
TOKENIZER_SHA = "d7869c11809a28085d5156638086c4833a1dcbacfda4a60b69ec71e26ed3fefe"


def build_inputs(root: Path):
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2}
    vocab.update({f"token_{i}": i for i in range(3, VOCAB_SIZE)})
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = root / "synthetic-tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    rows = [
        {"id": f"smoke-{i}", "prompt": "", "completion": f"texto de prueba {i} para reanudar",
         "task": "raw_text_pretraining", "language": "es", "source": "synthetic_resume_smoke"}
        for i in range(16)
    ]
    data_path = root / "synthetic.jsonl"
    data_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                         encoding="utf-8")
    return data_path, tokenizer_path


def run(command):
    result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env={**os.environ, "PYTHONPATH": str(ROOT)})
    if result.returncode:
        raise RuntimeError("$ " + " ".join(map(str, command)) + "\n" + result.stdout[-12000:])
    print(result.stdout[-1800:])


def verify(path: Path, expected_step: int, expected_params: int, tokenizer_sha: str):
    import torch
    state = torch.load(path, map_location="cpu", weights_only=False)
    assert int(state["step"]) == expected_step
    assert int(state["params"]) == expected_params
    assert state["tokenizer_sha256"] == tokenizer_sha
    assert state["optimizer"] and state["scheduler"] and state["rng"]
    assert state["data_sha256"]
    return state


def main():
    import torch
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix="wise-x5-resume-") as temp:
        root = Path(temp)
        data, tokenizer = build_inputs(root)
        tokenizer_sha = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        assert tokenizer_sha != TOKENIZER_SHA  # synthetic tokenizer is intentionally isolated

        lumen_dir = root / "lumen"
        common_lumen = [
            sys.executable, "-m", "train.train", "--steps", "1", "--phase", "1",
            "--pilot-all-rows", "--v32-ze", "--model-config", "config/lumen_wise_x5.json",
            "--data", str(data), "--tokenizer", str(tokenizer), "--max-len", "8",
            "--log-every", "1", "--checkpoint-every", "1", "--parameter-check-every", "1",
            "--ckpt-dir", str(lumen_dir), "--runtime-profile", "config/lumen_cpu_profile.json",
        ]
        run(common_lumen)
        lumen_step1 = lumen_dir / "step_1.pt"
        state = verify(lumen_step1, 1, 12022016, tokenizer_sha)
        assert state["architecture_id"] == "ronin_lumen_v3_3_neuroplasticity"
        run(common_lumen + ["--resume", str(lumen_step1)])
        verify(lumen_dir / "step_2.pt", 2, 12022016, tokenizer_sha)

        transformer_dir = root / "transformer"
        common_transformer = [
            sys.executable, "-m", "train.transformer_baseline", "--steps", "1",
            "--model-dim", "488", "--heads", "8", "--ff-width", "1464",
            "--data", str(data), "--tokenizer", str(tokenizer), "--batch", "8",
            "--max-len", "128", "--seed", "42", "--threads", "4", "--interop-threads", "2",
            "--disable-mkldnn", "--disable-foreach", "--cpu-affinity",
            "--lr", "1.5e-4", "--warmup", "100", "--total-steps", "25000",
            "--parameter-check-every", "1", "--checkpoint-every", "1",
            "--ckpt-dir", str(transformer_dir),
        ]
        run(common_transformer)
        transformer_step1 = transformer_dir / "step_1.pt"
        state = verify(transformer_step1, 1, 11611472, tokenizer_sha)
        assert state["architecture_id"] == "decoder_transformer_4l_d488_h8_ff1464_tied_v1"
        run(common_transformer + ["--resume", str(transformer_step1)])
        verify(transformer_dir / "step_2.pt", 2, 11611472, tokenizer_sha)

    print("x5 synthetic train/checkpoint/resume smoke passed for both architectures (0→1→2).")


if __name__ == "__main__":
    main()
