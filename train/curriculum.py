"""Deterministic curriculum selection; phase 2 is stratified 70/30."""
import random

def datos_para_fase(rows, phase):
    if phase == 1:
        selected = [r for r in rows if r.get("dificil") is True]
        if not selected:
            raise ValueError("Phase 1 requires explicit difficult=true labels")
        return selected
    if phase == 2:
        easy = [r for r in rows if r.get("dificil") is False]
        hard = [r for r in rows if r.get("dificil") is True]
        if not easy or not hard:
            raise ValueError("Phase 2 requires both explicit easy and difficult labels")
        return {"easy": easy, "hard": hard}
    if phase == 3:
        selected = [r for r in rows if r.get("dificil") is True or r.get("holdout_like") is True]
        if not selected:
            raise ValueError("Phase 3 requires difficult or train-derived holdout_like rows")
        if any(r.get("split") == "holdout" for r in selected):
            raise ValueError("Real holdout records are forbidden in training")
        return selected
    raise ValueError("phase must be 1, 2 or 3")

def sample_phase2(groups, count, rng):
    easy_n = round(count * 0.70)
    hard_n = count - easy_n
    easy = rng.choices(groups["easy"], k=easy_n)
    hard = rng.choices(groups["hard"], k=hard_n)
    batch = easy + hard
    rng.shuffle(batch)
    return batch
