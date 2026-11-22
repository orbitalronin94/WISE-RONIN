"""Checkpointable LR transition with warmup, optional plateau and cosine floor."""
from __future__ import annotations

import math


class CosineTransitionScheduler:
    """Change only LR; optimizer moments remain owned by the optimizer."""

    def __init__(self, optimizer, *, start_step, end_step, old_lr, new_lr,
                 warmup_steps=20, plateau_steps=0, min_lr_ratio=0.0,
                 current_step=None):
        if end_step <= start_step:
            raise ValueError("end_step must be greater than start_step")
        if warmup_steps < 1 or plateau_steps < 0:
            raise ValueError("warmup_steps must be positive and plateau_steps non-negative")
        if start_step + warmup_steps + plateau_steps >= end_step:
            raise ValueError("warmup and plateau must leave at least one decay step")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be between zero and one")
        self.optimizer = optimizer
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.old_lr = float(old_lr)
        self.new_lr = float(new_lr)
        self.warmup_steps = int(warmup_steps)
        self.plateau_steps = int(plateau_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        self.current_step = int(start_step if current_step is None else current_step)
        if not self.start_step <= self.current_step <= self.end_step:
            raise ValueError("current_step is outside the transition range")
        self._apply(self.lr_at_step(self.current_step))

    @property
    def decay_start_step(self):
        return self.start_step + self.warmup_steps + self.plateau_steps

    def lr_at_step(self, step):
        step = int(step)
        if step <= self.start_step:
            return self.old_lr
        warmup_end = self.start_step + self.warmup_steps
        if step < warmup_end:
            alpha = (step - self.start_step) / self.warmup_steps
            return self.old_lr * (1.0 - alpha) + self.new_lr * alpha
        if step <= self.decay_start_step:
            return self.new_lr
        progress = (step - self.decay_start_step) / (self.end_step - self.decay_start_step)
        progress = min(1.0, max(0.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.new_lr * (self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine)

    def _apply(self, lr):
        for group in self.optimizer.param_groups:
            group["lr"] = float(lr)

    def step(self):
        self.current_step = min(self.end_step, self.current_step + 1)
        self._apply(self.lr_at_step(self.current_step))

    def get_last_lr(self):
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self):
        return {
            "kind": "cosine_transition_v1",
            "start_step": self.start_step,
            "end_step": self.end_step,
            "old_lr": self.old_lr,
            "new_lr": self.new_lr,
            "warmup_steps": self.warmup_steps,
            "plateau_steps": self.plateau_steps,
            "min_lr_ratio": self.min_lr_ratio,
            "current_step": self.current_step,
        }

    def load_state_dict(self, state):
        defaults = {"plateau_steps": 0, "min_lr_ratio": 0.0}
        expected = self.state_dict()
        for key in ("kind", "start_step", "end_step", "old_lr", "new_lr",
                    "warmup_steps", "plateau_steps", "min_lr_ratio"):
            if state.get(key, defaults.get(key)) != expected[key]:
                raise ValueError(f"Scheduler transition mismatch for {key}")
        self.current_step = int(state["current_step"])
        if not self.start_step <= self.current_step <= self.end_step:
            raise ValueError("Saved transition step is out of range")
        self._apply(self.lr_at_step(self.current_step))
