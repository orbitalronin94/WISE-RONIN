"""Explicit, checkpointable LR transition for the 1000 -> 2000 regime change."""
from __future__ import annotations

import math


class CosineTransitionScheduler:
    """Preserve optimizer moments while changing only the LR schedule."""

    def __init__(self, optimizer, *, start_step, end_step, old_lr, new_lr,
                 warmup_steps=20, current_step=None):
        if end_step <= start_step:
            raise ValueError("end_step must be greater than start_step")
        if warmup_steps < 1 or start_step + warmup_steps >= end_step:
            raise ValueError("warmup_steps must fit inside the transition range")
        self.optimizer = optimizer
        self.start_step = int(start_step)
        self.end_step = int(end_step)
        self.old_lr = float(old_lr)
        self.new_lr = float(new_lr)
        self.warmup_steps = int(warmup_steps)
        self.current_step = int(start_step if current_step is None else current_step)
        if not self.start_step <= self.current_step <= self.end_step:
            raise ValueError("current_step is outside the transition range")
        self._apply(self.lr_at_step(self.current_step))

    def lr_at_step(self, step):
        step = int(step)
        if step <= self.start_step:
            return self.old_lr
        transition_end = self.start_step + self.warmup_steps
        if step < transition_end:
            alpha = (step - self.start_step) / self.warmup_steps
            return self.old_lr * (1.0 - alpha) + self.new_lr * alpha
        progress = (step - transition_end) / (self.end_step - transition_end)
        progress = min(1.0, max(0.0, progress))
        return self.new_lr * 0.5 * (1.0 + math.cos(math.pi * progress))

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
            "current_step": self.current_step,
        }

    def load_state_dict(self, state):
        expected = self.state_dict()
        for key in ("kind", "start_step", "end_step", "old_lr", "new_lr", "warmup_steps"):
            if state.get(key) != expected[key]:
                raise ValueError(f"Scheduler transition mismatch for {key}")
        self.current_step = int(state["current_step"])
        if not self.start_step <= self.current_step <= self.end_step:
            raise ValueError("Saved scheduler transition step is out of range")
        self._apply(self.lr_at_step(self.current_step))
