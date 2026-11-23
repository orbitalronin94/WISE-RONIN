"""Shared, bounded neuroplasticity primitives; experimental mechanisms stay opt-in."""
from dataclasses import dataclass
import torch
from torch import nn


@dataclass(frozen=True)
class SignalContext:
    """Immutable per-forward signals; consumers share tensors, never recompute them."""
    surprise: torch.Tensor
    uncertainty: torch.Tensor
    step: int

    @classmethod
    def from_surprise(cls, surprise, step=0):
        if not torch.isfinite(surprise).all():
            raise ValueError("surprise signal must be finite")
        uncertainty = (surprise / (1.0 + surprise.abs())).detach()
        return cls(surprise=surprise.detach(), uncertainty=uncertainty, step=int(step))


class SynapticTaggingCapture(nn.Module):
    """Local eligibility tags captured only when a global resource is available."""
    def __init__(self, shape, decay=0.95, threshold=0.01):
        super().__init__()
        self.decay = float(decay)
        self.threshold = float(threshold)
        if not 0.0 <= self.decay < 1.0 or self.threshold < 0:
            raise ValueError("invalid tagging configuration")
        self.register_buffer("tag", torch.zeros(shape))
        self.register_buffer("captured", torch.zeros(shape))

    @torch.no_grad()
    def update(self, local_event, global_resource):
        event = torch.as_tensor(local_event, device=self.tag.device, dtype=self.tag.dtype)
        if event.shape != self.tag.shape or not torch.isfinite(event).all():
            raise ValueError("local tag event must be finite and match tag shape")
        resource = float(global_resource)
        if not 0.0 <= resource <= 1.0:
            raise ValueError("global resource must be in [0,1]")
        self.tag.mul_(self.decay).add_(event.clamp_min(0.0))
        eligible = self.tag >= self.threshold
        captured = eligible & (resource > 0.0)
        self.captured.copy_(captured.to(self.captured.dtype) * resource)
        return self.captured.clone()


class IntrinsicThreshold(nn.Module):
    """Slow bounded threshold adaptation driven by effective activity."""
    def __init__(self, initial=0.15, target=0.05, rate=0.001,
                 minimum=0.01, maximum=0.8):
        super().__init__()
        if not 0 < minimum <= initial <= maximum or not 0 <= target <= 1 or not 0 <= rate <= 0.1:
            raise ValueError("invalid intrinsic-threshold configuration")
        self.target, self.rate = float(target), float(rate)
        self.minimum, self.maximum = float(minimum), float(maximum)
        self.register_buffer("threshold", torch.tensor(float(initial)))

    @torch.no_grad()
    def update(self, effective_activity):
        value = torch.as_tensor(effective_activity, dtype=self.threshold.dtype,
                                device=self.threshold.device)
        if not torch.isfinite(value).all():
            raise ValueError("effective activity must be finite")
        error = value.mean().clamp(0.0, 1.0) - self.target
        self.threshold.add_(self.rate * error).clamp_(self.minimum, self.maximum)
        return self.threshold.clone()


class SimulatedSynapticPruner(nn.Module):
    """Non-destructive usage/strength mask; never deletes trained parameters."""
    def __init__(self, shape, usage_decay=0.99, strength_floor=1e-4):
        super().__init__()
        self.usage_decay = float(usage_decay)
        self.strength_floor = float(strength_floor)
        if not 0 <= self.usage_decay < 1 or self.strength_floor < 0:
            raise ValueError("invalid pruning configuration")
        self.register_buffer("usage", torch.zeros(shape))
        self.register_buffer("mask", torch.ones(shape))

    @torch.no_grad()
    def observe(self, activity, strength):
        activity = torch.as_tensor(activity, device=self.usage.device, dtype=self.usage.dtype)
        strength = torch.as_tensor(strength, device=self.usage.device, dtype=self.usage.dtype)
        if activity.shape != self.usage.shape or strength.shape != self.usage.shape:
            raise ValueError("pruning observations must match configured shape")
        if not torch.isfinite(activity).all() or not torch.isfinite(strength).all():
            raise ValueError("pruning observations must be finite")
        self.usage.mul_(self.usage_decay).add_(activity.clamp_min(0), alpha=1-self.usage_decay)
        candidate = (self.usage < self.strength_floor) & (strength.abs() < self.strength_floor)
        self.mask.copy_((~candidate).to(self.mask.dtype))
        return self.mask.clone()


class OneShotGate:
    """Bounded local fast-learning rate; uncertainty cannot alter global optimizer LR."""
    def __init__(self, base=0.05, maximum=0.20, gain=1.0):
        if not 0 <= base <= maximum <= 1 or not 0 <= gain <= 10:
            raise ValueError("invalid one-shot gate configuration")
        self.base, self.maximum, self.gain = float(base), float(maximum), float(gain)

    def __call__(self, uncertainty):
        value = torch.as_tensor(uncertainty).detach().clamp(0.0, 1.0)
        return (self.base + self.gain * value * (self.maximum-self.base)).clamp(
            self.base, self.maximum
        )


class ELMProbe(nn.Module):
    """Frozen random-feature probe with an analytic ridge readout; offline only."""
    def __init__(self, input_dim, hidden_dim, output_dim, seed=42):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        weight = torch.randn(hidden_dim, input_dim, generator=generator) / input_dim**0.5
        bias = torch.randn(hidden_dim, generator=generator)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)
        self.register_buffer("readout", torch.zeros(output_dim, hidden_dim))
        self.fitted = False

    def features(self, x):
        return torch.tanh(torch.nn.functional.linear(x, self.weight, self.bias))

    @torch.no_grad()
    def fit(self, x, targets, ridge=1e-3):
        if ridge <= 0:
            raise ValueError("ridge must be positive")
        h = self.features(x).reshape(-1, self.weight.shape[0]).double()
        y = targets.reshape(-1, targets.shape[-1]).double()
        gram = h.T @ h + ridge * torch.eye(h.shape[1], device=h.device, dtype=h.dtype)
        beta = torch.linalg.solve(gram, h.T @ y).T
        self.readout.copy_(beta.to(self.readout.dtype))
        self.fitted = True
        return self

    def forward(self, x):
        if not self.fitted:
            raise RuntimeError("ELM probe must be fitted offline before inference")
        return torch.nn.functional.linear(self.features(x), self.readout)
