"""Per-token energy allocation across delta, memory and output heads."""
import torch
from torch import nn


class Presupuesto(nn.Module):
    """Learned per-token allocation with a bounded, loss-feedback glucose reserve."""

    GLUCOSE_STEP = 0.02
    GLUCOSE_CAP = 0.25
    GLUCOSE_DECAY = 0.98
    IMPROVEMENT_THRESHOLD = 0.005

    def __init__(self, dim=256):
        super().__init__()
        self.proj = nn.Linear(dim, 3)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        # Buffers are checkpointed with the model but are not optimizer parameters.
        self.register_buffer("glucose", torch.tensor(0.0))
        self.register_buffer("last_check_loss", torch.tensor(float("nan")))
        # Affect organs modulate only the internal energy budget; LR/optimizer stay untouched.
        self.affect_scale = 1.0

    @torch.no_grad()
    def check_loss_signal(self, mean_loss):
        """Produce one shared normalized check-loss signal for all affect organs."""
        current = torch.as_tensor(
            mean_loss, dtype=self.last_check_loss.dtype,
            device=self.last_check_loss.device,
        )
        if current.numel() != 1 or not torch.isfinite(current) or current.item() < 0:
            raise ValueError("shared check-loss feedback requires one finite non-negative loss")
        previous = self.last_check_loss.clone()
        has_baseline = bool(torch.isfinite(previous).item())
        improvement = (
            float(((previous - current) / previous.clamp_min(1e-8)).item())
            if has_baseline else 0.0
        )
        regression = -improvement
        improved = (
            has_baseline
            and float(current.item()) < float(previous.item()) * (1.0 - self.IMPROVEMENT_THRESHOLD)
        )
        if improved:
            self.glucose.add_(self.GLUCOSE_STEP).clamp_(0.0, self.GLUCOSE_CAP)
        else:
            self.glucose.mul_(self.GLUCOSE_DECAY).clamp_(0.0, self.GLUCOSE_CAP)
        self.last_check_loss.copy_(current)
        return {
            "improved": improved if has_baseline else None,
            "improvement": improvement,
            "regression": regression,
            "reserve": float(self.glucose.item()),
            "has_baseline": has_baseline,
        }

    @torch.no_grad()
    def check_loss(self, mean_loss):
        """Backward-compatible glucose-only view of the shared check signal."""
        signal = self.check_loss_signal(mean_loss)
        return signal["improved"], signal["reserve"]

    def forward(self, x):
        allocation = torch.softmax(self.proj(x), dim=-1)
        # Glucose scales the total available energy; the learned allocator keeps
        # deciding how that energy is divided among Delta, memory and the heads.
        energy = allocation * (1.0 + self.glucose.clamp(0.0, self.GLUCOSE_CAP)) * float(self.affect_scale)
        return energy[..., 0:1], energy[..., 1:2], energy[..., 2:3], energy

    @staticmethod
    def entropy(weights):
        # Energy may sum above one after glucose; entropy is measured on shares.
        shares = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        shares = shares.clamp_min(1e-8)
        return -(shares * shares.log()).sum(-1).mean()
