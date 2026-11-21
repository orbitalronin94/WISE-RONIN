"""Explicit fake-int8 forward simulation for selected modules (not weight export)."""
import torch
from torch import nn

def fake_int8_tensor(x):
    scale = x.detach().abs().amax().clamp_min(1e-8) / 127.0
    quantized = torch.round(x / scale).clamp(-127, 127)
    return (quantized * scale - x).detach() + x

class FakeInt8(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, *args, **kwargs):
        return self.module(*[fake_int8_tensor(x) if torch.is_tensor(x) and x.is_floating_point() else x for x in args], **kwargs)
