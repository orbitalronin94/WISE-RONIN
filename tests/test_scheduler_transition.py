import math

import torch

from train.scheduler_transition import CosineTransitionScheduler


def test_transition_preserves_optimizer_state_and_has_expected_endpoints():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.5e-4)
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    moments_before = {
        key: value.clone()
        for key, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }

    scheduler = CosineTransitionScheduler(
        optimizer, start_step=1000, end_step=2000,
        old_lr=1.47e-4, new_lr=2e-5, warmup_steps=20,
    )
    assert math.isclose(scheduler.get_last_lr()[0], 1.47e-4, rel_tol=1e-8)
    assert math.isclose(scheduler.lr_at_step(1020), 2e-5, rel_tol=1e-8)
    assert math.isclose(scheduler.lr_at_step(2000), 0.0, abs_tol=1e-12)
    for _ in range(20):
        scheduler.step()
    assert math.isclose(scheduler.get_last_lr()[0], 2e-5, rel_tol=1e-8)
    for key, value in moments_before.items():
        assert torch.equal(optimizer.state[parameter][key], value)


def test_transition_state_roundtrip():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4)
    scheduler = CosineTransitionScheduler(
        optimizer, start_step=1000, end_step=2000,
        old_lr=1e-4, new_lr=2e-5, warmup_steps=20,
    )
    for _ in range(250):
        scheduler.step()
    saved = scheduler.state_dict()

    restored = CosineTransitionScheduler(
        optimizer, start_step=1000, end_step=2000,
        old_lr=1e-4, new_lr=2e-5, warmup_steps=20,
    )
    restored.load_state_dict(saved)
    assert restored.current_step == 1250
    assert restored.get_last_lr() == scheduler.get_last_lr()


def test_plateau_then_cosine_decay_keeps_nonzero_lr_floor():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.0)
    scheduler = CosineTransitionScheduler(
        optimizer, start_step=16000, end_step=21000,
        old_lr=0.0, new_lr=3e-5, warmup_steps=100,
        plateau_steps=3900, min_lr_ratio=0.5,
    )
    assert math.isclose(scheduler.lr_at_step(16000), 0.0, abs_tol=1e-12)
    assert math.isclose(scheduler.lr_at_step(16100), 3e-5, rel_tol=1e-8)
    assert math.isclose(scheduler.lr_at_step(20000), 3e-5, rel_tol=1e-8)
    assert math.isclose(scheduler.lr_at_step(20500), 2.25e-5, rel_tol=1e-8)
    assert math.isclose(scheduler.lr_at_step(21000), 1.5e-5, rel_tol=1e-8)
    saved = scheduler.state_dict()
    restored = CosineTransitionScheduler(
        optimizer, start_step=16000, end_step=21000,
        old_lr=0.0, new_lr=3e-5, warmup_steps=100,
        plateau_steps=3900, min_lr_ratio=0.5,
    )
    restored.load_state_dict(saved)
    assert restored.get_last_lr() == scheduler.get_last_lr()
