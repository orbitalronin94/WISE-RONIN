import pytest
import torch

from model.neuroplasticity import (
    ELMProbe, IntrinsicThreshold, OneShotGate, SignalContext,
    SimulatedSynapticPruner, SynapticTaggingCapture,
)


def test_signal_context_reuses_detached_surprise_and_derives_uncertainty():
    surprise = torch.tensor([[[0.5]]], requires_grad=True)
    signals = SignalContext.from_surprise(surprise, step=12)
    assert signals.step == 12
    assert not signals.surprise.requires_grad
    assert torch.allclose(signals.uncertainty, torch.tensor([[[1/3]]]))


def test_tag_capture_requires_both_local_tag_and_global_resource():
    organ = SynapticTaggingCapture((2,), decay=0.5, threshold=0.4)
    event = torch.tensor([0.8, 0.1])
    assert torch.equal(organ.update(event, 0.0), torch.zeros(2))
    captured = organ.update(torch.zeros(2), 0.5)
    assert captured.tolist() == pytest.approx([0.5, 0.0])


def test_intrinsic_threshold_is_bounded_and_moves_toward_activity_target():
    organ = IntrinsicThreshold(initial=0.2, target=0.1, rate=0.01)
    before = organ.threshold.clone()
    organ.update(torch.tensor([0.8, 0.8]))
    assert organ.threshold > before
    for _ in range(1000):
        organ.update(torch.ones(4))
    assert organ.threshold <= organ.maximum


def test_simulated_pruning_never_deletes_parameters_and_tracks_usage():
    organ = SimulatedSynapticPruner((3,), usage_decay=0.0, strength_floor=0.1)
    mask = organ.observe(torch.tensor([0.0, 0.5, 0.0]),
                         torch.tensor([0.0, 0.5, 0.2]))
    assert mask.tolist() == [0.0, 1.0, 1.0]


def test_one_shot_gate_is_bounded_and_monotonic():
    gate = OneShotGate(base=0.05, maximum=0.2, gain=1.0)
    rates = gate(torch.tensor([0.0, 0.5, 1.0]))
    assert rates.tolist() == pytest.approx([0.05, 0.125, 0.2])


def test_elm_probe_fits_analytic_readout_without_changing_random_projection():
    torch.manual_seed(7)
    probe = ELMProbe(4, 12, 2, seed=19)
    weight_before = probe.weight.clone()
    x = torch.randn(32, 4)
    y = torch.stack((x[:, 0] + x[:, 1], x[:, 2] - x[:, 3]), dim=-1)
    probe.fit(x, y)
    assert torch.equal(probe.weight, weight_before)
    assert probe(x).shape == y.shape
    assert torch.isfinite(probe(x)).all()
