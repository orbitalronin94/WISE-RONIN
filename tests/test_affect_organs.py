import pytest

from model.pain import Nociception
from model.pleasure import Recompensa


def test_first_check_only_initializes_reference():
    pain, pleasure = Nociception(), Recompensa()
    assert pain.update(5.0)["pain"] == 0.0
    assert pleasure.update(5.0)["pleasure"] == 0.0


def test_regression_activates_pain_not_pleasure():
    pain, pleasure = Nociception(ema=0.0), Recompensa(ema=0.0)
    pain.update(5.0)
    pleasure.update(5.0)
    p, r = pain.update(6.0), pleasure.update(6.0)
    assert p["pain"] > 0.0 and p["plasticity_scale"] < 1.0
    assert p["gate_scale"] < 1.0
    assert r["pleasure"] == 0.0 and r["plasticity_scale"] == 1.0


def test_improvement_activates_pleasure_not_pain():
    pain, pleasure = Nociception(ema=0.0), Recompensa(ema=0.0)
    pain.update(6.0)
    pleasure.update(6.0)
    p, r = pain.update(5.0), pleasure.update(5.0)
    assert p["pain"] == 0.0
    assert r["pleasure"] > 0.0
    assert r["plasticity_scale"] > 1.0 and r["gate_scale"] > 1.0


def test_small_changes_inside_deadband_do_not_trigger_either_organ():
    pain, pleasure = Nociception(ema=0.0), Recompensa(ema=0.0)
    pain.update(5.0)
    pleasure.update(5.0)
    assert pain.update(5.01)["pain"] == 0.0
    assert pleasure.update(4.99)["pleasure"] == 0.0


def test_signals_are_bounded_and_decay_without_new_stimulus():
    pain = Nociception(ema=0.0, response=1.0, decay=0.5)
    pain.update(5.0)
    assert pain.update(100.0)["pain"] == pytest.approx(1.0)
    assert pain.update(100.0)["pain"] == pytest.approx(0.5)


def test_invalid_loss_is_rejected():
    organ = Recompensa()
    with pytest.raises(ValueError):
        organ.update(float("nan"))
    with pytest.raises(ValueError):
        organ.update(-1.0)


def test_buffers_round_trip_in_state_dict():
    organ = Recompensa(ema=0.0)
    organ.update(5.0)
    organ.update(4.0)
    restored = Recompensa(ema=0.0)
    restored.load_state_dict(organ.state_dict())
    assert restored.state() == organ.state()
