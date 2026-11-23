import pytest
import torch
from model.pain import Nociception
from model.pleasure import Recompensa


def test_initial_check_only_sets_reference():
    assert Nociception().update(5.0)["pain"] == 0.0
    assert Recompensa().update(5.0)["pleasure"] == 0.0


def test_regression_and_improvement_are_independent():
    pain, reward = Nociception(ema=0.0), Recompensa(ema=0.0)
    pain.update(5.0); reward.update(5.0)
    assert pain.update(6.0)["pain"] > 0
    assert reward.update(6.0)["pleasure"] == 0
    pain2, reward2 = Nociception(ema=0.0), Recompensa(ema=0.0)
    pain2.update(6.0); reward2.update(6.0)
    assert pain2.update(5.0)["pain"] == 0
    assert reward2.update(5.0)["pleasure"] > 0


def test_deadband_and_bounded_decay():
    p = Nociception(ema=0.0, response=1.0, decay=0.5)
    p.update(5.0)
    assert p.update(5.01)["pain"] == 0
    assert p.update(100.0)["pain"] == pytest.approx(1.0)
    assert p.update(100.0)["pain"] == pytest.approx(0.5)


def test_invalid_values_rejected_and_state_persists():
    organ = Recompensa(ema=0.0)
    with pytest.raises(ValueError): organ.update(float("nan"))
    with pytest.raises(ValueError): organ.update(-1.0)
    organ.update(5.0); organ.update(4.0)
    restored = Recompensa(ema=0.0)
    restored.load_state_dict(organ.state_dict())
    assert restored.state() == organ.state()


def test_glucose_pain_and_pleasure_consume_one_shared_check_loss_signal():
    from model.energy import Presupuesto

    budget = Presupuesto(dim=4)
    pain, pleasure = Nociception(ema=0.0), Recompensa(ema=0.0)
    first = budget.check_loss_signal(5.0)
    assert first["has_baseline"] is False
    pain.update_signal(first["regression"])
    pleasure.update_signal(first["improvement"])

    second = budget.check_loss_signal(4.0)
    assert second["has_baseline"] is True
    assert second["improvement"] == pytest.approx(0.2)
    assert second["regression"] == pytest.approx(-0.2)
    pain_state = pain.update_signal(second["regression"])
    pleasure_state = pleasure.update_signal(second["improvement"])
    assert pain_state["pain"] == 0.0
    assert pleasure_state["pleasure"] > 0.0
    assert torch.isfinite(budget.glucose)
