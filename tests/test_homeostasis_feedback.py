import torch

from model.homeostasis import Homeostasis


def test_suppressed_gate_feedback_lowers_threshold_despite_positive_residual_input():
    organ = Homeostasis(dim=8, m_target=0.05, eta=0.01, k_top=2)
    organ.train()
    organ.bias.data.fill_(2.0)

    output = organ(torch.ones(2, 4, 8))

    assert organ.last_input_positive == 1.0
    assert organ.last_gate_positive == 0.0
    assert torch.equal(output, torch.ones_like(output))
    # No post-gate activity means negative feedback must lower the threshold.
    assert torch.all(organ.bias < 2.0)


def test_homeostatic_feedback_raises_threshold_for_sustained_post_gate_activity():
    organ = Homeostasis(dim=8, m_target=0.05, eta=0.01, k_top=2)
    organ.train()
    x = torch.zeros(2, 4, 8)
    x[..., 0] = 2.0
    x[..., 1] = 1.0

    for _ in range(100):
        organ(x)

    assert organ.effective_m[0] > organ.m_target
    assert organ.effective_m[1] > organ.m_target
    assert organ.bias[0] > 0
    assert organ.bias[1] > 0
    # Units that never survive the gate should not receive a larger threshold.
    assert torch.all(organ.bias[2:] < 0)



def test_homeostatic_residual_gain_is_bounded_by_ten_percent():
    organ = Homeostasis(dim=16, m_target=0.05, eta=0.001, k_top=4)
    x = torch.rand(2, 8, 16) + 0.1

    output = organ(x)

    update_norm = (output - x).norm(dim=-1)
    input_norm = x.norm(dim=-1)
    assert torch.isfinite(output).all()
    assert torch.all(update_norm <= 0.1001 * input_norm)
