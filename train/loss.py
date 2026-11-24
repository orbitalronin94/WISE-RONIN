"""Token, JEPA, diversity, router, repetition and optional variational-inspired losses."""
import torch
from torch.nn import functional as F


def repetition_loss_shared_normalization(logits):
    """Exact triple-token repetition probability with one shared log-softmax."""
    log_probs = F.log_softmax(logits, dim=-1)
    log_triple = torch.logsumexp(
        log_probs[:, :-2] + log_probs[:, 1:-1] + log_probs[:, 2:],
        dim=-1,
    )
    return log_triple.exp().mean()


def perdida_total(output, targets, model, use_jepa=True, use_rep=True,
                  use_free_energy=False, beta=0.01, pad_id=0,
                  shared_loss_normalization=False):
    logits = output["logits"]
    _, steps, vocab = logits.shape
    shared_log_probs = None
    if shared_loss_normalization and use_rep and steps >= 3:
        shared_log_probs = F.log_softmax(logits, dim=-1)
        token_loss = F.nll_loss(
            shared_log_probs[:, :-1].reshape(-1, vocab),
            targets[:, 1:].reshape(-1), ignore_index=pad_id,
        )
    else:
        token_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, vocab),
            targets[:, 1:].reshape(-1),
            ignore_index=pad_id,
        )
    latent_loss = logits.new_zeros(())
    if use_jepa and steps > 8:
        hidden = output["hidden"]
        target_emb = model.emb.emb(targets)
        target = target_emb[:, 1:, :].unfold(1, 8, 1).mean(dim=-1)
        prediction = model.predict_latent(hidden[:, :steps - 8])
        latent_loss = F.mse_loss(prediction, target.detach())

    activations = output["activations"].relu()
    frequency = activations.mean(dim=(0, 1))
    frequency = frequency / frequency.sum().clamp_min(1e-8)
    entropy = -(frequency * frequency.clamp_min(1e-8).log()).sum()
    diversity_loss = -entropy

    router_probability = output["router"][..., 0]
    router_loss = -(router_probability - 0.5).abs().mean()

    repetition_loss = logits.new_zeros(())
    if use_rep and steps >= 3:
        if shared_log_probs is None:
            repetition_loss = repetition_loss_shared_normalization(logits)
        else:
            log_triple = torch.logsumexp(
                shared_log_probs[:, :-2] + shared_log_probs[:, 1:-1] + shared_log_probs[:, 2:],
                dim=-1,
            )
            repetition_loss = log_triple.exp().mean()

    local_loss = logits.new_zeros(())
    local_states = output.get("local_hidden", ())
    if use_jepa and local_states and steps > 8:
        local_terms = [
            F.mse_loss(model.predict_latent(state[:, :steps - 8]), target.detach())
            for state in local_states
        ]
        local_loss = torch.stack(local_terms).mean()

    complexity = output.get("free_energy_complexity", logits.new_zeros(()))
    if not use_free_energy:
        complexity = complexity.detach() * 0.0

    total = (
        token_loss + 0.3 * latent_loss + 0.10 * local_loss + 0.01 * diversity_loss
        + 0.01 * router_loss + 0.05 * repetition_loss
        + (beta * complexity if use_free_energy else 0.0)
    )
    return total, {
        "L_token": token_loss.detach(),
        "L_latente": latent_loss.detach(),
        "L_div": diversity_loss.detach(),
        "L_router": router_loss.detach(),
        "L_rep": repetition_loss.detach(),
        "L_local": local_loss.detach(),
        "F_complexity": complexity.detach(),
    }
