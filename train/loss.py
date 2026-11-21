"""Combined token, latent-prediction, diversity and router objectives."""
import torch
from torch.nn import functional as F

def perdida_total(output, targets, model, use_jepa=True, use_rep=True, pad_id=0):
    logits = output["logits"]
    B, T, V = logits.shape
    token_loss = F.cross_entropy(logits[:, :-1].reshape(-1, V),
                                 targets[:, 1:].reshape(-1), ignore_index=pad_id)
    latent_loss = logits.new_zeros(())
    if use_jepa and T > 8:
        hidden = output["hidden"]
        target_emb = model.emb.emb(targets[:, 1:])
        windows = [target_emb[:, i+1:i+9].mean(1) for i in range(T-8)]
        target = torch.stack(windows, dim=1)
        pred = model.predict_latent(hidden[:, :T-8])
        latent_loss = F.mse_loss(pred, target.detach())
    activations = output["activations"].relu()
    freq = activations.mean((0, 1))
    freq = freq / freq.sum().clamp_min(1e-8)
    entropy = -(freq * freq.clamp_min(1e-8).log()).sum()
    diversity_loss = -entropy
    router_p = output["router"][..., 0]
    router_loss = -(router_p - 0.5).abs().mean()
    repetition_loss = logits.new_zeros(())
    if use_rep and T >= 3:
        probs = torch.softmax(logits, dim=-1)
        repeated = (probs[:, :-2] * probs[:, 1:-1] * probs[:, 2:]).sum(-1)
        repetition_loss = repeated.mean()
    total = token_loss + 0.3 * latent_loss + 0.01 * diversity_loss + 0.01 * router_loss + 0.05 * repetition_loss
    return total, {"L_token": token_loss.detach(), "L_latente": latent_loss.detach(),
                   "L_div": diversity_loss.detach(), "L_router": router_loss.detach(),\n                   "L_rep": repetition_loss.detach()}
