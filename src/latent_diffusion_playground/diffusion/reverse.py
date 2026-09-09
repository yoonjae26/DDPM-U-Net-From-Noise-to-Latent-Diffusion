from __future__ import annotations

import torch

from latent_diffusion_playground.diffusion.scheduler import DiffusionSchedule


def ddim_sample_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    t_prev: torch.Tensor,
    epsilon: torch.Tensor,
    schedule: DiffusionSchedule,
    eta: float = 0.0,
    clip_x0: bool = True,
    clip_range: float = 4.0,
) -> torch.Tensor:
    """DDIM reverse step (Song et al. 2020).

    eta=0 → fully deterministic; eta=1 → matches DDPM variance.
    Supports large timestep skips (sub-sequence sampling).

    clip_x0 (dynamic thresholding) bounds the predicted x0 at every step to
    `[-clip_range, clip_range]`. Without it, the huge 1/sqrt(alpha_hat_t)
    gain near t≈T (cosine schedule betas approach `cosine_max_beta`) amplifies
    any small model prediction error into a runaway trajectory that decodes
    to pure noise.
    """
    t = t.long().view(-1)
    t_prev = t_prev.long().view(-1)

    alpha_hat_t    = schedule.alpha_hat[t].view(-1, *([1] * (x_t.ndim - 1)))
    alpha_hat_prev = schedule.alpha_hat[t_prev].view(-1, *([1] * (x_t.ndim - 1)))

    # Predicted x0 from noisy sample and model noise estimate
    x0_pred = (x_t - torch.sqrt(1.0 - alpha_hat_t) * epsilon) / torch.clamp(torch.sqrt(alpha_hat_t), min=1e-8)

    if clip_x0:
        x0_pred = torch.clamp(x0_pred, -clip_range, clip_range)
        epsilon = (x_t - torch.sqrt(alpha_hat_t) * x0_pred) / torch.clamp(torch.sqrt(1.0 - alpha_hat_t), min=1e-8)

    if eta > 0.0:
        # Stochastic DDIM: variance follows DDPM-like schedule scaled by eta
        sigma_sq = eta ** 2 * (
            (1.0 - alpha_hat_prev) / torch.clamp(1.0 - alpha_hat_t, min=1e-8)
            * torch.clamp(1.0 - alpha_hat_t / torch.clamp(alpha_hat_prev, min=1e-8), min=0.0)
        )
        sigma_t  = torch.sqrt(torch.clamp(sigma_sq, min=0.0))
        dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_hat_prev - sigma_sq, min=0.0))
        x_prev = torch.sqrt(alpha_hat_prev) * x0_pred + dir_coeff * epsilon
        noise = torch.randn_like(x_t)
        nonzero = (t_prev > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
        x_prev = x_prev + nonzero * sigma_t * noise
    else:
        # Deterministic DDIM (eta=0)
        dir_coeff = torch.sqrt(torch.clamp(1.0 - alpha_hat_prev, min=0.0))
        x_prev = torch.sqrt(alpha_hat_prev) * x0_pred + dir_coeff * epsilon

    return x_prev


def predict_x0_from_epsilon(
    x_t: torch.Tensor,
    t: torch.Tensor,
    epsilon: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    t = t.long().view(-1)
    sqrt_alpha_hat_t = schedule.sqrt_alpha_hat[t].view(-1, *([1] * (x_t.ndim - 1)))
    sqrt_one_minus_alpha_hat_t = schedule.sqrt_one_minus_alpha_hat[t].view(-1, *([1] * (x_t.ndim - 1)))
    return (x_t - sqrt_one_minus_alpha_hat_t * epsilon) / torch.clamp(sqrt_alpha_hat_t, min=1e-8)


def p_sample_step(
    x_t: torch.Tensor,
    t: torch.Tensor,
    epsilon: torch.Tensor,
    schedule: DiffusionSchedule,
    clip_x0: bool = True,
    clip_range: float = 4.0,
) -> torch.Tensor:
    """DDPM ancestral sampling step (Ho et al. 2020).

    clip_x0 (dynamic thresholding) bounds the predicted x0 at every step to
    `[-clip_range, clip_range]` before it's folded back into epsilon. Without
    it, the huge 1/sqrt(alpha_t) gain near t≈T (cosine schedule betas approach
    `cosine_max_beta`) amplifies any small model prediction error into a
    runaway trajectory that decodes to pure noise instead of an image.
    """
    t = t.long().view(-1)

    beta_t = schedule.betas[t].view(-1, *([1] * (x_t.ndim - 1)))
    alpha_t = schedule.alphas[t].view(-1, *([1] * (x_t.ndim - 1)))
    sqrt_alpha_hat_t = schedule.sqrt_alpha_hat[t].view(-1, *([1] * (x_t.ndim - 1)))
    sqrt_one_minus_alpha_hat_t = schedule.sqrt_one_minus_alpha_hat[t].view(-1, *([1] * (x_t.ndim - 1)))
    alpha_hat_t = schedule.alpha_hat[t].view(-1, *([1] * (x_t.ndim - 1)))

    prev_t = torch.clamp(t - 1, min=0)
    alpha_hat_prev = schedule.alpha_hat[prev_t].view(-1, *([1] * (x_t.ndim - 1)))

    if clip_x0:
        x0_pred = (x_t - sqrt_one_minus_alpha_hat_t * epsilon) / torch.clamp(sqrt_alpha_hat_t, min=1e-8)
        x0_pred = torch.clamp(x0_pred, -clip_range, clip_range)
        epsilon = (x_t - sqrt_alpha_hat_t * x0_pred) / torch.clamp(sqrt_one_minus_alpha_hat_t, min=1e-8)

    mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.clamp(sqrt_one_minus_alpha_hat_t, min=1e-8)) * epsilon)

    beta_tilde = beta_t * (1.0 - alpha_hat_prev) / torch.clamp(1.0 - alpha_hat_t, min=1e-8)
    noise = torch.randn_like(x_t)
    nonzero_mask = (t > 0).float().view(-1, *([1] * (x_t.ndim - 1)))
    return mean + nonzero_mask * torch.sqrt(torch.clamp(beta_tilde, min=1e-12)) * noise
