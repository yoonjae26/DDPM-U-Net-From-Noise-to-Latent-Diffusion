from __future__ import annotations

import torch

from latent_diffusion_playground.diffusion.scheduler import DiffusionSchedule


def q_sample(
    x0: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x0)

    t = timesteps.long().view(-1)
    if t.shape[0] != x0.shape[0]:
        raise ValueError(f"timesteps batch size {t.shape[0]} must match x0 batch size {x0.shape[0]}")

    sqrt_alpha_hat_t = schedule.sqrt_alpha_hat[t].view(-1, *([1] * (x0.ndim - 1)))
    sqrt_one_minus_alpha_hat_t = schedule.sqrt_one_minus_alpha_hat[t].view(-1, *([1] * (x0.ndim - 1)))

    return sqrt_alpha_hat_t * x0 + sqrt_one_minus_alpha_hat_t * noise
