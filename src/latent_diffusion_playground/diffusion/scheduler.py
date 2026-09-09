from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DiffusionSchedule:
    name: str
    timesteps: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_hat: torch.Tensor
    sqrt_alpha_hat: torch.Tensor
    sqrt_one_minus_alpha_hat: torch.Tensor


def _linear_betas(timesteps: int, beta_start: float, beta_end: float, device: torch.device) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, device=device, dtype=torch.float32)


def _cosine_betas(timesteps: int, s: float, max_beta: float, device: torch.device) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, device=device, dtype=torch.float32) / timesteps
    alpha_bar = torch.cos(((t + s) / (1 + s)) * math.pi * 0.5).pow(2)
    alpha_bar = alpha_bar / alpha_bar[0]

    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return torch.clamp(betas, min=1e-8, max=max_beta)


def make_schedule(
    name: str,
    timesteps: int,
    device: torch.device,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
    cosine_s: float = 0.008,
    cosine_max_beta: float = 0.999,
) -> DiffusionSchedule:
    if timesteps <= 0:
        raise ValueError(f"timesteps must be > 0, got {timesteps}")

    scheduler = name.lower()
    if scheduler == "linear":
        betas = _linear_betas(timesteps, beta_start, beta_end, device)
    elif scheduler == "cosine":
        betas = _cosine_betas(timesteps, cosine_s, cosine_max_beta, device)
    else:
        raise ValueError(f"Unsupported scheduler: {name}")

    alphas = 1.0 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    return DiffusionSchedule(
        name=scheduler,
        timesteps=timesteps,
        betas=betas,
        alphas=alphas,
        alpha_hat=alpha_hat,
        sqrt_alpha_hat=torch.sqrt(alpha_hat),
        sqrt_one_minus_alpha_hat=torch.sqrt(1.0 - alpha_hat),
    )
