from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint_sequential


class VAE(nn.Module):
    """Spatial-latent convolutional VAE (Stable Diffusion style).

    Input:  B × 3 × H × W  (H, W divisible by 8)
    Latent: B × latent_channels × (H//8) × (W//8)

    For 128×128 input the latent is latent_channels × 16 × 16.
    Encoder indices: 0=Conv 1=Norm 2=SiLU 3=Conv 4=Norm 5=SiLU 6=Conv 7=Norm 8=SiLU 9=Conv1x1(mu∥logvar)
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        image_size: int = 128,
        base_channels: int = 128,
        use_batch_norm: bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if image_size % 8 != 0:
            raise ValueError(f"image_size must be divisible by 8, got {image_size}")

        self.latent_channels = latent_channels
        self.image_size = image_size
        self.feature_map_size = image_size // 8
        self.activation_checkpointing = activation_checkpointing

        bc = base_channels

        def norm(channels: int) -> nn.Module:
            if use_batch_norm:
                return nn.BatchNorm2d(channels)
            return nn.GroupNorm(min(channels // 8, 32), channels)

        # 3 stride-2 downsamples → H/8, then 1×1 conv outputs mu ∥ logvar
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, bc, kernel_size=4, stride=2, padding=1),         # H/2
            norm(bc),
            nn.SiLU(),
            nn.Conv2d(bc, bc * 2, kernel_size=4, stride=2, padding=1),              # H/4
            norm(bc * 2),
            nn.SiLU(),
            nn.Conv2d(bc * 2, bc * 4, kernel_size=4, stride=2, padding=1),          # H/8
            norm(bc * 4),
            nn.SiLU(),
            nn.Conv2d(bc * 4, latent_channels * 2, kernel_size=1),                  # mu ∥ logvar
        )

        # channel expansion + 3 stride-2 upsamples → H
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, bc * 4, kernel_size=1),
            norm(bc * 4),
            nn.SiLU(),
            nn.ConvTranspose2d(bc * 4, bc * 2, kernel_size=4, stride=2, padding=1),  # H/4
            norm(bc * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(bc * 2, bc, kernel_size=4, stride=2, padding=1),       # H/2
            norm(bc),
            nn.SiLU(),
            nn.ConvTranspose2d(bc, in_channels, kernel_size=4, stride=2, padding=1),  # H
            nn.Tanh(),
        )

    def _run_sequential(self, module: nn.Sequential, x: torch.Tensor, segments: int) -> torch.Tensor:
        if self.activation_checkpointing and self.training:
            return checkpoint_sequential(module, segments=segments, input=x, use_reentrant=False)
        return module(x)

    def encode_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic latent (mu). Used by diffusion training and phase-7 inference."""
        out = self._run_sequential(self.encoder, x, segments=4)
        return out[:, :self.latent_channels]

    def decode_feature_map(self, z: torch.Tensor) -> torch.Tensor:
        return self._run_sequential(self.decoder, z, segments=4)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._run_sequential(self.encoder, x, segments=4)
        mu = out[:, :self.latent_channels]
        logvar = out[:, self.latent_channels:]
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decode_feature_map(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z


def vae_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    recon_loss = nn.functional.mse_loss(recon, target, reduction="mean")
    # Mean KL over all elements (batch × channels × height × width).
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + kl_weight * kl_loss
    return total_loss, recon_loss, kl_loss
