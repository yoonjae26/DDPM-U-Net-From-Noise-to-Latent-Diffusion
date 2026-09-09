from __future__ import annotations

import torch
from torch import nn


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class PatchDiscriminator(nn.Module):
    """PatchGAN discriminator (Isola et al. 2017 / used in VQGAN, taming-transformers).

    Classifies overlapping NxN patches as real/fake instead of the whole image
    at once -- this is what actually pushes a VAE decoder toward sharp,
    high-frequency detail (skin texture, eye/mouth edges) that plain MSE and
    even LPIPS alone tend to smooth over.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = base_channels
        for _ in range(1, n_layers):
            next_channels = min(channels * 2, 512)
            layers += [
                nn.Conv2d(channels, next_channels, kernel_size=4, stride=2, padding=1),
                _group_norm(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            channels = next_channels

        next_channels = min(channels * 2, 512)
        layers += [
            nn.Conv2d(channels, next_channels, kernel_size=4, stride=1, padding=1),
            _group_norm(next_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        channels = next_channels
        layers += [nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def discriminator_hinge_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    loss_real = torch.relu(1.0 - real_logits).mean()
    loss_fake = torch.relu(1.0 + fake_logits).mean()
    return 0.5 * (loss_real + loss_fake)


def generator_hinge_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return -fake_logits.mean()
