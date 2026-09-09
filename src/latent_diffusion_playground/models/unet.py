from __future__ import annotations

import math

import torch
from torch import nn


def _group_norm(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    groups = min(max_groups, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    if dim <= 0:
        raise ValueError(f"dim must be > 0, got {dim}")

    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
    exponent = exponent / max(half_dim - 1, 1)
    angles = timesteps.float().unsqueeze(1) * torch.exp(exponent).unsqueeze(0)

    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=1)
    return embedding


class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(embedding_dim * 4, embedding_dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        embedding = sinusoidal_time_embedding(timesteps, self.embedding_dim)
        return self.mlp(embedding)


class SelfAttention2D(nn.Module):
    """Pre-norm self-attention over spatial positions (H*W tokens)."""

    def __init__(self, channels: int, num_heads: int = 8) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.norm = _group_norm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x).view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.permute(0, 2, 1).view(B, C, H, W)  # residual


class ResidualTimeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = _group_norm(out_channels)
        self.act1 = nn.SiLU()

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = _group_norm(out_channels)
        self.act2 = nn.SiLU()

        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        residual = self.skip_proj(x)

        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act1(h)

        t = self.time_proj(time_embedding).unsqueeze(-1).unsqueeze(-1)
        h = h + t

        h = self.conv2(h)
        h = self.bn2(h)

        return self.act2(h + residual)


class DownBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        downsample: bool,
        use_attention: bool = False,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.resblock = ResidualTimeBlock(in_channels, out_channels, time_dim)
        self.attn = SelfAttention2D(out_channels, num_heads) if use_attention else None
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1) if downsample else None

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.resblock(x, time_embedding)
        if self.attn is not None:
            h = self.attn(h)
        skip = h  # skip captured after attention
        if self.downsample is not None:
            h = self.downsample(h)
        return h, skip


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_dim: int,
        upsample: bool,
        use_attention: bool = False,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.upsample = (
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1) if upsample else None
        )
        self.resblock = ResidualTimeBlock(in_channels + skip_channels, out_channels, time_dim)
        self.attn = SelfAttention2D(out_channels, num_heads) if use_attention else None

    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = x
        if self.upsample is not None:
            h = self.upsample(h)
        h = torch.cat([h, skip], dim=1)
        h = self.resblock(h, time_embedding)
        if self.attn is not None:
            h = self.attn(h)
        return h


class Bottleneck(nn.Module):
    """ResBlock → Attention → ResBlock — standard LDM bottleneck pattern."""

    def __init__(self, channels: int, time_dim: int, num_heads: int = 8) -> None:
        super().__init__()
        self.res1 = ResidualTimeBlock(channels, channels, time_dim)
        self.attn = SelfAttention2D(channels, num_heads)
        self.res2 = ResidualTimeBlock(channels, channels, time_dim)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        h = self.res1(x, time_embedding)
        h = self.attn(h)
        return self.res2(h, time_embedding)


class LatentUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 128,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        num_heads: int = 8,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        if not channel_mults:
            raise ValueError("channel_mults cannot be empty")

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.channel_mults = channel_mults
        self.use_attention = use_attention

        time_embed_dim = base_channels
        self.time_embedding = TimeEmbedding(time_embed_dim)
        time_dim = time_embed_dim * 4

        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        n_levels = len(channel_mults)
        down_blocks: list[nn.Module] = []
        skip_channels: list[int] = []

        current_channels = base_channels
        for i, mult in enumerate(channel_mults):
            out_channels = base_channels * mult
            downsample = i < n_levels - 1
            # Attention at the two lowest spatial resolutions (last two down levels)
            use_attn = use_attention and (i >= n_levels - 2)
            down_blocks.append(DownBlock(current_channels, out_channels, time_dim, downsample, use_attn, num_heads))
            skip_channels.append(out_channels)
            current_channels = out_channels

        self.down_blocks = nn.ModuleList(down_blocks)

        if use_attention:
            self.bottleneck = Bottleneck(current_channels, time_dim, num_heads)
        else:
            # Legacy format: single ResidualTimeBlock — matches old checkpoint key names (bottleneck.conv1/conv2/...)
            self.bottleneck = ResidualTimeBlock(current_channels, current_channels, time_dim)

        up_blocks: list[nn.Module] = []
        reversed_skip_channels = list(reversed(skip_channels))
        for i, skip_ch in enumerate(reversed_skip_channels):
            upsample = i > 0
            out_channels = base_channels * channel_mults[max(n_levels - i - 2, 0)]
            # Attention mirrors down: first two up levels (lowest resolutions)
            use_attn = use_attention and (i < 2)
            up_blocks.append(UpBlock(current_channels, skip_ch, out_channels, time_dim, upsample, use_attn, num_heads))
            current_channels = out_channels

        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_head = nn.Sequential(
            nn.Conv2d(current_channels, current_channels, kernel_size=3, padding=1),
            _group_norm(current_channels),
            nn.SiLU(),
            nn.Conv2d(current_channels, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, noisy_latent: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if noisy_latent.ndim != 4:
            raise ValueError(f"Expected noisy_latent as [B,C,H,W], got shape {list(noisy_latent.shape)}")
        if timesteps.ndim != 1:
            raise ValueError(f"Expected timesteps as [B], got shape {list(timesteps.shape)}")
        if noisy_latent.shape[0] != timesteps.shape[0]:
            raise ValueError("Batch size mismatch between noisy_latent and timesteps")

        h = self.input_conv(noisy_latent)
        t_emb = self.time_embedding(timesteps)

        skips: list[torch.Tensor] = []
        for block in self.down_blocks:
            h, skip = block(h, t_emb)
            skips.append(skip)

        h = self.bottleneck(h, t_emb)

        for block in self.up_blocks:
            skip = skips.pop()
            h = block(h, skip, t_emb)

        return self.output_head(h)
