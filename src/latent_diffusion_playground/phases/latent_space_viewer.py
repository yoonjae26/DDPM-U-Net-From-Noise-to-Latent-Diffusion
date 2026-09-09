from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import torch
import yaml
from latent_diffusion_playground.models.vae import VAE
from latent_diffusion_playground.data.torch_dataset import NormalizationConfig


@dataclass
class ViewerArtifacts:
    model: VAE
    normalization: NormalizationConfig
    image_size: int
    latent_dim: int  # latent_channels for new-style VAE
    device: torch.device


def _infer_vae_config(state_dict: dict) -> dict:
    """Infer VAE constructor kwargs from a state dict.

    Supports both architectures:
    - New-style (spatial latent): no fc_mu, encoder.9 is 1×1 conv to mu∥logvar
    - Old-style (flat latent):    has fc_mu.weight
    """
    base_channels = int(state_dict["encoder.0.weight"].shape[0])

    if "fc_mu.weight" not in state_dict:
        # New-style spatial-latent VAE (3 downsamples, no fc layers)
        # encoder.9: Conv2d(bc*4, latent_channels*2, 1) → shape (lc*2, bc*4, 1, 1)
        latent_channels = int(state_dict["encoder.9.weight"].shape[0]) // 2
        use_batch_norm = any("running_mean" in k for k in state_dict)
        return {
            "latent_channels": latent_channels,
            "image_size": 128,  # fully convolutional; 128 is the project default
            "base_channels": base_channels,
            "use_batch_norm": use_batch_norm,
        }

    # Old-style flat-latent VAE (4 downsamples + fc_mu/fc_logvar)
    import math
    latent_dim = int(state_dict["fc_mu.weight"].shape[0])
    flatten_dim = int(state_dict["fc_mu.weight"].shape[1])
    last_channels = base_channels * 8
    feature_map_size = int(math.sqrt(flatten_dim / last_channels))
    image_size = feature_map_size * 16
    use_batch_norm = "encoder.1.running_mean" in state_dict
    return {
        "latent_dim": latent_dim,
        "image_size": image_size,
        "base_channels": base_channels,
        "use_batch_norm": use_batch_norm,
    }


def load_viewer_artifacts(
    checkpoint_path: Path,
    normalization_stats_path: Path,
    norm_mode: str,
    device: torch.device,
) -> ViewerArtifacts:
    """Load VAE model and associated metadata for visualization or downstream tasks."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    cfg = _infer_vae_config(state_dict)

    model = VAE(in_channels=3, **cfg)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    stats = yaml.safe_load(normalization_stats_path.read_text(encoding="utf-8"))
    if norm_mode == "dataset":
        mean, std = stats["mean"], stats["std"]
    else:
        mean, std = stats["diffusion_default_mean"], stats["diffusion_default_std"]

    # latent_dim stores latent_channels for new-style, latent_dim for old-style
    latent_dim = cfg.get("latent_channels", cfg.get("latent_dim", 0))

    return ViewerArtifacts(
        model=model,
        normalization=NormalizationConfig(mean=tuple(mean), std=tuple(std)),
        image_size=cfg["image_size"],
        latent_dim=latent_dim,
        device=device,
    )
