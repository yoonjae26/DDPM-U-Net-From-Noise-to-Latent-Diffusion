from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from latent_diffusion_playground.diffusion import ddim_sample_step, make_schedule, p_sample_step
from latent_diffusion_playground.models.unet import LatentUNet
from latent_diffusion_playground.phases.latent_space_viewer import ViewerArtifacts, load_viewer_artifacts


@dataclass(frozen=True)
class ReverseViewerArtifacts:
    vae_artifacts: ViewerArtifacts
    unet: LatentUNet
    unet_in_channels: int
    unet_base_channels: int
    unet_channel_mults: tuple[int, ...]
    latent_scaling_factor: float
    device: torch.device


def infer_unet_config(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, tuple[int, ...], bool, int]:
    in_channels = int(state_dict["input_conv.weight"].shape[1])
    base_channels = int(state_dict["input_conv.weight"].shape[0])

    mults: list[int] = []
    i = 0
    while True:
        key = f"down_blocks.{i}.resblock.conv1.weight"
        if key not in state_dict:
            break
        out_channels = int(state_dict[key].shape[0])
        mults.append(max(out_channels // base_channels, 1))
        i += 1

    if not mults:
        raise ValueError("Cannot infer channel multipliers from UNet checkpoint")

    # New-format checkpoints have bottleneck.res1.* (Bottleneck class with attention).
    # Old-format checkpoints have bottleneck.conv1.* (bare ResidualTimeBlock, no attention).
    has_attention = "bottleneck.res1.conv1.weight" in state_dict

    num_heads = 8  # default; infer from in_proj_weight if present
    if has_attention:
        for key, tensor in state_dict.items():
            if key.endswith("attn.attn.in_proj_weight"):
                embed_dim = tensor.shape[1]
                # Find matching out_channels at that block level to compute num_heads
                # embed_dim == channels at that block; num_heads must divide embed_dim
                for h in [8, 4, 2, 1]:
                    if embed_dim % h == 0:
                        num_heads = h
                        break
                break

    return in_channels, base_channels, tuple(mults), has_attention, num_heads


def load_reverse_viewer_artifacts(
    unet_checkpoint_path: Path,
    vae_checkpoint_path: Path,
    normalization_stats_path: Path,
    norm_mode: str,
    device: torch.device,
    latent_scaling_path: Path | None = None,
    latent_scaling_factor: float | None = None,
) -> ReverseViewerArtifacts:
    vae_artifacts = load_viewer_artifacts(
        checkpoint_path=vae_checkpoint_path,
        normalization_stats_path=normalization_stats_path,
        norm_mode=norm_mode,
        device=device,
    )

    payload = torch.load(unet_checkpoint_path, map_location=device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported UNet checkpoint format")

    in_channels, base_channels, channel_mults, has_attention, num_heads = infer_unet_config(state_dict)
    unet = LatentUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_heads=num_heads,
        use_attention=has_attention,
    ).to(device)
    has_bn_keys = any("running_mean" in k for k in state_dict)
    if has_bn_keys:
        state_dict = {k: v for k, v in state_dict.items()
                      if not any(x in k for x in ("running_mean", "running_var", "num_batches_tracked"))}
    unet.load_state_dict(state_dict, strict=not has_bn_keys)
    unet.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, vae_artifacts.image_size, vae_artifacts.image_size, device=device)
        latent_template = vae_artifacts.model.encode_feature_map(dummy)
    if int(latent_template.shape[1]) != in_channels:
        raise ValueError(
            f"UNet in_channels={in_channels} does not match VAE latent channels={int(latent_template.shape[1])}"
        )

    scaling = 1.0
    if latent_scaling_factor is not None:
        scaling = float(latent_scaling_factor)
    elif latent_scaling_path is not None and latent_scaling_path.exists():
        import json

        payload = json.loads(latent_scaling_path.read_text(encoding="utf-8"))
        scaling = float(payload.get("latent_scaling_factor", 1.0))

    return ReverseViewerArtifacts(
        vae_artifacts=vae_artifacts,
        unet=unet,
        unet_in_channels=in_channels,
        unet_base_channels=base_channels,
        unet_channel_mults=channel_mults,
        latent_scaling_factor=scaling,
        device=device,
    )


def parse_step_labels(raw: str, timesteps: int) -> dict[int, int]:
    labels = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not labels:
        raise ValueError("save step labels cannot be empty")

    mapping: dict[int, int] = {}
    for label in labels:
        if label < 0:
            raise ValueError(f"step label must be >=0, got {label}")
        index = min(label, timesteps - 1)
        mapping[index] = label
    mapping[0] = 0
    return mapping


def latent_to_heatmap(latent: torch.Tensor, size: int = 256) -> Image.Image:
    heat = latent[0].detach().float().cpu().abs().mean(dim=0).numpy()
    heat = heat - heat.min()
    denom = float(heat.max()) if float(heat.max()) > 0 else 1.0
    heat = heat / denom
    rgb = np.stack([heat, np.sqrt(heat), 1.0 - heat], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8)).resize((size, size), Image.Resampling.NEAREST)


def decoded_to_image(decoded: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> Image.Image:
    mean_t = torch.tensor(mean, dtype=decoded.dtype, device=decoded.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=decoded.dtype, device=decoded.device).view(1, 3, 1, 1)
    y = torch.clamp(decoded * std_t + mean_t, 0.0, 1.0)
    arr = (y[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def run_reverse_trajectory(
    artifacts: ReverseViewerArtifacts,
    scheduler_name: str,
    timesteps: int,
    seed: int,
    step_label_map: dict[int, int],
) -> list[tuple[int, int, Image.Image, Image.Image]]:
    device = artifacts.device
    vae = artifacts.vae_artifacts.model
    unet = artifacts.unet
    schedule = make_schedule(scheduler_name, timesteps, device=device)

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        dummy = torch.zeros(1, 3, artifacts.vae_artifacts.image_size, artifacts.vae_artifacts.image_size, device=device)
        latent_template = vae.encode_feature_map(dummy)
        x_t = torch.randn_like(latent_template)  # in scaled latent space

        snapshots: list[tuple[int, int, Image.Image, Image.Image]] = []
        for idx in range(timesteps - 1, -1, -1):
            t = torch.tensor([idx], dtype=torch.long, device=device)
            pred_noise = unet(x_t, t)
            x_t = p_sample_step(x_t, t, pred_noise, schedule)

            if idx in step_label_map:
                label = step_label_map[idx]
                x_unscaled = x_t / max(artifacts.latent_scaling_factor, 1e-8)
                decoded = vae.decode_feature_map(x_unscaled)
                latent_img = latent_to_heatmap(x_t)
                decoded_img = decoded_to_image(
                    decoded,
                    artifacts.vae_artifacts.normalization.mean,
                    artifacts.vae_artifacts.normalization.std,
                )
                snapshots.append((label, idx, latent_img, decoded_img))

    snapshots.sort(key=lambda item: item[0], reverse=True)
    return snapshots


def run_ddim_trajectory(
    artifacts: ReverseViewerArtifacts,
    scheduler_name: str,
    timesteps: int,
    ddim_steps: int,
    eta: float,
    seed: int,
    step_label_map: dict[int, int],
) -> list[tuple[int, int, Image.Image, Image.Image]]:
    device = artifacts.device
    vae = artifacts.vae_artifacts.model
    unet = artifacts.unet
    schedule = make_schedule(scheduler_name, timesteps, device=device)

    # Build sub-sequence [t_{S-1}, ..., t_1] then step to t=0
    step_size = max(timesteps // ddim_steps, 1)
    ddim_seq: list[int] = list(range(timesteps - 1, 0, -step_size))
    ddim_seq_prev: list[int] = ddim_seq[1:] + [0]

    # Map each requested save label to nearest step in ddim_seq
    save_at: dict[int, int] = {}  # t_idx → label (save after stepping FROM t_idx)
    for t_idx, label in step_label_map.items():
        if t_idx == 0:
            continue  # final state is always saved after the loop
        nearest = min(ddim_seq, key=lambda x: abs(x - t_idx))
        save_at[nearest] = label

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    with torch.no_grad():
        dummy = torch.zeros(
            1, 3,
            artifacts.vae_artifacts.image_size,
            artifacts.vae_artifacts.image_size,
            device=device,
        )
        x_t = torch.randn_like(vae.encode_feature_map(dummy))

        snapshots: list[tuple[int, int, Image.Image, Image.Image]] = []

        def _snap(label: int, idx: int) -> None:
            x_unscaled = x_t / max(artifacts.latent_scaling_factor, 1e-8)
            decoded = vae.decode_feature_map(x_unscaled)
            snapshots.append((
                label, idx,
                latent_to_heatmap(x_t),
                decoded_to_image(decoded, artifacts.vae_artifacts.normalization.mean, artifacts.vae_artifacts.normalization.std),
            ))

        for t_idx, t_prev_idx in zip(ddim_seq, ddim_seq_prev):
            t      = torch.tensor([t_idx],      dtype=torch.long, device=device)
            t_prev = torch.tensor([t_prev_idx], dtype=torch.long, device=device)
            pred_noise = unet(x_t, t)
            x_t = ddim_sample_step(x_t, t, t_prev, pred_noise, schedule, eta=eta)
            if t_idx in save_at:
                _snap(save_at[t_idx], t_prev_idx)

        # Final state (t=0) — always saved
        _snap(0, 0)

    snapshots.sort(key=lambda item: item[0], reverse=True)
    return snapshots


def export_gif(decoded_frames: list[Image.Image], output_path: Path, frame_duration_ms: int = 140) -> Path:
    if not decoded_frames:
        raise ValueError("decoded_frames cannot be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    decoded_frames[0].save(
        output_path,
        format="GIF",
        append_images=decoded_frames[1:],
        save_all=True,
        duration=frame_duration_ms,
        loop=0,
    )
    return output_path
