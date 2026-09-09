from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
import torch

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.diffusion import make_schedule, q_sample  # noqa: E402
from latent_diffusion_playground.phases.latent_space_viewer import load_viewer_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 4: Diffusion Core Engine in latent space")
    parser.add_argument(
        "--input-image",
        type=Path,
        required=True,
        help="Input image path",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase2_h200_full" / "vae_last.pt",
        help="VAE checkpoint path",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "normalization_stats.yaml",
        help="Normalization stats yaml path",
    )
    parser.add_argument(
        "--norm-mode",
        type=str,
        choices=["diffusion", "dataset"],
        default="diffusion",
        help="Normalization mode used by VAE",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["linear", "cosine", "both"],
        default="both",
        help="Noise scheduler variant",
    )
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--view-steps",
        type=str,
        default="0,100,300,700,1000",
        help="Comma separated timesteps to visualize",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase4",
    )
    return parser.parse_args()


def _center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def _normalize_image(image: Image.Image, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


def _denormalize_image(tensor: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> Image.Image:
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    x = torch.clamp(tensor * std_t + mean_t, 0.0, 1.0)
    array = (x[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(array)


def _parse_view_steps(raw: str, timesteps: int) -> list[tuple[int, int]]:
    labels: list[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        labels.append(int(token))
    if not labels:
        raise ValueError("view-steps cannot be empty")

    mapped: list[tuple[int, int]] = []
    for label in labels:
        if label < 0:
            raise ValueError(f"Invalid timestep label: {label}")
        index = min(label, timesteps - 1)
        mapped.append((label, index))
    return mapped


def _compose_grid(title: str, panels: list[tuple[str, Image.Image]]) -> Image.Image:
    if not panels:
        raise ValueError("panels cannot be empty")

    image_w, image_h = panels[0][1].size
    header_h = 48
    label_h = 28
    spacing = 14
    outer_pad = 18

    width = outer_pad * 2 + len(panels) * image_w + (len(panels) - 1) * spacing
    height = outer_pad * 2 + header_h + label_h + image_h
    canvas = Image.new("RGB", (width, height), color=(244, 247, 252))
    draw = ImageDraw.Draw(canvas)

    draw.text((outer_pad, outer_pad), title, fill=(25, 32, 45))

    y_label = outer_pad + header_h
    y_image = y_label + label_h
    for i, (label, panel) in enumerate(panels):
        x = outer_pad + i * (image_w + spacing)
        draw.rectangle([x - 2, y_image - 2, x + image_w + 2, y_image + image_h + 2], outline=(180, 188, 202), width=1)
        draw.text((x, y_label), label, fill=(48, 61, 82))
        canvas.paste(panel, (x, y_image))

    return canvas


def _scheduler_names(name: str) -> list[str]:
    if name == "both":
        return ["linear", "cosine"]
    return [name]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = load_viewer_artifacts(
        checkpoint_path=args.checkpoint.resolve(),
        normalization_stats_path=args.normalization_stats.resolve(),
        norm_mode=args.norm_mode,
        device=device,
    )

    source_image = Image.open(args.input_image).convert("RGB")
    preprocessed = _center_crop_resize(source_image, artifacts.image_size)
    x = _normalize_image(preprocessed, artifacts.normalization.mean, artifacts.normalization.std).to(device)

    with torch.no_grad():
        mu, _ = artifacts.model.encode(x)
        x0_latent = mu
        recon = artifacts.model.decode(x0_latent)

    recon_image = _denormalize_image(recon, artifacts.normalization.mean, artifacts.normalization.std)
    view_steps = _parse_view_steps(args.view_steps, args.timesteps)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    eps = torch.randn(x0_latent.shape, generator=generator, device=device, dtype=x0_latent.dtype)

    report: dict[str, object] = {
        "timesteps": args.timesteps,
        "input_image": str(args.input_image.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "norm_mode": args.norm_mode,
        "latent_shape": list(x0_latent.shape),
        "view_steps": [{"label": label, "index": index} for label, index in view_steps],
        "schedulers": {},
    }

    for scheduler_name in _scheduler_names(args.scheduler):
        schedule = make_schedule(
            name=scheduler_name,
            timesteps=args.timesteps,
            device=device,
        )

        panels: list[tuple[str, Image.Image]] = [("original", preprocessed), ("latent->decode", recon_image)]
        scheduler_stats = {
            "beta_min": float(schedule.betas.min().item()),
            "beta_max": float(schedule.betas.max().item()),
            "alpha_hat_final": float(schedule.alpha_hat[-1].item()),
            "samples": [],
        }

        for label, index in view_steps:
            t = torch.tensor([index], dtype=torch.long, device=device)
            with torch.no_grad():
                z_t = q_sample(x0_latent, t, schedule, noise=eps)
                decoded = artifacts.model.decode(z_t)

            decoded_image = _denormalize_image(decoded, artifacts.normalization.mean, artifacts.normalization.std)
            panels.append((f"t={label}", decoded_image))

            scheduler_stats["samples"].append(
                {
                    "label": label,
                    "index": index,
                    "latent_mean": float(z_t.mean().item()),
                    "latent_std": float(z_t.std().item()),
                }
            )

        title = f"Phase 4 Forward Diffusion in Latent Space ({scheduler_name})"
        grid = _compose_grid(title, panels)
        grid_path = output_dir / f"forward_diffusion_{scheduler_name}.png"
        grid.save(grid_path)

        scheduler_stats["grid_path"] = str(grid_path)
        report["schedulers"][scheduler_name] = scheduler_stats

    report_path = output_dir / "phase4_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Phase 4 completed")
    print(f"Output directory: {output_dir}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
