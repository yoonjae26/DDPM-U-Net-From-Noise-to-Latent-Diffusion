from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.diffusion import make_schedule, p_sample_step  # noqa: E402
from latent_diffusion_playground.phases.reverse_diffusion_viewer import (  # noqa: E402
    decoded_to_image,
    load_reverse_viewer_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Phase 11: Scheduler Stability Lab -- shows why naive DDPM ancestral "
            "sampling can explode near t=T, and how clipping the predicted x0 "
            "(dynamic thresholding) fixes it. Compares linear vs cosine schedules "
            "with clip_x0 on/off."
        )
    )
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase6_spatial_v3" / "unet_last.pt",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase2_spatial_v1" / "vae_best.pt",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "normalization_stats.yaml",
    )
    parser.add_argument(
        "--latent-scaling",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase6_spatial_v3" / "latent_scaling.json",
    )
    parser.add_argument("--norm-mode", type=str, choices=["diffusion", "dataset"], default="diffusion")
    parser.add_argument("--schedulers", type=str, default="linear,cosine")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--clip-range", type=float, default=4.0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase11_scheduler_stability_lab",
    )
    return parser.parse_args()


@torch.no_grad()
def run_instrumented_trajectory(
    unet: torch.nn.Module,
    schedule,
    shape: tuple[int, ...],
    device: torch.device,
    seed: int,
    clip_x0: bool,
    clip_range: float,
    log_every: int,
) -> tuple[torch.Tensor, list[dict]]:
    """Run full ancestral DDPM sampling, recording magnitude stats along the way."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    img = torch.randn(shape, device=device)
    trace: list[dict] = []
    exploded = False

    for step_number, i in enumerate(reversed(range(schedule.timesteps))):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        epsilon = unet(img, t)
        img = p_sample_step(img, t, epsilon, schedule, clip_x0=clip_x0, clip_range=clip_range)

        img_std = float(img.std().item())
        if step_number % log_every == 0 or i == 0:
            trace.append({
                "step_number": step_number,
                "t": int(i),
                "mean": float(img.mean().item()),
                "std": img_std,
                "max_abs": float(img.abs().max().item()),
            })

        if not exploded and (img_std > 1e4 or not torch.isfinite(img).all()):
            exploded = True
            trace.append({"step_number": step_number, "t": int(i), "note": "diverged (std>1e4 or non-finite)"})

    return img, trace


def _save_grid(images: list[Image.Image], out_path: Path) -> None:
    w, h = images[0].size
    canvas = Image.new("RGB", (len(images) * w, h), color=(248, 248, 248))
    for idx, img in enumerate(images):
        canvas.paste(img, (idx * w, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    artifacts = load_reverse_viewer_artifacts(
        unet_checkpoint_path=args.unet_checkpoint.resolve(),
        vae_checkpoint_path=args.vae_checkpoint.resolve(),
        normalization_stats_path=args.normalization_stats.resolve(),
        norm_mode=args.norm_mode,
        device=device,
        latent_scaling_path=args.latent_scaling.resolve() if args.latent_scaling is not None else None,
    )
    unet = artifacts.unet
    vae = artifacts.vae_artifacts.model
    shape = (args.num_samples, artifacts.unet_in_channels, 16, 16)

    scheduler_names = [s.strip() for s in args.schedulers.split(",") if s.strip()]
    conditions = [(name, clip) for name in scheduler_names for clip in (False, True)]

    report: dict[str, object] = {"conditions": {}}
    plt.figure(figsize=(8, 5))

    for scheduler_name, clip_x0 in conditions:
        schedule = make_schedule(scheduler_name, args.timesteps, device=device)
        beta_tail = schedule.betas[-5:].tolist()

        final_latent, trace = run_instrumented_trajectory(
            unet=unet,
            schedule=schedule,
            shape=shape,
            device=device,
            seed=args.seed,
            clip_x0=clip_x0,
            clip_range=args.clip_range,
            log_every=args.log_every,
        )

        label = f"{scheduler_name} / clip_x0={'on' if clip_x0 else 'off'}"
        key = f"{scheduler_name}_{'clip' if clip_x0 else 'noclip'}"

        stds = [p["std"] for p in trace if "std" in p]
        steps = [p["step_number"] for p in trace if "std" in p]
        plt.plot(steps, stds, label=label)

        decoded = vae.decode_feature_map(final_latent / artifacts.latent_scaling_factor)
        images = [
            decoded_to_image(decoded[i:i+1], artifacts.vae_artifacts.normalization.mean, artifacts.vae_artifacts.normalization.std)
            for i in range(args.num_samples)
        ]
        _save_grid(images, output_dir / f"samples_{key}.png")

        report["conditions"][key] = {
            "scheduler": scheduler_name,
            "clip_x0": clip_x0,
            "clip_range": args.clip_range,
            "beta_tail_last5": beta_tail,
            "final_std": stds[-1] if stds else None,
            "final_max_abs": trace[-1].get("max_abs") if trace else None,
            "diverged": any("note" in p for p in trace),
        }

    plt.xlabel("Denoising step (0 = start at t=T, higher = closer to t=0)")
    plt.ylabel("Latent std (log scale)")
    plt.yscale("log")
    plt.title("Why naive DDPM sampling explodes near t=T (and how x0-clipping fixes it)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot_path = output_dir / "latent_std_vs_step.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    (output_dir / "phase11_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Phase 11 completed")
    print(f"Plot: {plot_path}")
    print(f"Report: {output_dir / 'phase11_report.json'}")
    print(f"Sample grids: {output_dir}/samples_*.png")


if __name__ == "__main__":
    main()
