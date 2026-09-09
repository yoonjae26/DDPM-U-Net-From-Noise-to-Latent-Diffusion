from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.phases.reverse_diffusion_viewer import (  # noqa: E402
    load_reverse_viewer_artifacts,
    parse_step_labels,
    run_ddim_trajectory,
    run_reverse_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7: Reverse Diffusion in latent space")
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase6_h200_full" / "unet_last.pt",
        help="Path to trained latent UNet checkpoint",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase2_h200_full" / "vae_last.pt",
        help="Path to VAE checkpoint used for latent decoding",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "normalization_stats.yaml",
        help="Normalization stats from Phase 1",
    )
    parser.add_argument(
        "--latent-scaling",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase6_h200_full" / "latent_scaling.json",
        help="Path to latent scaling JSON from Phase 6",
    )
    parser.add_argument("--norm-mode", type=str, choices=["diffusion", "dataset"], default="diffusion")
    parser.add_argument("--scheduler", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampler", type=str, choices=["ddpm", "ddim"], default="ddpm",
                        help="Sampling algorithm: ddpm (stochastic 1000 steps) or ddim (deterministic, fewer steps)")
    parser.add_argument("--ddim-steps", type=int, default=50,
                        help="Number of denoising steps for DDIM (ignored for ddpm)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM stochasticity: 0=deterministic, 1=DDPM-equivalent noise")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-steps",
        type=str,
        default="1000,900,800,700,600,500,400,300,200,100,0",
        help="Comma-separated timestep labels to save",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase7",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    images_dir = output_dir / "image_space"
    latents_dir = output_dir / "latent_space"
    images_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

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

    step_map = parse_step_labels(args.save_steps, args.timesteps)
    for sample_idx in range(args.num_samples):
        if args.sampler == "ddim":
            snapshots = run_ddim_trajectory(
                artifacts=artifacts,
                scheduler_name=args.scheduler,
                timesteps=args.timesteps,
                ddim_steps=args.ddim_steps,
                eta=args.eta,
                seed=args.seed + sample_idx,
                step_label_map=step_map,
            )
        else:
            snapshots = run_reverse_trajectory(
                artifacts=artifacts,
                scheduler_name=args.scheduler,
                timesteps=args.timesteps,
                seed=args.seed + sample_idx,
                step_label_map=step_map,
            )

        for label, idx, latent_img, decoded_img in snapshots:
            decoded_img.save(images_dir / f"sample{sample_idx:02d}_t{label:04d}.png")
            latent_img.save(latents_dir / f"sample{sample_idx:02d}_t{label:04d}.png")

    report: dict[str, object] = {
        "scheduler": args.scheduler,
        "timesteps": args.timesteps,
        "sampler": args.sampler,
        "ddim_steps": args.ddim_steps if args.sampler == "ddim" else None,
        "eta": args.eta if args.sampler == "ddim" else None,
        "seed": args.seed,
        "num_samples": args.num_samples,
        "save_steps": [{"label": label, "index": idx} for label, idx, _, _ in snapshots],
        "unet_checkpoint": str(args.unet_checkpoint.resolve()),
        "vae_checkpoint": str(args.vae_checkpoint.resolve()),
        "latent_scaling": str(args.latent_scaling.resolve()) if args.latent_scaling else None,
        "latent_scaling_factor": artifacts.latent_scaling_factor,
        "channel_mults": list(artifacts.unet_channel_mults),
    }

    (output_dir / "phase7_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Phase 7 completed")
    print(f"Image-space outputs: {images_dir}")
    print(f"Latent-space outputs: {latents_dir}")
    print(f"Report: {output_dir / 'phase7_report.json'}")


if __name__ == "__main__":
    main()
