from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.models.unet import LatentUNet  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5: Latent U-Net Development")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--channel-mults",
        type=str,
        default="1,2,4",
        help="Comma-separated channel multipliers for U-Net stages",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase5",
    )
    return parser.parse_args()


def _parse_mults(raw: str) -> tuple[int, ...]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("channel-mults cannot be empty")
    if any(v <= 0 for v in values):
        raise ValueError(f"channel-mults must be positive, got {values}")
    return tuple(values)


def _count_params(model: torch.nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    channel_mults = _parse_mults(args.channel_mults)
    model = LatentUNet(
        in_channels=args.latent_channels,
        base_channels=args.base_channels,
        channel_mults=channel_mults,
    ).to(device)

    noisy_latent = torch.randn(
        args.batch_size,
        args.latent_channels,
        args.latent_size,
        args.latent_size,
        device=device,
    )
    t = torch.randint(0, args.timesteps, (args.batch_size,), device=device)

    with torch.no_grad():
        predicted_noise = model(noisy_latent, t)

    report = {
        "device": str(device),
        "batch_size": args.batch_size,
        "latent_channels": args.latent_channels,
        "latent_size": args.latent_size,
        "base_channels": args.base_channels,
        "channel_mults": list(channel_mults),
        "params": _count_params(model),
        "timesteps_range": [0, args.timesteps - 1],
        "input_shape": list(noisy_latent.shape),
        "output_shape": list(predicted_noise.shape),
        "output_mean": float(predicted_noise.mean().item()),
        "output_std": float(predicted_noise.std().item()),
    }

    report_path = output_dir / "phase5_model_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Phase 5 completed")
    print(f"Output shape: {list(predicted_noise.shape)}")
    print(f"Params: {report['params']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
