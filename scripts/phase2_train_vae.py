from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.data.torch_dataset import (  # noqa: E402
    CelebAManifestDataset,
    NormalizationConfig,
)
from latent_diffusion_playground.models.vae import VAE  # noqa: E402
from latent_diffusion_playground.training.vae_trainer import (  # noqa: E402
    VAETrainConfig,
    train_vae,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: Train VAE on preprocessed CelebA")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "manifest.parquet",
        help="Manifest generated in Phase 1",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "normalization_stats.yaml",
        help="Normalization stats from Phase 1",
    )
    parser.add_argument(
        "--norm-mode",
        type=str,
        choices=["diffusion", "dataset"],
        default="diffusion",
        help="Use diffusion default normalization or dataset mean/std",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase2",
    )
    return parser.parse_args()


def load_normalization(config_path: Path, mode: str) -> NormalizationConfig:
    stats = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if mode == "dataset":
        mean = stats["mean"]
        std = stats["std"]
    else:
        mean = stats["diffusion_default_mean"]
        std = stats["diffusion_default_std"]
    return NormalizationConfig(mean=tuple(mean), std=tuple(std))


def denormalize(tensor: torch.Tensor, norm: NormalizationConfig) -> torch.Tensor:
    mean = torch.tensor(norm.mean, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    x = tensor * std + mean
    return torch.clamp(x, 0.0, 1.0)


def save_reconstruction_grid(
    model: VAE,
    loader: DataLoader,
    norm: NormalizationConfig,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    batch = next(iter(loader)).to(device)
    with torch.no_grad():
        recon, _, _, _ = model(batch)

    inputs = denormalize(batch[:8], norm).cpu()
    outputs = denormalize(recon[:8], norm).cpu()

    rows = []
    for i in range(inputs.shape[0]):
        inp = (inputs[i].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        out = (outputs[i].permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        rows.append(Image.fromarray(inp))
        rows.append(Image.fromarray(out))

    width, height = rows[0].size
    canvas = Image.new("RGB", (2 * width, len(rows) // 2 * height))
    for i in range(0, len(rows), 2):
        r = i // 2
        canvas.paste(rows[i], (0, r * height))
        canvas.paste(rows[i + 1], (width, r * height))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def maybe_make_loader(
    manifest: Path,
    split: str,
    normalization: NormalizationConfig,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    shuffle: bool,
) -> DataLoader | None:
    try:
        dataset = CelebAManifestDataset(
            manifest_path=manifest,
            split=split,
            normalization=normalization,
            max_samples=max_samples,
        )
    except ValueError:
        return None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    normalization = load_normalization(args.normalization_stats.resolve(), args.norm_mode)

    train_loader = maybe_make_loader(
        manifest=args.manifest.resolve(),
        split="train",
        normalization=normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_train_samples,
        shuffle=True,
    )
    if train_loader is None:
        raise RuntimeError("Train split not found or empty in manifest.")

    val_loader = maybe_make_loader(
        manifest=args.manifest.resolve(),
        split="val",
        normalization=normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_val_samples,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader) if val_loader is not None else 0}")

    model = VAE(in_channels=3, latent_dim=args.latent_dim)
    config = VAETrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
    )

    train_vae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        output_dir=output_dir,
    )

    sample_loader = val_loader if val_loader is not None else train_loader
    save_reconstruction_grid(
        model=model.to(device),
        loader=sample_loader,
        norm=normalization,
        device=device,
        output_path=output_dir / "reconstruction_preview.png",
    )

    print("Phase 2 completed")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()
