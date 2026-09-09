from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
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
    parser.add_argument("--latent-channels", type=int, default=4)
    parser.add_argument("--kl-weight", type=float, default=1e-3) #5e-4 or 1e-3
    parser.add_argument(
        "--perceptual-weight",
        type=float,
        default=0.0,
        help="LPIPS perceptual loss weight added on top of MSE+KL. 0 (default) disables it, "
             "matching original behavior. Requires the 'lpips' package.",
    )
    parser.add_argument(
        "--adversarial-weight",
        type=float,
        default=0.0,
        help="PatchGAN adversarial loss weight added on top of MSE+KL(+LPIPS). 0 (default) "
             "disables it. Best enabled as a fine-tuning stage (--resume-checkpoint) on a VAE "
             "that already reconstructs well -- training a fresh decoder against a fresh "
             "discriminator from scratch is far less stable.",
    )
    parser.add_argument("--disc-lr", type=float, default=None, help="Discriminator LR (defaults to --lr).")
    parser.add_argument("--disc-base-channels", type=int, default=64)
    parser.add_argument("--disc-n-layers", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision during training and validation.",
    )
    parser.add_argument(
        "--amp-dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="Autocast dtype to use when AMP is enabled.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the model with torch.compile for faster steady-state training.",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="max-autotune",
        help="torch.compile mode, for example default, reduce-overhead, or max-autotune.",
    )
    parser.add_argument(
        "--activation-checkpointing",
        action="store_true",
        help="Enable activation checkpointing in encoder and decoder to reduce memory usage.",
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        help="Use channels_last memory format for better GPU throughput on conv-heavy models.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Save full training-state checkpoint every N epochs.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Path to a training-state checkpoint file for resume.",
    )
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


def infer_image_size(manifest_path: Path) -> int:
    if manifest_path.suffix == ".parquet":
        manifest = pd.read_parquet(manifest_path, columns=["processed_width", "processed_height"])
    else:
        manifest = pd.read_csv(manifest_path, usecols=["processed_width", "processed_height"])

    if manifest.empty:
        raise ValueError("Manifest is empty; cannot infer image size.")

    width = int(manifest.iloc[0]["processed_width"])
    height = int(manifest.iloc[0]["processed_height"])
    if width != height:
        raise ValueError(f"Expected square processed images, got {width}x{height}.")
    return width


def save_reconstruction_grid(
    model: VAE,
    loader: DataLoader,
    norm: NormalizationConfig,
    device: torch.device,
    output_path: Path,
) -> None:
    model.eval()
    batch = next(iter(loader)).to(device, non_blocking=True)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            recon, _, _, _ = model(batch)

    inputs = denormalize(batch[:8], norm).cpu()
    outputs = denormalize(recon[:8], norm).cpu()

    rows = []
    for i in range(inputs.shape[0]):
        inp = (inputs[i].float().permute(1, 2, 0).numpy() * 255.0).astype("uint8")
        out = (outputs[i].float().permute(1, 2, 0).numpy() * 255.0).astype("uint8")
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
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        drop_last=False,
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve()

    normalization = load_normalization(args.normalization_stats.resolve(), args.norm_mode)
    image_size = infer_image_size(manifest_path)

    train_loader = maybe_make_loader(
        manifest=manifest_path,
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
        manifest=manifest_path,
        split="val",
        normalization=normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_val_samples,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"Using device: {device}")
    print(f"Input image size: {image_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader) if val_loader is not None else 0}")
    print(f"AMP: {args.amp} ({args.amp_dtype})")
    print(f"Compile: {args.compile} ({args.compile_mode})")
    print(f"Activation checkpointing: {args.activation_checkpointing}")

    model = VAE(
        in_channels=3,
        latent_channels=args.latent_channels,
        image_size=image_size,
        activation_checkpointing=args.activation_checkpointing,
    )
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)

    config = VAETrainConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        adversarial_weight=args.adversarial_weight,
        disc_lr=args.disc_lr,
        disc_base_channels=args.disc_base_channels,
        disc_n_layers=args.disc_n_layers,
        use_amp=args.amp,
        amp_dtype=args.amp_dtype,
        checkpoint_every=args.checkpoint_every,
        resume_checkpoint=args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None,
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
