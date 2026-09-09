from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
import yaml

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.data.torch_dataset import (
    CelebAManifestDataset,
    NormalizationConfig,
)
from latent_diffusion_playground.diffusion import make_schedule, q_sample, p_sample_loop
from latent_diffusion_playground.models.unet import LatentUNet
from latent_diffusion_playground.phases.latent_space_viewer import load_viewer_artifacts


# =========================================================
# ARGUMENTS
# =========================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6: Latent Diffusion Training")

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "manifest.parquet",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "normalization_stats.yaml",
    )
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase2_h200_full" / "vae_last.pt",
    )

    parser.add_argument("--norm-mode", type=str, choices=["diffusion", "dataset"], default="diffusion")
    parser.add_argument("--scheduler", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--timestep-sampling",
        type=str,
        choices=["uniform", "high_noise", "mixed"],
        default="mixed",
        help="Sampling policy for diffusion timesteps during training.",
    )
    parser.add_argument(
        "--val-timestep-sampling",
        type=str,
        choices=["uniform", "high_noise", "mixed"],
        default="uniform",
        help="Sampling policy for diffusion timesteps during validation.",
    )

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--base-channels", type=int, default=128)
    parser.add_argument("--channel-mults", type=str, default="1,2,4")
    parser.add_argument("--num-heads", type=int, default=8,
                        help="Number of attention heads in UNet self-attention layers.")
    parser.add_argument("--no-attention", action="store_true",
                        help="Disable self-attention in UNet (matches legacy no-attention architecture).")

    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, choices=["bf16", "fp16"], default="bf16")

    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase6",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint_last.pt path to resume interrupted training.",
    )
    parser.add_argument(
        "--reset-optimizer",
        action="store_true",
        help="When resuming, discard the saved optimizer state and start Adam fresh. "
             "Use this when resuming with a different --lr to avoid stale momentum.",
    )

    # =====================================================
    # ⭐ ADDED: LOG FREQUENCY CONTROL
    # =====================================================
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--latent-calib-batches",
        type=int,
        default=32,
        help="Number of train batches used to estimate latent std for diffusion scaling.",
    )
    parser.add_argument(
        "--latent-scaling-factor",
        type=float,
        default=None,
        help="Optional manual scaling factor. If unset, scaling is estimated as 1/std(latent).",
    )

    return parser.parse_args()


# =========================================================
# HELPERS
# =========================================================
def _parse_channel_mults(raw: str) -> tuple[int, ...]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("channel-mults cannot be empty")
    if any(v <= 0 for v in values):
        raise ValueError(f"channel-mults must be positive, got {values}")
    return tuple(values)


def _load_normalization(config_path: Path, mode: str) -> NormalizationConfig:
    stats = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    if mode == "dataset":
        mean = stats["mean"]
        std = stats["std"]
    else:
        mean = stats["diffusion_default_mean"]
        std = stats["diffusion_default_std"]

    return NormalizationConfig(mean=tuple(mean), std=tuple(std))


def _make_loader(
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
    )


def _denormalize(x: torch.Tensor, norm: NormalizationConfig) -> torch.Tensor:
    mean = torch.tensor(norm.mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(norm.std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return torch.clamp(x * std + mean, 0.0, 1.0)


def _amp_dtype(name: str) -> torch.dtype:
    return torch.bfloat16 if name == "bf16" else torch.float16


def _to_diffusion_latent(vae, images: torch.Tensor) -> torch.Tensor:
    """Encode image to VAE spatial mu. Shape: B × latent_channels × H × W."""
    mu, _ = vae.encode(images)
    return mu


def _estimate_latent_std(vae, loader: DataLoader, device: torch.device, max_batches: int) -> float:
    sum_sq = 0.0
    count = 0
    with torch.no_grad():
        for i, images in enumerate(loader):
            if i >= max_batches:
                break
            images = images.to(device, non_blocking=True)
            latent = _to_diffusion_latent(vae, images)
            sum_sq += float(torch.sum(latent.pow(2)).item())
            count += int(latent.numel())
    if count == 0:
        raise RuntimeError("No samples available for latent std estimation")
    return float(np.sqrt(sum_sq / count))


def _save_sample_grid(
    vae,
    unet,
    images: torch.Tensor,
    schedule,
    norm: NormalizationConfig,
    device: torch.device,
    latent_scaling: float,
    out_path: Path,
) -> None:
    vae.eval()
    unet.eval()
    sample = images[:4].to(device, non_blocking=True)
    # Pick a mid-range timestep to observe the UNet's x0 prediction ability
    viz_t_idx = int(0.5 * (schedule.timesteps - 1))
    t_viz = torch.tensor([viz_t_idx] * sample.shape[0], dtype=torch.long, device=device)

    with torch.no_grad():
        latent = _to_diffusion_latent(vae, sample)
        latent_scaled = latent * latent_scaling
        recon = vae.decode_feature_map(latent)

        # Predict x0 from the noisy latent at t=500
        eps_viz = torch.randn_like(latent_scaled)
        z_t_scaled = q_sample(latent_scaled, t_viz, schedule, noise=eps_viz)
        pred_noise = unet(z_t_scaled, t_viz)
        
        a = schedule.sqrt_alpha_hat[t_viz].view(-1, 1, 1, 1)
        b = schedule.sqrt_one_minus_alpha_hat[t_viz].view(-1, 1, 1, 1)
        x0_scaled_pred = (z_t_scaled - b * pred_noise) / torch.clamp(a, min=1e-8)
        
        # Generate a brand-new sample from pure noise (batched for speed)
        noise_init = torch.randn((4, vae.latent_channels, vae.feature_map_size, vae.feature_map_size), device=device)
        sampled_latent_scaled = p_sample_loop(unet, noise_init.shape, schedule, device=device)
        gen_imgs = _denormalize(vae.decode_feature_map(sampled_latent_scaled / latent_scaling), norm)

        rows: list[list[Image.Image]] = []
        for i in range(sample.shape[0]):
            row = []
            # Column 1: original image
            base = _denormalize(sample[i:i+1], norm)
            row.append(Image.fromarray((base[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
            # Column 2: VAE reconstruction (if this column is blurry, diffusion can never be sharper than this)
            rec = _denormalize(recon[i:i+1], norm)
            row.append(Image.fromarray((rec[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
            # Column 3: predicted x0 at t=500 (shows what the UNet 'sees' in the noise)
            p_img = _denormalize(vae.decode_feature_map(x0_scaled_pred[i:i+1] / latent_scaling), norm)
            row.append(Image.fromarray((p_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
            # Column 4: brand-new generated image
            row.append(Image.fromarray((gen_imgs[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)))
            
            rows.append(row)

    w, h = rows[0][0].size
    cols = len(rows[0])
    canvas = Image.new("RGB", (cols * w, len(rows) * h), color=(248, 248, 248))
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            canvas.paste(img, (c * w, r * h))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _convert_bn_to_gn_state_dict(bn_state: dict) -> dict:
    """
    Convert BatchNorm state_dict to compatible GroupNorm state_dict.
    
    BatchNorm has: weight, bias, running_mean, running_var, num_batches_tracked
    GroupNorm has: weight, bias (affine parameters only, no running stats)
    
    This function removes BN-specific keys so the checkpoint can load into a GN model.
    We keep weight and bias which have the same semantics between BN and GN (scale γ and shift β).
    """
    gn_state = {}
    
    for key, value in bn_state.items():
        # Skip BN-specific keys: running_mean, running_var, num_batches_tracked
        if any(x in key for x in ['running_mean', 'running_var', 'num_batches_tracked']):
            continue
        # Keep weight, bias, and other parameters
        gn_state[key] = value
    
    return gn_state


def _sample_timesteps(
    batch_size: int,
    timesteps: int,
    device: torch.device,
    schedule,
    mode: str,
) -> torch.Tensor:
    if mode == "uniform":
        return torch.randint(0, timesteps, (batch_size,), device=device)

    # High-noise emphasis: sample t using probabilities proportional to sigma_t.
    sigma = schedule.sqrt_one_minus_alpha_hat
    sigma_probs = sigma / torch.clamp(sigma.sum(), min=1e-12)

    if mode == "high_noise":
        probs = sigma_probs
    else:
        # Mixed policy: combine uniform and high-noise sampling.
        # Prevents low-timestep underfitting while keeping late-step supervision strong.
        uniform_probs = torch.full_like(sigma_probs, 1.0 / float(timesteps))
        probs = 0.5 * uniform_probs + 0.5 * sigma_probs

    return torch.multinomial(probs, num_samples=batch_size, replacement=True)


# =========================================================
# MAIN
# =========================================================
def main() -> None:
    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================================
    # SPEED OPTIMIZATION (CUDA)
    # =====================================================
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    normalization = _load_normalization(args.normalization_stats.resolve(), args.norm_mode)

    train_loader = _make_loader(
        manifest=args.manifest.resolve(),
        split="train",
        normalization=normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_train_samples,
        shuffle=True,
    )

    val_loader = _make_loader(
        manifest=args.manifest.resolve(),
        split="val",
        normalization=normalization,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_val_samples,
        shuffle=False,
    )

    if train_loader is None:
        raise RuntimeError("Train split not found")

    # =====================================================
    # VAE
    # =====================================================
    vae_artifacts = load_viewer_artifacts(
        checkpoint_path=args.vae_checkpoint.resolve(),
        normalization_stats_path=args.normalization_stats.resolve(),
        norm_mode=args.norm_mode,
        device=device,
    )

    vae = vae_artifacts.model
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        batch0 = next(iter(train_loader)).to(device, non_blocking=True)
        latent0 = _to_diffusion_latent(vae, batch0)

    in_channels = int(latent0.shape[1])
    latent_size = int(latent0.shape[-1])

    latent_std = _estimate_latent_std(vae, train_loader, device, max_batches=args.latent_calib_batches)
    if args.latent_scaling_factor is not None:
        latent_scaling = float(args.latent_scaling_factor)
    else:
        latent_scaling = 1.0 / max(latent_std, 1e-8)

    scaling_payload = {
        "latent_std": latent_std,
        "latent_scaling_factor": latent_scaling,
        "calib_batches": args.latent_calib_batches,
    }
    (output_dir / "latent_scaling.json").write_text(json.dumps(scaling_payload, indent=2), encoding="utf-8")

    # =====================================================
    # MODEL
    # =====================================================
    unet = LatentUNet(
        in_channels=in_channels,
        base_channels=args.base_channels,
        channel_mults=_parse_channel_mults(args.channel_mults),
        num_heads=args.num_heads,
        use_attention=not args.no_attention,
    ).to(device)

    schedule = make_schedule(args.scheduler, args.timesteps, device=device)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.lr)
    scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    amp_dtype = _amp_dtype(args.amp_dtype)
    use_autocast = args.amp and device.type == "cuda"

    scaler = torch.amp.GradScaler(device=device.type, enabled=use_autocast and amp_dtype == torch.float16)

    best_val = float("inf")
    logs: list[str] = []
    start_epoch = 1

    resume_path = args.resume_checkpoint.resolve() if args.resume_checkpoint is not None else None
    if resume_path is not None:
        payload = torch.load(resume_path, map_location=device)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError(f"Unsupported resume checkpoint format: {resume_path}")

        # ⭐ Handle BN→GN conversion if checkpoint contains BatchNorm keys
        model_state = payload["model_state_dict"]
        has_bn_keys = any('running_mean' in k for k in model_state.keys())
        
        if has_bn_keys:
            print("⚠️  Checkpoint contains BatchNorm keys, converting to GroupNorm-compatible state...")
            model_state = _convert_bn_to_gn_state_dict(model_state)
            unet.load_state_dict(model_state, strict=False)
            print("   ℹ️  Skipping optimizer state (parameter names changed)")
        else:
            unet.load_state_dict(model_state, strict=True)
            if not args.reset_optimizer and "optimizer_state_dict" in payload:
                optimizer.load_state_dict(payload["optimizer_state_dict"])
            elif args.reset_optimizer:
                print("   ℹ️  --reset-optimizer: discarding saved optimizer state, Adam starts fresh.")

        if not args.reset_optimizer and "scaler_state_dict" in payload:
            scaler.load_state_dict(payload["scaler_state_dict"])

        best_val = float(payload.get("best_val", best_val))
        start_epoch = int(payload.get("epoch", 0)) + 1

        existing_log = output_dir / "train_log.txt"
        if existing_log.exists():
            logs = [line for line in existing_log.read_text(encoding="utf-8").splitlines() if line.strip()]

        print(f"Resuming from checkpoint: {resume_path}")
        print(f"Resume start epoch: {start_epoch}")

    if start_epoch > args.epochs:
        print(f"Nothing to do: start_epoch={start_epoch} > epochs={args.epochs}")
        return

    global_step = (start_epoch - 1) * len(train_loader)
    total_steps = len(train_loader) * args.epochs

    print(f"Device: {device}")
    print(f"Total steps: {total_steps}")
    print(f"Latent shape: {list(latent0.shape)}")
    print(f"Latent std: {latent_std:.6f}, scaling_factor: {latent_scaling:.6f}")

    # =====================================================
    # TRAIN LOOP
    # =====================================================
    for epoch in range(start_epoch, args.epochs + 1):
        unet.train()
        train_loss_sum = 0.0

        for images in train_loader:
            images = images.to(device, non_blocking=True)

            with torch.no_grad():
                latent = _to_diffusion_latent(vae, images)
                latent = latent * latent_scaling

            t = _sample_timesteps(
                batch_size=latent.shape[0],
                timesteps=args.timesteps,
                device=device,
                schedule=schedule,
                mode=args.timestep_sampling,
            )
            noise = torch.randn_like(latent)
            noisy_latent = q_sample(latent, t, schedule, noise=noise)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                pred_noise = unet(noisy_latent, t)
                loss = torch.nn.functional.mse_loss(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss_sum += float(loss.item())

            # =================================================
            # ⭐ ADDED: CLEAN ETA LOGGING
            # =================================================
            global_step += 1

            if global_step % args.log_every == 0:
                print(f"[Step {global_step}/{total_steps}] loss={loss.item():.6f}")

        train_loss = train_loss_sum / max(len(train_loader), 1)
        scheduler_lr.step()

        # =====================================================
        # VALIDATION
        # =====================================================
        val_loss = float("nan")
        if val_loader is not None:
            unet.eval()
            val_sum = 0.0

            with torch.no_grad():
                for images in val_loader:
                    images = images.to(device, non_blocking=True)
                    latent = _to_diffusion_latent(vae, images)
                    latent = latent * latent_scaling

                    t = _sample_timesteps(
                        batch_size=latent.shape[0],
                        timesteps=args.timesteps,
                        device=device,
                        schedule=schedule,
                        mode=args.val_timestep_sampling,
                    )
                    noise = torch.randn_like(latent)
                    noisy_latent = q_sample(latent, t, schedule, noise=noise)

                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_autocast):
                        pred_noise = unet(noisy_latent, t)
                        loss = torch.nn.functional.mse_loss(pred_noise, noise)

                    val_sum += float(loss.item())

            val_loss = val_sum / max(len(val_loader), 1)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(unet.state_dict(), output_dir / "unet_best.pt")

        torch.save(unet.state_dict(), output_dir / "unet_last.pt")
        if args.checkpoint_every > 0 and epoch % args.checkpoint_every == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unet.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "best_val": best_val,
                    "scheduler": args.scheduler,
                    "timesteps": args.timesteps,
                    "in_channels": in_channels,
                    "latent_size": latent_size,
                    "base_channels": args.base_channels,
                    "channel_mults": list(_parse_channel_mults(args.channel_mults)),
                    "latent_std": latent_std,
                    "latent_scaling_factor": latent_scaling,
                },
                output_dir / "checkpoint_last.pt",
            )

        sample_loader = val_loader if val_loader is not None else train_loader
        sample_batch = next(iter(sample_loader))
        sample_out = output_dir / "samples" / f"epoch_{epoch:03d}.png"
        _save_sample_grid(
            vae=vae,
            unet=unet,
            images=sample_batch,
            schedule=schedule,
            norm=normalization,
            device=device,
            latent_scaling=latent_scaling,
            out_path=sample_out,
        )
        for old in sorted((output_dir / "samples").glob("epoch_*.png"))[:-1]:
            old.unlink(missing_ok=True)

        latent_stats = {
            "epoch": epoch,
            "latent_mean": float(latent0.mean().item()),
            "latent_std": float(latent0.std().item()),
            "latent_scaling_factor": latent_scaling,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        (output_dir / "latent_stats.json").write_text(json.dumps(latent_stats, indent=2), encoding="utf-8")

        # =====================================================
        # EPOCH LOG
        # =====================================================
        line = f"epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f}"
        print(line)
        logs.append(line)

        (output_dir / "train_log.txt").write_text("\n".join(logs) + "\n", encoding="utf-8")

    print("Phase 6 completed")
    print(f"Outputs: {output_dir}")


if __name__ == "__main__":
    main()