from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.phases.latent_space_viewer import load_viewer_artifacts
from latent_diffusion_playground.data.torch_dataset import NormalizationConfig

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3: Latent Space Viewer")
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase2_high_res" / "vae_last.pt",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "normalization_stats.yaml",
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        help="Path to an image to inspect. If not provided, you might want to pick one from artifacts/phase1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase3",
    )
    return parser.parse_args()

def preprocess_image(img_path: Path, norm_config: NormalizationConfig, size: int = 128) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm_config.mean, std=norm_config.std),
    ])
    return transform(img).unsqueeze(0)

def denormalize(tensor: torch.Tensor, norm: NormalizationConfig) -> np.ndarray:
    mean = torch.tensor(norm.mean).view(3, 1, 1)
    std = torch.tensor(norm.std).view(3, 1, 1)
    img = tensor.squeeze().cpu() * std + mean
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load Model
    artifacts = load_viewer_artifacts(
        checkpoint_path=args.vae_checkpoint,
        normalization_stats_path=args.normalization_stats,
        norm_mode="diffusion",
        device=device
    )
    
    norm_config = artifacts.normalization

    # Select an image
    if args.image_path and args.image_path.exists():
        target_img = args.image_path
    else:
        # Try to find an image from phase 1 preprocessed images
        img_dir = Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "preprocessed_images" / "val"
        target_img = next(img_dir.glob("*.jpg"))
        print(f"No image provided, using sample: {target_img.name}")

    # Run Inference
    img_tensor = preprocess_image(target_img, norm_config).to(device)
    
    with torch.no_grad():
        mu, logvar = artifacts.model.encode(img_tensor)
        z = artifacts.model.reparameterize(mu, logvar)
        recon = artifacts.model.decode(z)

    # Plotting
    fig = plt.figure(figsize=(15, 10))
    plt.suptitle(f"Latent Space Inspection: {target_img.name}", fontsize=16)

    # 1. Original Image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(denormalize(img_tensor, norm_config))
    ax1.set_title("Original")
    ax1.axis("off")

    # 2. Reconstructed Image
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(denormalize(recon, norm_config))
    ax2.set_title("VAE Reconstruction")
    ax2.axis("off")

    # 3. Latent Vector Heatmap (Reshaped for visualization)
    ax3 = plt.subplot(2, 3, 3)
    z_np = z.squeeze().cpu().numpy()
    side = int(np.sqrt(len(z_np)))
    if side * side == len(z_np):
        im = ax3.imshow(z_np.reshape(side, side), cmap="viridis")
    else:
        im = ax3.imshow(z_np.reshape(1, -1), aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax3)
    ax3.set_title(f"Latent Vector z (dim={len(z_np)})")

    # 4. Latent Distribution (Histogram)
    ax4 = plt.subplot(2, 1, 2)
    ax4.hist(z_np, bins=50, alpha=0.7, color='blue', label='z values')
    ax4.axvline(x=z_np.mean(), color='red', linestyle='--', label=f'Mean: {z_np.mean():.4f}')
    ax4.set_title("Distribution of values in z")
    ax4.set_xlabel("Value")
    ax4.set_ylabel("Frequency")
    ax4.legend()

    plt.tight_layout()
    out_path = args.output_dir / f"latent_analysis_{target_img.stem}.png"
    plt.savefig(out_path)
    print(f"Analysis saved to: {out_path}")
    plt.close()

if __name__ == "__main__":
    main()