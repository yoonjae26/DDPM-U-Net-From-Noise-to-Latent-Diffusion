from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import AutoencoderKL
from PIL import Image


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "output"


def select_input_image() -> Path:
    from tkinter import Tk, filedialog

    root = Tk()
    root.title("사진 선택")
    root.overrideredirect(True)
    root.update_idletasks()

    center_x = (root.winfo_screenwidth() - 2) // 2
    center_y = (root.winfo_screenheight() - 2) // 2
    root.geometry(f"2x2+{center_x}+{center_y}")
    root.attributes("-alpha", 0.01)
    root.attributes("-topmost", True)
    root.deiconify()
    root.lift()
    root.focus_force()
    root.update()

    try:
        selected_path = filedialog.askopenfilename(
            parent=root,
            title="VAE로 압축하고 복원할 사진을 선택하세요",
            initialdir=PROJECT_DIR / "img",
            filetypes=(
                ("Image files", "*.jpg *.jpeg *.png *.webp *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ),
        )
    finally:
        root.destroy()

    if not selected_path:
        raise SystemExit("사진 선택이 취소되었습니다.")
    return Path(selected_path)


def choose_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resize_to_multiple(image: Image.Image, max_size: int, multiple: int = 8) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    scale = min(max_size / max(width, height), 1.0)
    new_width = max(multiple, int(width * scale) // multiple * multiple)
    new_height = max(multiple, int(height * scale) // multiple * multiple)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return (tensor * 2.0 - 1.0).to(device)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    tensor = (tensor.clamp(-1, 1) + 1.0) / 2.0
    array = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    array = (array * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def save_comparison(original: Image.Image, reconstruction: Image.Image, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(original)
    axes[0].set_title("Input image")
    axes[0].axis("off")
    axes[1].imshow(reconstruction)
    axes[1].set_title("Pretrained VAE reconstruction")
    axes[1].axis("off")
    fig.suptitle("Stable Diffusion VAE: Image -> Latent -> Image")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def normalize_feature_map(feature_map: np.ndarray) -> np.ndarray:
    low, high = np.percentile(feature_map, [1, 99])
    if high <= low:
        return np.zeros_like(feature_map)
    return np.clip((feature_map - low) / (high - low), 0, 1)


def save_latent_channels(latent: torch.Tensor, output_path: Path) -> None:
    channels = latent.squeeze(0).detach().float().cpu().numpy()
    fig, axes = plt.subplots(1, channels.shape[0], figsize=(10, 2.8))
    for index, (axis, channel) in enumerate(zip(axes, channels, strict=True)):
        axis.imshow(normalize_feature_map(channel), cmap="viridis", interpolation="nearest")
        axis.set_title(f"Latent channel {index + 1}")
        axis.axis("off")
    fig.suptitle(f"Compressed latent representation: {tuple(latent.shape)}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def calculate_metrics(
    original: Image.Image,
    reconstruction: Image.Image,
    latent: torch.Tensor,
) -> tuple[dict[str, float | int | tuple[int, ...]], np.ndarray]:
    original_array = np.asarray(original, dtype=np.float32) / 255.0
    reconstruction_array = np.asarray(reconstruction, dtype=np.float32) / 255.0
    absolute_difference = np.abs(original_array - reconstruction_array)

    mse = float(np.mean((original_array - reconstruction_array) ** 2))
    mae = float(np.mean(absolute_difference))
    psnr = float("inf") if mse == 0 else float(10 * np.log10(1.0 / mse))

    input_shape = (original_array.shape[2], original_array.shape[0], original_array.shape[1])
    latent_shape = tuple(latent.shape[1:])
    input_values = int(np.prod(input_shape))
    latent_values = int(np.prod(latent_shape))

    metrics: dict[str, float | int | tuple[int, ...]] = {
        "input_shape_chw": input_shape,
        "latent_shape_chw": latent_shape,
        "input_values": input_values,
        "latent_values": latent_values,
        "compression_ratio": input_values / latent_values,
        "mse": mse,
        "mae": mae,
        "psnr_db": psnr,
    }
    return metrics, absolute_difference.mean(axis=2)


def save_difference_heatmap(difference: np.ndarray, output_path: Path) -> None:
    display_max = max(float(np.percentile(difference, 99)), 1e-6)
    fig, ax = plt.subplots(figsize=(5, 5))
    image = ax.imshow(difference, cmap="inferno", vmin=0, vmax=display_max)
    ax.set_title("Absolute reconstruction error (enhanced)")
    ax.axis("off")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Mean RGB error")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_compression_pipeline(
    original: Image.Image,
    reconstruction: Image.Image,
    latent: torch.Tensor,
    difference: np.ndarray,
    metrics: dict[str, float | int | tuple[int, ...]],
    output_path: Path,
) -> None:
    channels = latent.squeeze(0).detach().float().cpu().numpy()
    display_max = max(float(np.percentile(difference, 99)), 1e-6)

    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 4, width_ratios=(1.15, 1, 1, 1.15))

    input_axis = fig.add_subplot(grid[0, 0])
    input_axis.imshow(original)
    input_axis.set_title(f"1. Input\n{metrics['input_shape_chw']} = {metrics['input_values']:,} values")
    input_axis.axis("off")

    difference_axis = fig.add_subplot(grid[1, 0])
    heatmap = difference_axis.imshow(difference, cmap="inferno", vmin=0, vmax=display_max)
    difference_axis.set_title("4. Reconstruction error\n(bright = larger difference)")
    difference_axis.axis("off")
    fig.colorbar(heatmap, ax=difference_axis, fraction=0.046, pad=0.04)

    latent_grid = grid[:, 1:3].subgridspec(2, 2, wspace=0.08, hspace=0.15)
    for index, channel in enumerate(channels):
        latent_axis = fig.add_subplot(latent_grid[index // 2, index % 2])
        latent_axis.imshow(
            normalize_feature_map(channel),
            cmap="viridis",
            interpolation="nearest",
        )
        latent_axis.set_title(f"Channel {index + 1}: {channel.shape[1]} x {channel.shape[0]}")
        latent_axis.axis("off")

    reconstruction_axis = fig.add_subplot(grid[0, 3])
    reconstruction_axis.imshow(reconstruction)
    reconstruction_axis.set_title("3. VAE reconstruction\nDecoded back to RGB")
    reconstruction_axis.axis("off")

    stats_axis = fig.add_subplot(grid[1, 3])
    stats_axis.axis("off")
    stats_axis.text(
        0.03,
        0.95,
        "Compression summary\n\n"
        f"Input: {metrics['input_values']:,} values\n"
        f"Latent: {metrics['latent_values']:,} values\n"
        f"Compression: {metrics['compression_ratio']:.1f}:1\n\n"
        f"MSE: {metrics['mse']:.6f}\n"
        f"MAE: {metrics['mae']:.6f}\n"
        f"PSNR: {metrics['psnr_db']:.2f} dB",
        va="top",
        fontsize=13,
        linespacing=1.45,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f2f2f2", "edgecolor": "#777777"},
    )

    fig.suptitle(
        f"Stable Diffusion VAE: Input -> 2. Compressed latent {metrics['latent_shape_chw']} -> Reconstruction",
        fontsize=17,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode and decode an image with a pretrained Stable Diffusion VAE."
    )
    parser.add_argument(
        "--input-image",
        type=Path,
        default=None,
        help="Image path. When omitted, a file selection window opens.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--max-size", type=int, default=512)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    input_image = args.input_image or select_input_image()
    if not input_image.is_file():
        raise SystemExit(f"입력 사진을 찾을 수 없습니다: {input_image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device()
    print(f"Using device: {device}")
    print(f"Loading VAE: {args.model_id}")

    vae = AutoencoderKL.from_pretrained(args.model_id).to(device)
    vae.eval()

    print(f"Selected image: {input_image}")
    original = Image.open(input_image)
    resized = resize_to_multiple(original, max_size=args.max_size)
    pixel_values = image_to_tensor(resized, device)

    latent = vae.encode(pixel_values).latent_dist.mode()
    scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))
    scaled_latent = latent * scaling_factor
    reconstruction_tensor = vae.decode(scaled_latent / scaling_factor).sample
    reconstruction = tensor_to_image(reconstruction_tensor)
    metrics, difference = calculate_metrics(resized, reconstruction, scaled_latent)

    resized.save(args.output_dir / "input_resized.png")
    reconstruction.save(args.output_dir / "vae_reconstruction.png")
    save_comparison(resized, reconstruction, args.output_dir / "comparison.png")
    save_latent_channels(scaled_latent, args.output_dir / "latent_channels.png")
    save_difference_heatmap(difference, args.output_dir / "difference_heatmap.png")
    save_compression_pipeline(
        resized,
        reconstruction,
        scaled_latent,
        difference,
        metrics,
        args.output_dir / "compression_pipeline.png",
    )

    stats = {
        "model_id": args.model_id,
        "input_image": str(input_image),
        "preprocessed_size": resized.size,
        "latent_shape": tuple(scaled_latent.shape),
        "scaling_factor": scaling_factor,
        "latent_mean": float(scaled_latent.mean().detach().cpu()),
        "latent_std": float(scaled_latent.std().detach().cpu()),
        **metrics,
    }
    (args.output_dir / "latent_stats.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in stats.items()) + "\n",
        encoding="utf-8",
    )

    print(f"Latent shape: {tuple(scaled_latent.shape)}")
    print(f"Compression ratio: {metrics['compression_ratio']:.1f}:1")
    print(f"MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr_db']:.2f} dB")
    print(f"Saved results to: {args.output_dir}")


if __name__ == "__main__":
    main()
