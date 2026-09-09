from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.diffusion import make_schedule, q_sample  # noqa: E402
from latent_diffusion_playground.phases.latent_space_viewer import load_viewer_artifacts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 8: Forward Diffusion Viewer")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase2_h200_full" / "vae_last.pt",
    )
    parser.add_argument(
        "--normalization-stats",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1" / "normalization_stats.yaml",
    )
    parser.add_argument(
        "--latent-scaling",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase6_h200_full" / "latent_scaling.json",
    )
    parser.add_argument("--norm-mode", type=str, choices=["diffusion", "dataset"], default="diffusion")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def _center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def _normalize(image: Image.Image, mean: tuple[float, float, float], std: tuple[float, float, float]) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
    return (tensor - mean_t) / std_t


def _denormalize(x: torch.Tensor, mean: tuple[float, float, float], std: tuple[float, float, float]) -> Image.Image:
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    y = torch.clamp(x * std_t + mean_t, 0.0, 1.0)
    arr = (y[0].detach().float().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _latent_to_heatmap(latent: torch.Tensor) -> Image.Image:
    # Aggregate channels for a compact latent-space visualization.
    heat = latent[0].detach().float().cpu().abs().mean(dim=0).numpy()
    heat = heat - heat.min()
    denom = float(heat.max()) if float(heat.max()) > 0 else 1.0
    heat = heat / denom
    rgb = np.stack([heat, np.sqrt(heat), 1.0 - heat], axis=-1)
    return Image.fromarray((rgb * 255.0).astype(np.uint8)).resize((256, 256), Image.Resampling.NEAREST)


def main() -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError("Gradio is required. Install with: pip install -e '.[webui]'") from exc

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    artifacts = load_viewer_artifacts(
        checkpoint_path=args.checkpoint.resolve(),
        normalization_stats_path=args.normalization_stats.resolve(),
        norm_mode=args.norm_mode,
        device=device,
    )

    latent_scaling = 1.0
    if args.latent_scaling.exists():
        payload = json.loads(args.latent_scaling.read_text(encoding="utf-8"))
        latent_scaling = float(payload.get("latent_scaling_factor", 1.0))

    levels = [0.1, 0.5, 1.0]

    def render(upload_image: Image.Image, scheduler_name: str, view_mode: str):
        if upload_image is None:
            raise ValueError("Please upload an image")

        processed = _center_crop_resize(upload_image.convert("RGB"), artifacts.image_size)
        x = _normalize(processed, artifacts.normalization.mean, artifacts.normalization.std).to(device)

        with torch.no_grad():
            mu, _ = artifacts.model.encode(x)
            latent = artifacts.model.decoder_input(mu).view(-1, 256, artifacts.model.feature_map_size, artifacts.model.feature_map_size)
            latent = artifacts.model.decoder_input(mu).view(-1, artifacts.model.base_channels * 8, artifacts.model.feature_map_size, artifacts.model.feature_map_size)
            latent = latent * latent_scaling

        schedule = make_schedule(scheduler_name, args.timesteps, device=device)
        eps = torch.randn_like(latent)

        outputs = []
        for level in levels:
            t_idx = min(int(level * (args.timesteps - 1)), args.timesteps - 1)
            t = torch.tensor([t_idx], dtype=torch.long, device=device)
            with torch.no_grad():
                z_t = q_sample(latent, t, schedule, noise=eps)
                if view_mode == "Image Space":
                    img = _denormalize(
                        artifacts.model.decode_feature_map(z_t / max(latent_scaling, 1e-8)),
                        artifacts.normalization.mean,
                        artifacts.normalization.std,
                    )
                else:
                    img = _latent_to_heatmap(z_t)
            outputs.append(img)

        latent_view = _latent_to_heatmap(latent)
        return processed, latent_view, outputs[0], outputs[1], outputs[2]

    with gr.Blocks(title="Phase 8 Forward Diffusion Viewer") as demo:
        gr.Markdown("# Phase 8 - Forward Diffusion Viewer")
        gr.Markdown("Image -> Encoded Latent -> Noise Level 10% / 50% / 100% with toggle for Image Space or Latent Space")

        with gr.Row():
            upload = gr.Image(type="pil", label="Input Image")
            scheduler = gr.Radio(choices=["linear", "cosine"], value="cosine", label="Scheduler")
            mode = gr.Radio(choices=["Image Space", "Latent Space"], value="Image Space", label="View Mode")

        run_btn = gr.Button("Run Forward Diffusion", variant="primary")

        with gr.Row():
            original = gr.Image(type="pil", label="Preprocessed Image")
            encoded_latent = gr.Image(type="pil", label="Encoded Latent")

        with gr.Row():
            noise10 = gr.Image(type="pil", label="Noise Level 10%")
            noise50 = gr.Image(type="pil", label="Noise Level 50%")
            noise100 = gr.Image(type="pil", label="Noise Level 100%")

        run_btn.click(
            fn=render,
            inputs=[upload, scheduler, mode],
            outputs=[original, encoded_latent, noise10, noise50, noise100],
        )

    print(f"Using device: {device}")
    print(f"Loaded VAE: {args.checkpoint.resolve()}")
    print(f"Latent scaling factor: {latent_scaling:.6f}")
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
