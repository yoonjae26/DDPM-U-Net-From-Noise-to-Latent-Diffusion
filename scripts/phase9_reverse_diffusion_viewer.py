from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image
import torch

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.phases.reverse_diffusion_viewer import (  # noqa: E402
    export_gif,
    load_reverse_viewer_artifacts,
    parse_step_labels,
    run_reverse_trajectory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9: Reverse Diffusion Viewer")
    parser.add_argument(
        "--unet-checkpoint",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase6_h200_full" / "unet_last.pt",
    )
    parser.add_argument(
        "--vae-checkpoint",
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
        help="Path to latent scaling JSON from Phase 6",
    )
    parser.add_argument("--norm-mode", type=str, choices=["diffusion", "dataset"], default="diffusion")
    parser.add_argument("--scheduler", type=str, choices=["linear", "cosine"], default="cosine")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument(
        "--save-steps",
        type=str,
        default="1000,900,800,700,600,500,400,300,200,100,0",
        help="Comma-separated labels for visualization snapshots",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gif-duration-ms", type=int, default=140)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase9",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main() -> None:
    try:
        import gradio as gr
    except Exception as exc:
        raise RuntimeError("Gradio is required. Install with: pip install -e '.[webui]'") from exc

    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    gif_path = output_dir / "latent_reverse.gif"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifacts = load_reverse_viewer_artifacts(
        unet_checkpoint_path=args.unet_checkpoint.resolve(),
        vae_checkpoint_path=args.vae_checkpoint.resolve(),
        normalization_stats_path=args.normalization_stats.resolve(),
        norm_mode=args.norm_mode,
        device=device,
        latent_scaling_path=args.latent_scaling.resolve() if args.latent_scaling is not None else None,
    )

    print(f"Using device: {device}")
    print(f"UNet checkpoint: {args.unet_checkpoint.resolve()}")
    print(f"VAE checkpoint: {args.vae_checkpoint.resolve()}")
    print(f"Latent scaling factor: {artifacts.latent_scaling_factor:.6f}")

    def run_viewer(seed: int, scheduler_name: str, timesteps: int, save_steps: str, gif_duration_ms: int):
        step_map = parse_step_labels(save_steps, timesteps)
        snapshots = run_reverse_trajectory(
            artifacts=artifacts,
            scheduler_name=scheduler_name,
            timesteps=timesteps,
            seed=seed,
            step_label_map=step_map,
        )

        if not snapshots:
            raise ValueError("No snapshots generated. Check timesteps/save-steps values.")

        initial_latent = snapshots[0][2]
        clean_latent = snapshots[-1][2]
        final_image = snapshots[-1][3]

        gallery = []
        decoded_frames: list[Image.Image] = []
        for label, idx, latent_img, decoded_img in snapshots:
            gallery.append((decoded_img, f"t={label} (index={idx})"))
            decoded_frames.append(decoded_img)

        exported = export_gif(decoded_frames, gif_path, frame_duration_ms=max(int(gif_duration_ms), 1))

        summary = (
            "### Reverse Diffusion Summary\n"
            f"- Scheduler: {scheduler_name}\n"
            f"- Timesteps: {timesteps}\n"
            f"- Seed: {seed}\n"
            f"- Saved frames: {len(snapshots)}\n"
            f"- GIF: {exported}"
        )

        return initial_latent, gallery, clean_latent, final_image, str(exported), summary

    with gr.Blocks(title="Phase 9 Reverse Diffusion Viewer") as demo:
        gr.Markdown("# Phase 9 - Reverse Diffusion Viewer")
        gr.Markdown("Random Latent -> Denoising -> Clean Latent -> Decoded Image")

        with gr.Row():
            seed_in = gr.Number(value=args.seed, precision=0, label="Seed")
            scheduler_in = gr.Radio(choices=["linear", "cosine"], value=args.scheduler, label="Scheduler")
            timesteps_in = gr.Number(value=args.timesteps, precision=0, label="Timesteps")

        with gr.Row():
            save_steps_in = gr.Textbox(value=args.save_steps, label="Save Steps")
            gif_duration_in = gr.Number(value=args.gif_duration_ms, precision=0, label="GIF Frame Duration (ms)")

        run_btn = gr.Button("Run Reverse Diffusion", variant="primary")

        with gr.Row():
            random_latent = gr.Image(type="pil", label="Random Latent")
            clean_latent = gr.Image(type="pil", label="Clean Latent")

        denoise_gallery = gr.Gallery(label="Denoising (Decoded Frames)", columns=4, height=280)
        final_image = gr.Image(type="pil", label="Decoded Image")
        gif_file = gr.File(label="Export GIF (latent_reverse.gif)")
        summary = gr.Markdown(label="Summary")

        run_btn.click(
            fn=run_viewer,
            inputs=[seed_in, scheduler_in, timesteps_in, save_steps_in, gif_duration_in],
            outputs=[random_latent, denoise_gallery, clean_latent, final_image, gif_file, summary],
        )

    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
