# Stable Diffusion Pretrained VAE Demo

This demo uses a pretrained Stable Diffusion VAE to show the actual latent-space bridge used in latent diffusion.

## Setup

```bash
cd contributions/yuseungjun/stable-diffusion-vae
uv sync
```

The first run downloads the VAE model from Hugging Face.

## Run

Run without an image path to open a file selection window:

```bash
uv run python reconstruct_image.py
```

Choose the photo to compress and reconstruct in the window that appears.

The generated visualizations and metrics are saved in `output`.

You can still provide the image path directly:

```bash
uv run python reconstruct_image.py --input-image img/your-image.jpg
```

Optional:

```bash
uv run python reconstruct_image.py --input-image img/your-image.jpg --max-size 512
```

## Outputs

The script saves results to `output`.

- `input_resized.png`: resized input image
- `vae_reconstruction.png`: image reconstructed by the pretrained VAE
- `comparison.png`: side-by-side comparison for slides
- `latent_channels.png`: visualization of the four compressed latent channels
- `difference_heatmap.png`: enhanced reconstruction-error heatmap
- `compression_pipeline.png`: presentation-ready input, latent, reconstruction, and metrics overview
- `latent_stats.txt`: shapes, compression ratio, and reconstruction metrics

## Presentation Script

Stable Diffusion은 이미지를 바로 픽셀 공간에서 처리하지 않고, pretrained VAE Encoder로 이미지를 latent 공간에 압축한다. 그 latent 공간에서 Diffusion/U-Net이 작동하고, 마지막에 VAE Decoder가 latent를 다시 이미지로 복원한다.

Use this result to explain:

`Image -> VAE Encoder -> Latent -> Diffusion/U-Net -> VAE Decoder -> Generated Image`
