from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Phase:
    id: int
    name: str
    description: str
    status: str


PHASES = [
    Phase(1, "Dataset Pipeline", "Read, classify, preprocess, and normalize CelebA metadata/images.", "implemented"),
    Phase(2, "VAE Development", "Build VAE encoder, reparameterization, decoder, and train loop.", "implemented"),
    Phase(3, "Latent Space Viewer", "Visualize latent vectors and distribution from uploaded images.", "planned"),
    Phase(4, "Diffusion Core Engine", "Schedulers and latent-space forward diffusion.", "planned"),
    Phase(5, "Latent U-Net Development", "Build U-Net with time embedding for noise prediction.", "planned"),
    Phase(6, "Latent Diffusion Training", "Train latent-space DDPM with logging and checkpoints.", "planned"),
    Phase(8, "Forward Diffusion Viewer", "Visualize noisy transitions in image/latent space.", "planned"),
    Phase(9, "Reverse Diffusion Viewer", "Stepwise denoising and decoded outputs.", "planned"),
    Phase(10, "Explainable U-Net + VAE", "Feature map visualization for interpretability.", "planned"),
    Phase(11, "Scheduler Comparison Lab", "Compare Linear vs Cosine on quality and speed.", "planned"),
    Phase(12, "Interactive Web Application", "Unified tabs for VAE, diffusion, U-Net features, benchmark.", "planned"),
    Phase(13, "GIF Export System", "Export animations for reconstruction, forward/reverse, sampling.", "planned"),
    Phase(14, "GitHub Release", "Finalize docs, demo assets, and release artifacts.", "planned"),
]
