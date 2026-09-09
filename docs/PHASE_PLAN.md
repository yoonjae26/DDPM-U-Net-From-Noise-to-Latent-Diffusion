# Latent Diffusion Playground — Phase Plan

## Project goal

Build an interactive platform that helps people understand, step by step:

- How a VAE compresses an image into a latent space
- What the latent representation actually looks like
- How DDPM operates in that latent space (forward and reverse process)
- The role of the U-Net in predicting and removing noise
- How noise schedulers (linear vs. cosine) compare
- The architecture behind models like Stable Diffusion

Dataset: [CelebA (CelebFaces Attributes Dataset)](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Current status

- **Phase 1 — Dataset Pipeline**: implemented.
- **Phase 2 — VAE Development**: implemented. Final VAE: `artifacts/phase2_lpips/vae_last.pt` —
  a spatial-latent VAE (4×16×16, batch=512) fine-tuned with an added LPIPS perceptual loss
  (`--perceptual-weight 0.1`) on top of a converged MSE+KL run. Visibly sharper reconstructions
  than a plain MSE+KL VAE.
- **Phase 6 — Latent Diffusion Training**: implemented. Final U-Net: `artifacts/phase6_v2`
  (base_channels=256, 500 epochs, trained on the LPIPS-tuned VAE's latents). Its validation loss
  (0.327) is numerically *higher* than an earlier, smaller run (base_channels=192, val_loss=0.295),
  but its decoded samples are visually sharper — see "Loss vs. perceptual quality" below for why
  that isn't a contradiction.
- **Phase 7 — Reverse Diffusion**: **fixed** — see "Known issue: reverse sampling exploded to
  pure noise" below. DDPM sampling (`--sampler ddpm`) now reliably produces recognizable faces.
- **Phase 11 — Scheduler Stability Lab**: implemented (`scripts/phase11_scheduler_stability_lab.py`)
  — reproduces and visualizes the bug above, comparing linear vs. cosine schedules with x0-clipping
  on/off.
- **Phase 12 — Interactive Web Application**: implemented (`scripts/web_demo.py`, Streamlit). Three
  tabs: VAE Explorer (encode/decode + reparameterization toggle), Forward Diffusion (noise a latent
  at any timestep, linear vs. cosine), Reverse Diffusion (generate from pure noise with DDPM/DDIM).
- **Phases 3–5, 8–10, 13–14**: scaffolded / partially implemented, ready to build on incrementally.

## Known issue: reverse sampling exploded to pure noise (fixed)

Even with a well-trained VAE + U-Net (low validation loss), `phase7_reverse_diffusion.py` always
decoded to pure noise. Root cause, confirmed experimentally: the cosine noise schedule clamps
`beta` to `cosine_max_beta=0.999` for the last few timesteps near `t=T`, producing a
`1/sqrt(alpha_t) ≈ 31.6` amplification factor on the very first ancestral sampling step. With no
bound on the predicted `x0`, this amplifies any small model error into a runaway trajectory —
latent std goes from ~1 to ~3000 within 1000 steps, and the U-Net's output stops responding to its
input entirely once it's this far out of distribution.

**Fix**: `p_sample_step` and `ddim_sample_step` in
`src/latent_diffusion_playground/diffusion/reverse.py` now clip the predicted `x0` to
`[-clip_range, clip_range]` at every step before folding it back into the noise estimate
("dynamic thresholding" — standard practice in every real DDPM/LDM implementation, previously
missing here). `p_sample_loop` and the Phase 6 training-time sample preview inherit the fix
automatically (default `clip_x0=True`).

**Open follow-up**: DDIM sampling with large step strides (e.g. 50–200 steps) still struggles at
this same extreme tail even with clipping — a subtler, separate instability worth exploring
further in the Scheduler Stability Lab.

## Loss vs. perceptual quality

`phase6_v2`'s validation loss (0.327) is higher than the earlier `phase6_final` (0.295), yet its
generated faces look sharper. This isn't a contradiction: validation loss measures noise-prediction
MSE in *latent* space, and the two runs use latents from two different VAEs. The LPIPS-tuned VAE's
latent carries more high-frequency detail, which is inherently harder to predict noise on — even
though the resulting *decoded images* look better. Visual inspection, not the raw loss number, is
the real quality signal for a generative model like this.

## Attempted: adversarial (GAN) loss for a sharper VAE decoder (not adopted)

`phase2_lpips` (MSE+KL+LPIPS) reconstructs and generates clearly better than plain MSE+KL, but
eyes/mouth/skin texture stay soft — the missing ingredient in every real sharp VAE (Stable
Diffusion's autoencoder included) is an adversarial (PatchGAN) loss. Implemented in
`src/latent_diffusion_playground/models/discriminator.py` +
`training/vae_trainer.py` (`--adversarial-weight`, `--disc-lr`), including VQGAN-style adaptive
discriminator weighting (Esser et al., 2021, `_adaptive_discriminator_weight`) to auto-balance the
two losses.

Three fine-tuning attempts on top of `phase2_lpips`, all converging to the same failure signature
(`train_disc` → ~0, i.e. the discriminator perfectly separates real photos from reconstructions
within a handful of epochs):

1. Fixed `--adversarial-weight 0.1`, `--disc-lr` = generator LR: discriminator collapsed to ~0 in
   ~3 epochs; no change to reconstruction (`train_recon` stayed frozen at 0.008537 throughout —
   the adversarial gradient contributed nothing useful before the run was stopped).
2. Same, `--disc-lr` 10x lower: identical collapse speed — ruled out a simple LR-imbalance
   explanation.
3. Adaptive weighting enabled: correctly detected the imbalance and shrank its own contribution
   toward 0 (`train_dweight` → ~0.0003), which *did* prevent any degradation (recon/KL/LPIPS terms
   stayed perfectly stable) but also meant the GAN term stopped mattering at all — no sharpening.

**Root cause**: the adaptive-weight formula scales the adversarial gradient relative to the
reconstruction+LPIPS gradient's magnitude. `phase2_lpips` was resumed from a checkpoint already
fully converged on those terms (that gradient is ~0 at a local minimum), so *any* nonzero
adversarial gradient looks disproportionately large by comparison and gets suppressed almost
entirely — a mismatch between "fine-tune GAN on an already-converged generator" and a technique
designed for a generator still actively learning recon+perceptual loss (which is how VQGAN/LDM
autoencoders are normally trained: all terms together from the start, not GAN bolted on
afterward).

**Not pursued further** (diminishing returns for the time invested): a correct fix would mean
either training recon+KL+LPIPS+GAN together from an earlier, non-fully-converged checkpoint, or
substantially weakening the discriminator (fewer channels/layers) so it can't win so trivially
against an already-blurry-by-construction VAE. **`phase2_lpips` remains the project's final VAE**
— the GAN code stays in the repo (default `--adversarial-weight 0.0`, fully opt-in) as a
documented, working building block for anyone who wants to continue this experiment.

## Phase specifications

### Phase 1 — Dataset Pipeline
Read, classify, preprocess (center-crop + resize), and normalize CelebA images and metadata.

### Phase 2 — VAE Development
- **Encoder**: `Image → Conv layers → μ, σ`
- **Reparameterization**: `z = μ + σ·ε` (the central VAE formula)
- **Decoder**: `Latent → Upsampling → Image`
- **Training loss**: reconstruction loss + KL divergence (+ optional LPIPS / adversarial loss)

### Phase 3 — Latent Space Viewer
Upload an image, run it through the encoder, and inspect the resulting latent vector: its shape,
value distribution, and dimensionality. Rare in DDPM tutorial repos, but a great way to build
intuition for what "latent space" actually contains.

### Phase 4 — Diffusion Core Engine
The heart of DDPM: linear and cosine noise **schedulers** (computing `beta`, `alpha`, `alpha_hat`),
and the **forward process** — `Image → VAE Encoder → Latent → Add Noise` — run on the *latent*
space rather than raw pixels, viewable at any timestep (t=0, 100, 300, 700, 1000, ...).

### Phase 5 — Latent U-Net Development
A U-Net built from scratch (encoder, bottleneck, decoder, skip connections, sinusoidal time
embedding) that takes a noisy latent + timestep and predicts the noise that was added.

### Phase 6 — Latent Diffusion Training
`Image → VAE Encoder → Latent → Add Noise → U-Net → Predict Noise → Loss`, with logging of loss
curves, checkpoints, and generated samples throughout training.

### Phase 7 — Reverse Diffusion
Iteratively denoise a latent starting from pure noise (DDPM ancestral sampling, or DDIM for fewer
steps), then decode through the VAE to get a final image.

### Phase 8 — Forward Diffusion Viewer
Visualize an image at increasing noise levels (10%, 50%, 100%, ...), toggling between image space
and latent space.

### Phase 9 — Reverse Diffusion Viewer
Visualize the denoising trajectory from random latent → clean latent → decoded image, exportable
as a GIF (`latent_reverse.gif`).

### Phase 10 — Explainable U-Net + VAE
Feature-map visualization for both networks: VAE encoder/decoder activations, and U-Net
encoder/bottleneck/decoder layers.

### Phase 11 — Scheduler Comparison Lab
Compare linear vs. cosine schedulers on loss, sampling speed, and (see the stability lab already
implemented) numerical stability.

### Phase 12 — Interactive Web Application
A unified app with tabs for VAE exploration, forward diffusion, reverse diffusion, U-Net feature
maps, and scheduler comparison.

### Phase 13 — GIF Export System
Generate `vae_reconstruction.gif`, `forward.gif`, `reverse.gif`, `sampling.gif` for the README and
demo.

### Phase 14 — GitHub Release
Finalize documentation and demo assets: VAE architecture, latent space visualization, forward/
reverse diffusion GIFs, VAE reconstructions, U-Net feature maps, scheduler benchmark, interactive
demo link.

## Execution

```bash
uv sync
uv run python scripts/phase1_dataset_pipeline.py --data-root ../Data --image-size 128
```

Or run by phase index:

```bash
uv run python scripts/run_phase.py 1 -- --data-root ../Data --image-size 128
```

Quick sanity run on a subset:

```bash
uv run python scripts/phase1_dataset_pipeline.py --data-root ../Data --max-images 2000 --stats-sample-size 1000
```

VAE training (see README for the full recommended command, including LPIPS):

```bash
uv sync --extra train
uv run python scripts/phase2_train_vae.py --epochs 50 --batch-size 512 --num-workers 16 \
  --latent-channels 4 --kl-weight 1e-3 --amp --amp-dtype bf16
```

Quick sanity run:

```bash
uv run python scripts/phase2_train_vae.py --epochs 1 --batch-size 8 --max-train-samples 128 --max-val-samples 64
```

Interactive web demo:

```bash
uv sync --extra gui
uv run streamlit run scripts/web_demo.py
```
