from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from latent_diffusion_playground.models.discriminator import (
    PatchDiscriminator,
    discriminator_hinge_loss,
    generator_hinge_loss,
)
from latent_diffusion_playground.models.vae import VAE, vae_loss


@dataclass
class VAETrainConfig:
    epochs: int
    learning_rate: float
    kl_weight: float
    grad_clip: float | None = 1.0
    use_amp: bool = False
    amp_dtype: str = "bf16"
    checkpoint_every: int = 1
    resume_checkpoint: Path | None = None
    perceptual_weight: float = 0.0
    """LPIPS perceptual loss weight. 0 disables it (pure MSE+KL, original behavior).

    LPIPS expects inputs in [-1, 1] -- matches this project's 'diffusion' norm
    mode (mean=std=0.5) and the VAE decoder's Tanh output exactly, so recon/
    target tensors are fed to it as-is, no extra rescaling needed.
    """
    adversarial_weight: float = 0.0
    """PatchGAN adversarial loss weight. 0 disables it (default).

    This is what pushes the decoder toward genuinely sharp, high-frequency
    detail (eyes, mouth, skin texture) -- MSE and LPIPS alone tend to still
    average toward a blurry-but-plausible reconstruction. Best enabled as a
    fine-tuning stage on top of a VAE that already reconstructs well under
    MSE+KL(+LPIPS): an untrained decoder facing an untrained discriminator
    from scratch is a much less stable starting point.
    """
    disc_lr: float | None = None
    """Discriminator learning rate. Defaults to `learning_rate` when unset."""
    disc_base_channels: int = 64
    disc_n_layers: int = 3


def _unwrap_model(model: VAE) -> VAE:
    return getattr(model, "_orig_mod", model)


def _set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad_(flag)


def _adaptive_discriminator_weight(
    nll_loss: torch.Tensor,
    adv_g_loss: torch.Tensor,
    last_layer: torch.nn.Parameter,
    max_weight: float = 1e4,
) -> torch.Tensor:
    """VQGAN-style adaptive weighting (Esser et al. 2021, taming-transformers).

    Scales the adversarial term so its gradient at the decoder's last layer
    matches the reconstruction(+perceptual) gradient's magnitude, instead of
    a fixed --adversarial-weight. A discriminator's task (real photo vs. a
    VAE reconstruction) is usually much easier than the generator's, so it
    tends to converge first regardless of learning rate -- once it does, a
    fixed weight lets its (now large, confident) gradient dominate and starve
    reconstruction learning. This keeps the two contributions comparable in
    magnitude throughout training instead of manually re-tuning the weight.
    """
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    adv_grads = torch.autograd.grad(adv_g_loss, last_layer, retain_graph=True)[0]
    weight = torch.norm(nll_grads.float()) / (torch.norm(adv_grads.float()) + 1e-4)
    return torch.clamp(weight, 0.0, max_weight).detach()


def _checkpoint_payload(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_val: float,
    perceptual_weight: float,
    adversarial_weight: float,
    discriminator: PatchDiscriminator | None,
    disc_optimizer: torch.optim.Optimizer | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "epoch": epoch,
        "model_state_dict": _unwrap_model(model).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": best_val,
        "perceptual_weight": perceptual_weight,
        "adversarial_weight": adversarial_weight,
    }
    if discriminator is not None and disc_optimizer is not None:
        payload["discriminator_state_dict"] = discriminator.state_dict()
        payload["disc_optimizer_state_dict"] = disc_optimizer.state_dict()
    return payload


def _amp_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported amp dtype: {name}")
    return mapping[name]


def train_vae(
    model: VAE,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    config: VAETrainConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    model.to(device)

    perceptual_model = None
    if config.perceptual_weight > 0.0:
        import lpips  # lazy import: only required when perceptual loss is enabled

        perceptual_model = lpips.LPIPS(net="alex").to(device)
        perceptual_model.eval()
        _set_requires_grad(perceptual_model, False)

    discriminator = None
    disc_optimizer = None
    if config.adversarial_weight > 0.0:
        discriminator = PatchDiscriminator(
            in_channels=3,
            base_channels=config.disc_base_channels,
            n_layers=config.disc_n_layers,
        ).to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=config.disc_lr or config.learning_rate
        )

    amp_dtype = _amp_dtype(config.amp_dtype)
    autocast_enabled = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=autocast_enabled and amp_dtype == torch.float16,
    )

    best_val = float("inf")
    start_epoch = 1
    if config.resume_checkpoint is not None and config.resume_checkpoint.exists():
        payload = torch.load(config.resume_checkpoint, map_location=device)
        _unwrap_model(model).load_state_dict(payload["model_state_dict"])
        # Initialize scheduler state properly if resuming
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, last_epoch=int(payload.get("epoch", 0)) - 1)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler_state = payload.get("scaler_state_dict")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        best_val = float(payload.get("best_val", best_val))
        start_epoch = int(payload.get("epoch", 0)) + 1

        checkpoint_perceptual_weight = float(payload.get("perceptual_weight", 0.0))
        checkpoint_adversarial_weight = float(payload.get("adversarial_weight", 0.0))
        if (
            checkpoint_perceptual_weight != config.perceptual_weight
            or checkpoint_adversarial_weight != config.adversarial_weight
        ):
            # best_val was computed under a different loss composition (e.g. LPIPS or
            # the GAN term just added/changed weight) -- it is not comparable to the
            # new val_total scale, so "save vae_best.pt if improved" would never fire.
            print(
                f"   Loss composition changed (perceptual_weight="
                f"{checkpoint_perceptual_weight}->{config.perceptual_weight}, "
                f"adversarial_weight={checkpoint_adversarial_weight}->{config.adversarial_weight}): "
                f"resetting best_val so vae_best.pt can be saved again."
            )
            best_val = float("inf")

        if (
            discriminator is not None
            and disc_optimizer is not None
            and checkpoint_adversarial_weight == config.adversarial_weight
            and "discriminator_state_dict" in payload
        ):
            discriminator.load_state_dict(payload["discriminator_state_dict"])
            disc_optimizer_state = payload.get("disc_optimizer_state_dict")
            if isinstance(disc_optimizer_state, dict):
                disc_optimizer.load_state_dict(disc_optimizer_state)

        print(f"Resumed training from {config.resume_checkpoint} at epoch {start_epoch}.")

    log_lines: list[str] = []

    def _generator_forward(batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """VAE forward pass + every generator-side loss term (recon/KL/LPIPS/adv-G).

        Discriminator params are frozen (requires_grad=False) for this call so
        no gradient is wasted on them -- gradient still flows through to
        `recon` normally, only the discriminator's own `.grad` is skipped.
        Does NOT touch the discriminator's weights; that happens afterwards in
        `_discriminator_step`, once the generator's backward+step is done and
        the graph built here is no longer needed. Updating the discriminator
        first would mutate the very weight tensors this call's backward pass
        needs, which autograd forbids (in-place-modified-during-backward).
        """
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            recon, mu, logvar, _ = model(batch)
            total_loss, recon_loss, kl_loss = vae_loss(
                recon=recon,
                target=batch,
                mu=mu,
                logvar=logvar,
                kl_weight=config.kl_weight,
            )
            percep_loss = torch.zeros((), device=device)
            if perceptual_model is not None:
                percep_loss = perceptual_model(recon, batch).mean()
                total_loss = total_loss + config.perceptual_weight * percep_loss

            adv_g_loss = torch.zeros((), device=device)
            # Eval runs under torch.no_grad() (no graph -> can't measure a gradient
            # ratio), so default to a flat 1.0 there -- only used for monitoring
            # (val_total / best_val selection), never for an actual backward pass.
            adaptive_weight = torch.ones((), device=device)
            if discriminator is not None:
                _set_requires_grad(discriminator, False)
                fake_logits = discriminator(recon)
                adv_g_loss = generator_hinge_loss(fake_logits)
                if torch.is_grad_enabled():
                    nll_loss = recon_loss + config.perceptual_weight * percep_loss
                    last_layer = _unwrap_model(model).decoder[-2].weight
                    adaptive_weight = _adaptive_discriminator_weight(nll_loss, adv_g_loss, last_layer)
                total_loss = total_loss + adaptive_weight * config.adversarial_weight * adv_g_loss
                _set_requires_grad(discriminator, True)

        return total_loss, recon, recon_loss, kl_loss, percep_loss, adv_g_loss, adaptive_weight

    def _discriminator_step(batch: torch.Tensor, recon: torch.Tensor, update: bool) -> torch.Tensor:
        """Hinge loss on real vs. reconstructed-and-detached images.

        `recon` must come from a generator step whose backward+optimizer.step
        has already completed -- see the docstring on `_generator_forward`.
        `update` is False during validation (loss is still reported, but the
        discriminator is not trained on val data).
        """
        if discriminator is None or disc_optimizer is None:
            return torch.zeros((), device=device)

        if update:
            disc_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            real_logits = discriminator(batch)
            fake_logits_detached = discriminator(recon.detach())
            d_loss = discriminator_hinge_loss(real_logits, fake_logits_detached)
        if update:
            d_loss.backward()
            disc_optimizer.step()
        return d_loss

    for epoch in range(start_epoch, config.epochs + 1):
        model.train()
        if discriminator is not None:
            discriminator.train()
        train_total = 0.0
        train_recon = 0.0
        train_kl = 0.0
        train_percep = 0.0
        train_adv_g = 0.0
        train_disc = 0.0
        train_dweight = 0.0

        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            total_loss, recon, recon_loss, kl_loss, percep_loss, adv_g_loss, dweight = _generator_forward(batch)

            scaler.scale(total_loss).backward()
            if config.grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # Discriminator update happens only after the generator's backward+step
            # is fully done -- see _generator_forward's docstring for why the order matters.
            d_loss = _discriminator_step(batch, recon, update=True)

            train_total += float(total_loss.item())
            train_recon += float(recon_loss.item())
            train_kl += float(kl_loss.item())
            train_percep += float(percep_loss.item())
            train_adv_g += float(adv_g_loss.item())
            train_disc += float(d_loss.item())
            train_dweight += float(dweight.item())

        n_train = len(train_loader)
        train_total /= max(n_train, 1)
        train_recon /= max(n_train, 1)
        train_kl /= max(n_train, 1)
        train_percep /= max(n_train, 1)
        train_adv_g /= max(n_train, 1)
        train_disc /= max(n_train, 1)
        train_dweight /= max(n_train, 1)

        scheduler.step()

        val_total = float("nan")
        val_recon = float("nan")
        val_kl = float("nan")
        val_percep = float("nan")
        val_adv_g = float("nan")
        val_disc = float("nan")

        if val_loader is not None:
            model.eval()
            if discriminator is not None:
                discriminator.eval()
            running_total = 0.0
            running_recon = 0.0
            running_kl = 0.0
            running_percep = 0.0
            running_adv_g = 0.0
            running_disc = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device, non_blocking=True)
                    total_loss, recon, recon_loss, kl_loss, percep_loss, adv_g_loss, _dweight = _generator_forward(batch)
                    d_loss = _discriminator_step(batch, recon, update=False)
                    running_total += float(total_loss.item())
                    running_recon += float(recon_loss.item())
                    running_kl += float(kl_loss.item())
                    running_percep += float(percep_loss.item())
                    running_adv_g += float(adv_g_loss.item())
                    running_disc += float(d_loss.item())

            n_val = len(val_loader)
            val_total = running_total / max(n_val, 1)
            val_recon = running_recon / max(n_val, 1)
            val_kl = running_kl / max(n_val, 1)
            val_percep = running_percep / max(n_val, 1)
            val_adv_g = running_adv_g / max(n_val, 1)
            val_disc = running_disc / max(n_val, 1)

            if val_total < best_val:
                best_val = val_total
                torch.save(_unwrap_model(model).state_dict(), output_dir / "vae_best.pt")
                torch.save(
                    _checkpoint_payload(
                        model, optimizer, scaler, epoch, best_val,
                        config.perceptual_weight, config.adversarial_weight,
                        discriminator, disc_optimizer,
                    ),
                    output_dir / "checkpoint_best.pt",
                )

        torch.save(_unwrap_model(model).state_dict(), output_dir / "vae_last.pt")
        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            torch.save(
                _checkpoint_payload(
                    model, optimizer, scaler, epoch, best_val,
                    config.perceptual_weight, config.adversarial_weight,
                    discriminator, disc_optimizer,
                ),
                output_dir / "checkpoint_last.pt",
            )

        line = (
            f"epoch={epoch} "
            f"train_total={train_total:.6f} train_recon={train_recon:.6f} train_kl={train_kl:.6f} "
            f"train_percep={train_percep:.6f} train_adv_g={train_adv_g:.6f} train_disc={train_disc:.6f} "
            f"train_dweight={train_dweight:.6f} "
            f"val_total={val_total:.6f} val_recon={val_recon:.6f} val_kl={val_kl:.6f} "
            f"val_percep={val_percep:.6f} val_adv_g={val_adv_g:.6f} val_disc={val_disc:.6f}"
        )
        log_lines.append(line)
        print(line)

    (output_dir / "train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
