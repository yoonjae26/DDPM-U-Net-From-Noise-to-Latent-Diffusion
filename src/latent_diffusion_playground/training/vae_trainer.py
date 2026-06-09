from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from latent_diffusion_playground.models.vae import VAE, vae_loss


@dataclass
class VAETrainConfig:
    epochs: int
    learning_rate: float
    kl_weight: float
    grad_clip: float | None = 1.0


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

    model.to(device)

    best_val = float("inf")
    log_lines: list[str] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_total = 0.0
        train_recon = 0.0
        train_kl = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)

            recon, mu, logvar, _ = model(batch)
            total_loss, recon_loss, kl_loss = vae_loss(
                recon=recon,
                target=batch,
                mu=mu,
                logvar=logvar,
                kl_weight=config.kl_weight,
            )

            total_loss.backward()
            if config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            train_total += float(total_loss.item())
            train_recon += float(recon_loss.item())
            train_kl += float(kl_loss.item())

        n_train = len(train_loader)
        train_total /= max(n_train, 1)
        train_recon /= max(n_train, 1)
        train_kl /= max(n_train, 1)

        val_total = float("nan")
        val_recon = float("nan")
        val_kl = float("nan")

        if val_loader is not None:
            model.eval()
            running_total = 0.0
            running_recon = 0.0
            running_kl = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    recon, mu, logvar, _ = model(batch)
                    total_loss, recon_loss, kl_loss = vae_loss(
                        recon=recon,
                        target=batch,
                        mu=mu,
                        logvar=logvar,
                        kl_weight=config.kl_weight,
                    )
                    running_total += float(total_loss.item())
                    running_recon += float(recon_loss.item())
                    running_kl += float(kl_loss.item())

            n_val = len(val_loader)
            val_total = running_total / max(n_val, 1)
            val_recon = running_recon / max(n_val, 1)
            val_kl = running_kl / max(n_val, 1)

            if val_total < best_val:
                best_val = val_total
                torch.save(model.state_dict(), output_dir / "vae_best.pt")

        torch.save(model.state_dict(), output_dir / "vae_last.pt")

        line = (
            f"epoch={epoch} "
            f"train_total={train_total:.6f} train_recon={train_recon:.6f} train_kl={train_kl:.6f} "
            f"val_total={val_total:.6f} val_recon={val_recon:.6f} val_kl={val_kl:.6f}"
        )
        log_lines.append(line)
        print(line)

    (output_dir / "train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
