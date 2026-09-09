"""Download the pretrained VAE + U-Net checkpoints from a GitHub Release.

The trained weights (~440MB total) are too large for plain git, so they are
distributed as GitHub Release assets instead of being committed to the repo.
This script fetches them into `artifacts/` so the web demo and phase7 scripts
work immediately after a `git clone` or in a fresh GitHub Codespace.

One-time setup (repo maintainer only): create a release tagged
`checkpoints-v1` and upload these three files as its assets:
    artifacts/phase2_lpips/vae_last.pt
    artifacts/phase6_v2/unet_last.pt
    artifacts/phase6_v2/latent_scaling.json

    gh release create checkpoints-v1 \\
      artifacts/phase2_lpips/vae_last.pt \\
      artifacts/phase6_v2/unet_last.pt \\
      artifacts/phase6_v2/latent_scaling.json \\
      --title "Pretrained checkpoints v1" \\
      --notes "VAE (MSE+KL+LPIPS) + latent diffusion U-Net (base_channels=256)."

Usage:
    python scripts/download_checkpoints.py
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REPO = "yoonjae26/DDPM-U-Net-From-Noise-to-Latent-Diffusion"
RELEASE_TAG = "checkpoints-v1"
BASE_URL = f"https://github.com/{REPO}/releases/download/{RELEASE_TAG}"

FILES: dict[str, str] = {
    "artifacts/phase2_lpips/vae_last.pt": f"{BASE_URL}/vae_last.pt",
    "artifacts/phase6_v2/unet_last.pt": f"{BASE_URL}/unet_last.pt",
    "artifacts/phase6_v2/latent_scaling.json": f"{BASE_URL}/latent_scaling.json",
}


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _report(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        done = min(block_num * block_size, total_size)
        pct = 100 * done // total_size
        print(f"\r  {destination.name}: {pct}% ({done // 1_000_000}MB/{total_size // 1_000_000}MB)", end="")

    try:
        urllib.request.urlretrieve(url, destination, reporthook=_report)
        print()
    except urllib.error.HTTPError as exc:
        print()
        raise RuntimeError(
            f"Could not download {url} ({exc.code} {exc.reason}).\n"
            f"The release '{RELEASE_TAG}' may not exist yet on {REPO} -- "
            "see the setup instructions at the top of this script."
        ) from exc


def main() -> None:
    missing = {
        relative_path: url
        for relative_path, url in FILES.items()
        if not (ROOT / relative_path).exists()
    }
    if not missing:
        print("All checkpoints already present -- nothing to do.")
        return

    print(f"Downloading {len(missing)} checkpoint file(s) from GitHub Release '{RELEASE_TAG}'...")
    for relative_path, url in missing.items():
        _download(url, ROOT / relative_path)
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
