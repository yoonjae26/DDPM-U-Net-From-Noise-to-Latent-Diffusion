from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PHASE_TO_SCRIPT = {
    1: "scripts/phase1_dataset_pipeline.py",
    2: "scripts/phase2_train_vae.py",
    3: "scripts/phase3_latent_space_viewer.py",
    4: "scripts/phase4_diffusion_core.py",
    5: "scripts/phase5_latent_unet.py",
    6: "scripts/phase6_train_latent_diffusion.py",
    7: "scripts/phase7_reverse_diffusion.py",
    8: "scripts/phase8_forward_diffusion_viewer.py",
    9: "scripts/phase9_reverse_diffusion_viewer.py",
    11: "scripts/phase11_scheduler_stability_lab.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run project by phase")
    parser.add_argument("phase", type=int, help="Phase number")
    args, extra = parser.parse_known_args()

    if extra and extra[0] == "--":
        extra = extra[1:]

    if args.phase not in PHASE_TO_SCRIPT:
        print(
            f"Phase {args.phase} is scaffolded but not implemented yet. "
            "Implemented phases: "
            f"{sorted(PHASE_TO_SCRIPT.keys())}"
        )
        sys.exit(1)

    script = ROOT / PHASE_TO_SCRIPT[args.phase]
    cmd = [sys.executable, str(script), *extra]
    print(f"Running phase {args.phase}: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
