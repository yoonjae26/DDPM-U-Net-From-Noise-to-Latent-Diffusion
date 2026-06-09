from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """Centralized filesystem layout for all phases."""

    project_root: Path
    data_root: Path
    outputs_root: Path

    @property
    def celeba_images_dir(self) -> Path:
        return self.data_root / "img_align_celeba"

    @property
    def metadata_dir(self) -> Path:
        return self.data_root


def default_paths(project_root: Path | None = None) -> ProjectPaths:
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    return ProjectPaths(
        project_root=root,
        data_root=(root.parent / "Data").resolve(),
        outputs_root=(root / "artifacts").resolve(),
    )
