from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class NormalizationConfig:
    mean: tuple[float, float, float]
    std: tuple[float, float, float]


def _load_manifest(manifest_path: Path) -> pd.DataFrame:
    if manifest_path.suffix == ".parquet":
        return pd.read_parquet(manifest_path)
    if manifest_path.suffix == ".csv":
        return pd.read_csv(manifest_path)
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def _resolve_processed_path(sample: pd.Series, manifest_dir: Path) -> Path:
    candidates: list[Path] = []

    raw_processed_path = sample.get("processed_path")
    if isinstance(raw_processed_path, str) and raw_processed_path:
        processed_path = Path(raw_processed_path)
        candidates.append(processed_path)
        if not processed_path.is_absolute():
            candidates.append(manifest_dir / processed_path)

    raw_relative_path = sample.get("processed_relative_path")
    if isinstance(raw_relative_path, str) and raw_relative_path:
        candidates.append(manifest_dir / raw_relative_path)

    split = sample.get("split")
    image_id = sample.get("image_id")
    if isinstance(split, str) and isinstance(image_id, str):
        candidates.append(manifest_dir / "preprocessed_images" / split / image_id)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Unable to resolve processed image path from manifest row. "
        f"Tried: {[str(path) for path in candidates]}"
    )


class CelebAManifestDataset(Dataset[torch.Tensor]):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        normalization: NormalizationConfig,
        max_samples: int | None = None,
    ) -> None:
        manifest_path = manifest_path.resolve()
        manifest = _load_manifest(manifest_path)
        manifest = manifest.loc[manifest["split"] == split].reset_index(drop=True)

        if max_samples is not None:
            manifest = manifest.iloc[:max_samples].reset_index(drop=True)

        if manifest.empty:
            raise ValueError(f"No samples found for split '{split}'.")

        self.samples = manifest
        self.manifest_dir = manifest_path.parent
        self.normalization = normalization
        self.mean = torch.tensor(normalization.mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(normalization.std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self.samples.iloc[index]
        image_path = _resolve_processed_path(sample, self.manifest_dir)
        with Image.open(image_path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return (tensor - self.mean) / self.std