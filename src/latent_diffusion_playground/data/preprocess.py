from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def _center_crop_resize(image: Image.Image, image_size: int) -> Image.Image:
    width, height = image.size
    crop_size = min(width, height)
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    image = image.crop((left, top, left + crop_size, top + crop_size))
    return image.resize((image_size, image_size), Image.Resampling.LANCZOS)


def preprocess_images(
    metadata: pd.DataFrame,
    output_dir: Path,
    image_size: int,
    max_images: int | None = None,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    subset = metadata.copy()
    if max_images is not None:
        subset = subset.iloc[:max_images].copy()

    processed_paths: list[str] = []
    processed_relative_paths: list[str] = []
    widths: list[int] = []
    heights: list[int] = []

    iterator = subset.itertuples(index=False)
    for row in tqdm(iterator, total=len(subset), desc="Preprocessing CelebA"):
        split_dir = output_dir / row.split
        split_dir.mkdir(parents=True, exist_ok=True)
        destination = split_dir / row.image_id

        with Image.open(row.source_path) as image:
            image = image.convert("RGB")
            image = _center_crop_resize(image, image_size)
            image.save(destination, format="JPEG", quality=95)
            widths.append(image.width)
            heights.append(image.height)

        processed_paths.append(str(destination))
        processed_relative_paths.append(str(Path(output_dir.name) / row.split / row.image_id))

    subset["processed_path"] = processed_paths
    subset["processed_relative_path"] = processed_relative_paths
    subset["processed_width"] = widths
    subset["processed_height"] = heights
    return subset


def compute_rgb_mean_std(image_paths: list[str], sample_size: int | None = None) -> dict[str, object]:
    if not image_paths:
        raise ValueError("No image paths provided for normalization statistics.")

    if sample_size is not None and sample_size > 0 and len(image_paths) > sample_size:
        indices = np.linspace(0, len(image_paths) - 1, num=sample_size, dtype=int)
        selected_paths = [image_paths[index] for index in indices]
    else:
        selected_paths = list(image_paths)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_squared_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for path in tqdm(selected_paths, total=len(selected_paths), desc="Computing RGB stats"):
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0

        pixels = array.reshape(-1, 3)
        channel_sum += pixels.sum(axis=0)
        channel_squared_sum += np.square(pixels).sum(axis=0)
        total_pixels += pixels.shape[0]

    mean = channel_sum / total_pixels
    variance = channel_squared_sum / total_pixels - np.square(mean)
    std = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))

    return {
        "sample_size": int(len(selected_paths)),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "diffusion_default_mean": [0.5, 0.5, 0.5],
        "diffusion_default_std": [0.5, 0.5, 0.5],
    }