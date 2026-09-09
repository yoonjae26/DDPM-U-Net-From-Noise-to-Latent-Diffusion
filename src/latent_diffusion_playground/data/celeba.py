from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PARTITION_MAP = {0: "train", 1: "val", 2: "test"}
FOCUS_ATTRIBUTES = ("Male", "Smiling", "Young")


@dataclass(frozen=True)
class CelebAPaths:
    data_root: Path
    images_dir: Path
    attributes_csv: Path
    partitions_csv: Path
    bbox_csv: Path | None = None
    landmarks_csv: Path | None = None


def _first_existing(paths: list[Path]) -> Path:
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the expected paths exist: {paths}")


def resolve_celeba_paths(data_root: Path) -> CelebAPaths:
    root = data_root.resolve()
    images_dir = _first_existing(
        [
            root / "img_align_celeba" / "img_align_celeba",
            root / "img_align_celeba",
        ]
    )

    return CelebAPaths(
        data_root=root,
        images_dir=images_dir,
        attributes_csv=root / "list_attr_celeba.csv",
        partitions_csv=root / "list_eval_partition.csv",
        bbox_csv=(root / "list_bbox_celeba.csv") if (root / "list_bbox_celeba.csv").exists() else None,
        landmarks_csv=(root / "list_landmarks_align_celeba.csv") if (root / "list_landmarks_align_celeba.csv").exists() else None,
    )


def load_celeba_metadata(paths: CelebAPaths) -> pd.DataFrame:
    attributes = pd.read_csv(paths.attributes_csv)
    partitions = pd.read_csv(paths.partitions_csv)

    metadata = attributes.merge(partitions, on="image_id", how="inner")
    metadata = metadata.rename(columns={"partition": "partition_id"})
    metadata["split"] = metadata["partition_id"].map(PARTITION_MAP)

    if paths.bbox_csv is not None:
        bbox = pd.read_csv(paths.bbox_csv)
        metadata = metadata.merge(bbox, on="image_id", how="left")

    if paths.landmarks_csv is not None:
        landmarks = pd.read_csv(paths.landmarks_csv)
        metadata = metadata.merge(landmarks, on="image_id", how="left")

    metadata["source_path"] = metadata["image_id"].map(lambda image_id: str(paths.images_dir / image_id))
    metadata["image_exists"] = metadata["source_path"].map(lambda path: Path(path).exists())

    return metadata


def convert_attributes_to_binary(metadata: pd.DataFrame) -> pd.DataFrame:
    converted = metadata.copy()
    excluded = {"image_id", "partition_id", "split", "source_path", "image_exists"}

    for column in converted.columns:
        if column in excluded:
            continue
        series = converted[column]
        if pd.api.types.is_numeric_dtype(series):
            values = set(series.dropna().astype(int).unique().tolist())
            if values.issubset({-1, 1}):
                converted[column] = ((series + 1) // 2).astype("Int64")

    return converted


def build_dataset_summary(metadata: pd.DataFrame) -> dict[str, object]:
    split_counts = metadata["split"].value_counts().sort_index().to_dict()
    focus_ratios: dict[str, dict[str, float]] = {}

    for attribute in FOCUS_ATTRIBUTES:
        if attribute not in metadata.columns:
            continue
        ratios = metadata.groupby("split", observed=True)[attribute].mean().to_dict()
        focus_ratios[attribute] = {str(split): float(value) for split, value in ratios.items()}

    return {
        "total_samples": int(len(metadata)),
        "missing_images": int((~metadata["image_exists"]).sum()),
        "split_counts": {str(split): int(count) for split, count in split_counts.items()},
        "focus_attribute_positive_ratio": focus_ratios,
    }