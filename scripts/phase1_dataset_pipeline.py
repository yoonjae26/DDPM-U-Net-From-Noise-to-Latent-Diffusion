from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src").resolve()))

from latent_diffusion_playground.data.celeba import (
    build_dataset_summary,
    convert_attributes_to_binary,
    load_celeba_metadata,
    resolve_celeba_paths,
)
from latent_diffusion_playground.data.preprocess import (
    compute_rgb_mean_std,
    preprocess_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1: CelebA data pipeline")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "Data",
        help="Path to CelebA data directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts" / "phase1",
        help="Directory to save processed files",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="Image size after center-crop and resize",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick experiments",
    )
    parser.add_argument(
        "--stats-sample-size",
        type=int,
        default=5000,
        help="Maximum number of images used for RGB mean/std",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    paths = resolve_celeba_paths(args.data_root)
    metadata = load_celeba_metadata(paths)
    metadata = convert_attributes_to_binary(metadata)

    summary = build_dataset_summary(metadata)
    summary_path = output_root / "dataset_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    processed_dir = output_root / "preprocessed_images"
    processed_df = preprocess_images(
        metadata,
        output_dir=processed_dir,
        image_size=args.image_size,
        max_images=args.max_images,
    )

    manifest_csv = output_root / "manifest.csv"
    manifest_parquet = output_root / "manifest.parquet"
    processed_df.to_csv(manifest_csv, index=False)
    processed_df.to_parquet(manifest_parquet, index=False)

    train_paths = processed_df.loc[processed_df["split"] == "train", "processed_path"].tolist()
    stats = compute_rgb_mean_std(train_paths, sample_size=args.stats_sample_size)

    stats_path = output_root / "normalization_stats.yaml"
    stats_path.write_text(yaml.safe_dump(stats, sort_keys=False), encoding="utf-8")

    print("Phase 1 completed")
    print(f"Summary: {summary_path}")
    print(f"Manifest CSV: {manifest_csv}")
    print(f"Manifest Parquet: {manifest_parquet}")
    print(f"Normalization stats: {stats_path}")


if __name__ == "__main__":
    main()
