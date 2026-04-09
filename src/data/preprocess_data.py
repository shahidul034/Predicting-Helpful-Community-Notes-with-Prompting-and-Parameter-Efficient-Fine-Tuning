"""
Preprocess Community Notes data.

This script:
1. Loads raw TSV files
2. Filters notes based on status and ratings
3. Creates train/val/test splits based on temporal cutoff
4. Saves processed data as parquet files
"""

import os
import sys
import argparse
import gzip
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()
config = Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Community Notes data")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing raw TSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed data",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="community_notes_cleaned.parquet",
        help="Output parquet filename",
    )
    return parser.parse_args()


def load_notes(filepath: Path) -> pd.DataFrame:
    """Load notes TSV file."""
    logger.info(f"Loading notes from {filepath}")
    if filepath.suffix == ".gz":
        df = pd.read_csv(filepath, sep="\t", compression="gzip")
    else:
        df = pd.read_csv(filepath, sep="\t")
    logger.info(f"Loaded {len(df)} notes")
    return df


def load_ratings(filepath: Path) -> pd.DataFrame:
    """Load ratings TSV file."""
    logger.info(f"Loading ratings from {filepath}")
    if filepath.suffix == ".gz":
        df = pd.read_csv(filepath, sep="\t", compression="gzip")
    else:
        df = pd.read_csv(filepath, sep="\t")
    logger.info(f"Loaded {len(df)} ratings")
    return df


def load_status_history(filepath: Path) -> pd.DataFrame:
    """Load note status history TSV file."""
    logger.info(f"Loading status history from {filepath}")
    if filepath.suffix == ".gz":
        df = pd.read_csv(filepath, sep="\t", compression="gzip")
    else:
        df = pd.read_csv(filepath, sep="\t")
    logger.info(f"Loaded {len(df)} status records")
    return df


def filter_notes(df: pd.DataFrame, config_dict: dict) -> pd.DataFrame:
    """Filter notes based on configuration."""
    initial_count = len(df)

    # Get configuration values
    exclude_statuses = config_dict.get("filtering", {}).get(
        "exclude_statuses", ["NEEDS_MORE_RATINGS"]
    )
    min_ratings = config_dict.get("filtering", {}).get("min_ratings", 5)
    status_col = config_dict.get("columns", {}).get("status", "currentLabelStatus")

    # Exclude notes with certain statuses
    if exclude_statuses:
        mask = ~df[status_col].isin(exclude_statuses)
        df = df[mask]
        logger.info(f"After excluding statuses: {len(df)} notes")

    # Filter by minimum ratings if 'ratingsCount' column exists
    if "ratingsCount" in df.columns and min_ratings > 0:
        mask = df["ratingsCount"] >= min_ratings
        df = df[mask]
        logger.info(f"After filtering by min ratings ({min_ratings}): {len(df)} notes")

    logger.info(f"Filtered {initial_count - len(df)} notes")
    return df


def create_labels(df: pd.DataFrame, config_dict: dict) -> pd.DataFrame:
    """Create binary labels from status column."""
    status_col = config_dict.get("columns", {}).get("status", "currentLabelStatus")
    label_mapping = config_dict.get("label_mapping", {})

    # Map status to binary labels
    status_to_label = {"CURRENTLY_RATED_HELPFUL": 1, "CURRENTLY_RATED_NOT_HELPFUL": 0}

    df["label"] = df[status_col].map(status_to_label)

    # Drop rows with NaN labels
    initial_count = len(df)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    logger.info(f"Dropped {initial_count - len(df)} notes with missing labels")

    return df


def temporal_split(df: pd.DataFrame, config_dict: dict) -> dict:
    """Split data temporally based on creation date."""
    date_col = config_dict.get("columns", {}).get("created_at", "createdAt")

    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    # Get cutoff dates
    train_cutoff = config_dict.get("temporal_split", {}).get(
        "train_cutoff_date", "2024-10-01"
    )
    val_cutoff = config_dict.get("temporal_split", {}).get(
        "val_cutoff_date", "2024-11-01"
    )

    train_cutoff = pd.to_datetime(train_cutoff)
    val_cutoff = pd.to_datetime(val_cutoff)

    # Create splits
    train_mask = df[date_col] < train_cutoff
    val_mask = (df[date_col] >= train_cutoff) & (df[date_col] < val_cutoff)
    test_mask = df[date_col] >= val_cutoff

    splits = {"train": df[train_mask], "val": df[val_mask], "test": df[test_mask]}

    logger.info(f"Temporal split:")
    logger.info(f"  Train: {len(splits['train'])} notes (before {train_cutoff.date()})")
    logger.info(
        f"  Val:   {len(splits['val'])} notes ({train_cutoff.date()} to {val_cutoff.date()})"
    )
    logger.info(f"  Test:  {len(splits['test'])} notes (after {val_cutoff.date()})")

    return splits


def save_splits(splits: dict, output_dir: Path, output_file: str) -> None:
    """Save splits to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, df in splits.items():
        filepath = (
            output_dir / f"{output_file.replace('.parquet', f'_{split_name}.parquet')}"
        )
        df.to_parquet(filepath, index=False)
        logger.info(f"Saved {split_name} split to {filepath}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Get data configuration with defaults
    data_config = config.get("data")
    if data_config is None:
        data_config = {
            "filtering": {"exclude_statuses": ["NEEDS_MORE_RATINGS"], "min_ratings": 5},
            "columns": {"status": "currentLabelStatus", "created_at": "createdAt"},
            "temporal_split": {
                "train_cutoff_date": "2024-10-01",
                "val_cutoff_date": "2024-11-01",
            },
            "label_mapping": {
                "CURRENTLY_RATED_HELPFUL": 1,
                "CURRENTLY_RATED_NOT_HELPFUL": 0,
            },
        }

    logger.info("Starting data preprocessing...")

    # Load raw data
    notes_file = input_dir / "notes.tsv.gz"
    notes_df = load_notes(notes_file)

    # Filter notes
    notes_df = filter_notes(notes_df, data_config)

    # Create labels
    notes_df = create_labels(notes_df, data_config)

    # Temporal split
    splits = temporal_split(notes_df, data_config)

    # Save splits
    save_splits(splits, output_dir, args.output_file)

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
