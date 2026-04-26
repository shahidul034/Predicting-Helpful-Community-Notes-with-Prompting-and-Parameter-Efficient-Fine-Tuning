"""
Exploratory Data Analysis for Community Notes.

This script generates:
1. Distribution of note statuses
2. Note length statistics
3. Temporal trends
4. Class balance analysis
5. Sample notes visualization
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logging

logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDA on Community Notes data")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/visualizations",
        help="Directory to save visualizations",
    )
    return parser.parse_args()


def load_data(input_dir: Path) -> dict:
    """Load all splits from parquet files."""
    splits = {}
    for filepath in input_dir.glob("*.parquet"):
        split_name = filepath.stem.split("_")[-1]
        if split_name in ["train", "val", "test"]:
            splits[split_name] = pd.read_parquet(filepath)
            logger.info(f"Loaded {split_name}: {len(splits[split_name])} samples")
    return splits


def plot_label_distribution(splits: dict, output_dir: Path) -> None:
    """Plot label distribution across splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (split_name, df) in zip(axes, splits.items()):
        if "label" in df.columns:
            label_counts = df["label"].value_counts().sort_index()
            labels = ["Not Helpful", "Helpful"]
            ax.bar(labels, label_counts.values)
            ax.set_title(f"{split_name.capitalize()} Split")
            ax.set_ylabel("Count")
            for i, v in enumerate(label_counts.values):
                ax.text(i, v + max(label_counts.values) * 0.01, str(v), ha="center")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "label_distribution.png", dpi=150)
    logger.info("Saved: label_distribution.png")


def plot_note_length_distribution(splits: dict, output_dir: Path) -> None:
    """Plot distribution of note lengths."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, split_name in zip(axes, ["train", "test"]):
        if split_name in splits:
            df = splits[split_name]
            if "summary" in df.columns:
                df["note_length"] = df["summary"].str.len()
                ax.hist(df["note_length"], bins=50, alpha=0.7, edgecolor="black")
                ax.set_title(f"Note Length Distribution ({split_name.capitalize()})")
                ax.set_xlabel("Character Count")
                ax.set_ylabel("Frequency")
                ax.axvline(
                    df["note_length"].mean(),
                    color="red",
                    linestyle="--",
                    label=f"Mean: {df['note_length'].mean():.0f}",
                )
                ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "note_length_distribution.png", dpi=150)
    logger.info("Saved: note_length_distribution.png")


def plot_class_balance_by_length(splits: dict, output_dir: Path) -> None:
    """Plot class balance across different note lengths."""
    df = splits.get("train")
    if df is None or "summary" not in df.columns or "label" not in df.columns:
        return

    df["note_length"] = df["summary"].str.len()

    # Create bins
    df["length_bin"] = pd.cut(
        df["note_length"], bins=[0, 100, 200, 400, 800, float("inf")]
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    pivot = df.groupby(["length_bin", "label"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", stacked=True, ax=ax)

    ax.set_title("Class Balance by Note Length")
    ax.set_xlabel("Note Length (characters)")
    ax.set_ylabel("Count")
    ax.legend(["Not Helpful", "Helpful"])
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_dir / "class_balance_by_length.png", dpi=150)
    logger.info("Saved: class_balance_by_length.png")


def generate_statistics(splits: dict, output_dir: Path) -> None:
    """Generate and save descriptive statistics."""
    stats = {}

    for split_name, df in splits.items():
        stats[split_name] = {
            "total_notes": len(df),
            "helpful_count": int((df["label"] == 1).sum())
            if "label" in df.columns
            else 0,
            "not_helpful_count": int((df["label"] == 0).sum())
            if "label" in df.columns
            else 0,
            "helpful_ratio": float((df["label"] == 1).mean())
            if "label" in df.columns
            else 0,
        }

        if "summary" in df.columns:
            lengths = df["summary"].str.len()
            stats[split_name]["avg_note_length"] = float(lengths.mean())
            stats[split_name]["min_note_length"] = int(lengths.min())
            stats[split_name]["max_note_length"] = int(lengths.max())

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "statistics.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Saved: statistics.json")
    print("\n=== Dataset Statistics ===")
    for split_name, s in stats.items():
        print(f"\n{split_name.capitalize()}:")
        for k, v in s.items():
            print(f"  {k}: {v}")


def sample_notes(splits: dict, output_dir: Path, n_samples: int = 5) -> None:
    """Save sample notes for each class."""
    df = splits.get("train")
    if df is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    samples = {"helpful": [], "not_helpful": []}

    for label in [0, 1]:
        label_df = df[df["label"] == label] if "label" in df.columns else df
        sample = label_df.head(n_samples)

        for _, row in sample.iterrows():
            note_data = {
                "note_id": row.get("noteId", "N/A"),
                "summary": row.get("summary", "N/A")[:500],
                "classification": row.get("classification", "N/A"),
            }
            if label == 1:
                samples["helpful"].append(note_data)
            else:
                samples["not_helpful"].append(note_data)

    with open(output_dir / "sample_notes.json", "w") as f:
        json.dump(samples, f, indent=2)

    logger.info("Saved: sample_notes.json")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    logger.info("Starting Exploratory Data Analysis...")

    # Load data
    splits = load_data(input_dir)

    if not splits:
        logger.error("No data found. Please run preprocess_data.py first.")
        return

    # Generate visualizations
    plot_label_distribution(splits, output_dir)
    plot_note_length_distribution(splits, output_dir)
    plot_class_balance_by_length(splits, output_dir)

    # Generate statistics
    generate_statistics(splits, output_dir)

    # Save sample notes
    sample_notes(splits, output_dir)

    logger.info("EDA complete!")


if __name__ == "__main__":
    main()
