"""
Ablation Study.

Analyze the impact of different features on model performance.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logging

logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluations",
        help="Directory to save ablation results",
    )
    return parser.parse_args()


def load_data(input_dir: Path) -> pd.DataFrame:
    """Load training data for analysis."""
    train_df = pd.read_parquet(input_dir / "community_notes_cleaned_train.parquet")
    test_df = pd.read_parquet(input_dir / "community_notes_cleaned_test.parquet")
    return pd.concat([train_df, test_df])


def analyze_note_length(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze impact of note length on label distribution."""
    df = df.copy()
    df["note_length"] = df["summary"].str.len()

    # Create length bins
    bins = [0, 100, 200, 400, 800, 1600, float("inf")]
    labels = ["0-100", "100-200", "200-400", "400-800", "800-1600", "1600+"]
    df["length_bin"] = pd.cut(df["note_length"], bins=bins, labels=labels)

    # Calculate helpfulness rate by length
    analysis = df.groupby("length_bin").agg({"label": ["count", "mean"]}).reset_index()
    analysis.columns = ["length_bin", "count", "helpfulness_rate"]

    return analysis


def analyze_url_presence(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze impact of URL presence on label distribution."""
    df = df.copy()

    # Detect URLs in notes
    import re

    df["has_url"] = df["summary"].str.contains(
        r"https?://|www\.", regex=True, case=False
    )

    analysis = df.groupby("has_url").agg({"label": ["count", "mean"]}).reset_index()
    analysis.columns = ["has_url", "count", "helpfulness_rate"]
    analysis["has_url"] = analysis["has_url"].map(
        {True: "With URL", False: "Without URL"}
    )

    return analysis


def analyze_classification_type(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze impact of classification type on label distribution."""
    if "classification" not in df.columns:
        return pd.DataFrame()

    analysis = (
        df.groupby("classification").agg({"label": ["count", "mean"]}).reset_index()
    )
    analysis.columns = ["classification", "count", "helpfulness_rate"]

    return analysis


def analyze_source_citation(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze impact of source citations on label distribution."""
    df = df.copy()

    # Common source citation patterns
    source_patterns = [
        r"according to",
        r"according\sto",
        r"study",
        r"research",
        r"report",
        r"source",
        r"evidence",
        r"data",
        r"nasa",
        r"ipcc",
        r"cdc",
        r"who",
        r"\[.*?\]",  # Bracketed citations
    ]

    df["has_citation"] = df["summary"].str.contains(
        "|".join(source_patterns), regex=True, case=False
    )

    analysis = (
        df.groupby("has_citation").agg({"label": ["count", "mean"]}).reset_index()
    )
    analysis.columns = ["has_citation", "count", "helpfulness_rate"]
    analysis["has_citation"] = analysis["has_citation"].map(
        {True: "With Citation", False: "Without Citation"}
    )

    return analysis


def analyze_tone(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze tone characteristics."""
    df = df.copy()

    # Detect emotional language
    emotional_words = [
        "obviously",
        "clearly",
        "stupid",
        "idiot",
        "garbage",
        "fake",
        "hate",
        "love",
        "amazing",
        "terrible",
        "worst",
        "best",
    ]

    df["has_emotional"] = df["summary"].str.contains(
        r"\b(" + "|".join(emotional_words) + r")\b", regex=True, case=False
    )

    # Detect neutral language indicators
    neutral_indicators = [
        "research shows",
        "studies indicate",
        "data suggests",
        "according to",
        "report states",
        "evidence shows",
    ]

    df["has_neutral"] = df["summary"].str.contains(
        r"\b(" + "|".join(neutral_indicators) + r")\b", regex=True, case=False
    )

    analysis = (
        df.groupby(["has_emotional", "has_neutral"])
        .agg({"label": ["count", "mean"]})
        .reset_index()
    )
    analysis.columns = ["has_emotional", "has_neutral", "count", "helpfulness_rate"]

    return analysis


def plot_ablation_results(results: dict, output_dir: Path) -> None:
    """Plot ablation study results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Note length analysis
    if "note_length" in results:
        ax = axes[0, 0]
        length_df = results["note_length"]
        ax.bar(length_df["length_bin"].astype(str), length_df["helpfulness_rate"])
        ax.set_xlabel("Note Length (characters)")
        ax.set_ylabel("Helpfulness Rate")
        ax.set_title("Helpfulness by Note Length")
        ax.set_xticklabels(length_df["length_bin"].astype(str), rotation=45, ha="right")
        ax.set_ylim(0, 1)

    # URL presence
    if "url_presence" in results:
        ax = axes[0, 1]
        url_df = results["url_presence"]
        ax.bar(url_df["has_url"], url_df["helpfulness_rate"])
        ax.set_xlabel("URL Presence")
        ax.set_ylabel("Helpfulness Rate")
        ax.set_title("Helpfulness by URL Presence")
        ax.set_ylim(0, 1)

    # Source citation
    if "source_citation" in results:
        ax = axes[1, 0]
        citation_df = results["source_citation"]
        ax.bar(citation_df["has_citation"], citation_df["helpfulness_rate"])
        ax.set_xlabel("Source Citation")
        ax.set_ylabel("Helpfulness Rate")
        ax.set_title("Helpfulness by Source Citation")
        ax.set_ylim(0, 1)

    # Classification type
    if "classification_type" in results:
        ax = axes[1, 1]
        class_df = results["classification_type"]
        ax.bar(class_df["classification"].astype(str), class_df["helpfulness_rate"])
        ax.set_xlabel("Classification Type")
        ax.set_ylabel("Helpfulness Rate")
        ax.set_title("Helpfulness by Classification Type")
        ax.set_xticklabels(
            class_df["classification"].astype(str), rotation=45, ha="right"
        )
        ax.set_ylim(0, 1)

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "ablation_study.png", dpi=150)
    logger.info("Saved: ablation_study.png")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting ablation study...")

    # Load data
    df = load_data(input_dir)
    logger.info(f"Loaded {len(df)} notes for analysis")

    # Run analyses
    results = {}

    logger.info("Analyzing note length...")
    results["note_length"] = analyze_note_length(df)

    logger.info("Analyzing URL presence...")
    results["url_presence"] = analyze_url_presence(df)

    logger.info("Analyzing classification type...")
    results["classification_type"] = analyze_classification_type(df)

    logger.info("Analyzing source citations...")
    results["source_citation"] = analyze_source_citation(df)

    logger.info("Analyzing tone...")
    results["tone"] = analyze_tone(df)

    # Print results
    print("\n=== Ablation Study Results ===\n")

    print("Note Length Analysis:")
    print(results["note_length"])

    print("\nURL Presence Analysis:")
    print(results["url_presence"])

    print("\nSource Citation Analysis:")
    print(results["source_citation"])

    # Save results
    results_dict = {k: v.to_dict("records") for k, v in results.items()}
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    # Plot results
    plot_ablation_results(results, output_dir)

    logger.info("Ablation study complete!")


if __name__ == "__main__":
    main()
