"""
Error Analysis.

Analyze model prediction errors to understand failure modes.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logging

logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Perform error analysis")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed parquet files",
    )
    parser.add_argument(
        "--predictions-file", type=str, help="Path to predictions JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluations",
        help="Directory to save error analysis",
    )
    return parser.parse_args()


def load_predictions(predictions_file: Path) -> tuple:
    """Load predictions from JSON file."""
    with open(predictions_file) as f:
        data = json.load(f)

    predictions = data.get("predictions", [])
    true_labels = data.get("true_labels", [])

    return predictions, true_labels


def load_test_data(input_dir: Path) -> pd.DataFrame:
    """Load test data."""
    test_df = pd.read_parquet(input_dir / "community_notes_cleaned_test.parquet")
    return test_df


def categorize_errors(
    predictions: list, true_labels: list, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Categorize prediction errors."""
    errors = []

    for i, (pred, true) in enumerate(zip(predictions, true_labels)):
        if pred != ("HELPFUL" if true == 1 else "NOT_HELPFUL"):
            row = test_df.iloc[i].to_dict()
            row["true_label"] = true
            row["predicted_label"] = pred
            row["error"] = True

            # Categorize error type
            if pred == "UNKNOWN":
                row["error_type"] = "model_uncertain"
            elif true == 1 and pred == "NOT_HELPFUL":
                row["error_type"] = "false_negative"
            else:
                row["error_type"] = "false_positive"

            errors.append(row)

    return pd.DataFrame(errors)


def analyze_error_patterns(errors_df: pd.DataFrame) -> dict:
    """Analyze patterns in errors."""
    if len(errors_df) == 0:
        return {}

    analysis = {}

    # Error type distribution
    analysis["error_types"] = errors_df["error_type"].value_counts().to_dict()

    # Note length analysis
    if "summary" in errors_df.columns:
        errors_df["note_length"] = errors_df["summary"].str.len()
        analysis["avg_error_length"] = errors_df["note_length"].mean()
        analysis["length_by_error_type"] = (
            errors_df.groupby("error_type")["note_length"].mean().to_dict()
        )

    # URL presence in errors
    if "summary" in errors_df.columns:
        import re

        errors_df["has_url"] = errors_df["summary"].str.contains(
            r"https?://|www\.", regex=True, case=False
        )
        analysis["url_in_errors"] = errors_df["has_url"].mean()

    return analysis


def get_sample_errors(errors_df: pd.DataFrame, n_samples: int = 10) -> list:
    """Get sample errors for manual review."""
    samples = []

    for error_type in errors_df["error_type"].unique():
        type_df = errors_df[errors_df["error_type"] == error_type]
        type_samples = type_df.head(
            n_samples // len(errors_df["error_type"].unique())
        ).to_dict("records")

        for sample in type_samples:
            sample["error_type"] = error_type
            samples.append(sample)

    return samples[:n_samples]


def plot_error_analysis(
    errors_df: pd.DataFrame, analysis: dict, output_dir: Path
) -> None:
    """Create error analysis visualizations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error type distribution
    if len(errors_df) > 0:
        ax = axes[0]
        error_counts = errors_df["error_type"].value_counts()
        ax.pie(error_counts.values, labels=error_counts.index, autopct="%1.1f%%")
        ax.set_title("Error Type Distribution")

        # Error by note length
        ax = axes[1]
        if "note_length" in errors_df.columns:
            errors_df.boxplot(column="note_length", by="error_type", ax=ax)
            ax.set_xlabel("Error Type")
            ax.set_ylabel("Note Length (characters)")
            ax.set_title("Note Length by Error Type")
            plt.suptitle("")

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "error_analysis.png", dpi=150)
    logger.info("Saved: error_analysis.png")


def generate_error_report(
    errors_df: pd.DataFrame, analysis: dict, samples: list, output_dir: Path
) -> None:
    """Generate error analysis report."""
    report = f"""# Error Analysis Report

## Summary

- Total Errors: {len(errors_df)}
- Error Rate: {len(errors_df) / (len(errors_df) + 100) * 100:.2f}% (based on first 100 predictions)

## Error Type Distribution

| Error Type | Count | Percentage |
|------------|-------|------------|
"""

    if len(errors_df) > 0:
        error_counts = errors_df["error_type"].value_counts()
        total = len(errors_df)
        for error_type, count in error_counts.items():
            pct = count / total * 100
            report += f"| {error_type} | {count} | {pct:.1f}% |\n"

    report += """
## Error Patterns

### Note Length
"""

    if "avg_error_length" in analysis:
        report += f"- Average note length in errors: {analysis['avg_error_length']:.0f} characters\n"

    report += """
### URL Presence
"""

    if "url_in_errors" in analysis:
        report += (
            f"- Notes with URLs in errors: {analysis['url_in_errors'] * 100:.1f}%\n"
        )

    report += """
## Sample Errors for Manual Review

### False Negatives (Helpful notes predicted as Not Helpful)
"""

    for i, sample in enumerate(samples[:5], 1):
        if sample.get("error_type") == "false_negative":
            report += f"""
**Sample {i}:**
- Note: "{sample.get("summary", "N/A")[:200]}..."
- True Label: {"Helpful" if sample.get("true_label") == 1 else "Not Helpful"}
- Predicted: {sample.get("predicted_label")}
"""

    report += """
### False Positives (Not Helpful notes predicted as Helpful)
"""

    for i, sample in enumerate(samples[5:10], 1):
        if sample.get("error_type") == "false_positive":
            report += f"""
**Sample {i}:**
- Note: "{sample.get("summary", "N/A")[:200]}..."
- True Label: {"Helpful" if sample.get("true_label") == 1 else "Not Helpful"}
- Predicted: {sample.get("predicted_label")}
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "error_analysis_report.md", "w") as f:
        f.write(report)

    logger.info("Saved: error_analysis_report.md")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting error analysis...")

    # Load predictions
    if not args.predictions_file:
        # Find latest inference results
        inference_files = list(output_dir.glob("inference_results_*.json"))
        if inference_files:
            args.predictions_file = max(
                inference_files, key=lambda x: x.stat().st_mtime
            )
        else:
            logger.error("No predictions file found. Run inference first.")
            return

    predictions, true_labels = load_predictions(Path(args.predictions_file))
    logger.info(f"Loaded {len(predictions)} predictions")

    # Load test data
    test_df = load_data(input_dir)

    # Categorize errors
    errors_df = categorize_errors(predictions, true_labels, test_df)
    logger.info(f"Found {len(errors_df)} errors")

    # Analyze error patterns
    analysis = analyze_error_patterns(errors_df)

    # Get sample errors
    samples = get_sample_errors(errors_df)

    # Print summary
    print("\n=== Error Analysis Summary ===")
    print(f"Total errors: {len(errors_df)}")
    if analysis:
        print(f"\nError types: {analysis.get('error_types', {})}")

    # Plot analysis
    plot_error_analysis(errors_df, analysis, output_dir)

    # Generate report
    generate_error_report(errors_df, analysis, samples, output_dir)

    # Save detailed errors
    errors_df.to_csv(output_dir / "error_samples.csv", index=False)

    logger.info("Error analysis complete!")


if __name__ == "__main__":
    main()
