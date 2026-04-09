"""
Model Evaluation.

Compare all models and generate comprehensive evaluation reports.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logging_utils import setup_logging

logger = setup_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and compare all models")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="outputs/evaluations",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluations",
        help="Directory to save comparison results",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> dict:
    """Load all model evaluation results."""
    results = {}

    # Load TF-IDF results
    tfidf_file = results_dir / "tfidf_lr_metrics.json"
    if tfidf_file.exists():
        with open(tfidf_file) as f:
            results["tfidf_lr"] = json.load(f)

    # Load BERT results
    bert_dir = results_dir.parent / "models" / "bert_baseline"
    if bert_dir.exists():
        metrics_file = bert_dir / "metrics.json"
        # Check for evaluation results in trainer state
        state_file = bert_dir / "trainer_state.json"
        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
                results["bert"] = {
                    "eval_loss": state.get("eval_loss"),
                    "eval_accuracy": state.get("eval_accuracy"),
                    "eval_f1": state.get("eval_f1"),
                }

    # Load prompting results
    for method in ["zero-shot", "few-shot", "cot"]:
        results_file = results_dir / f"{method}_results.json"
        if results_file.exists():
            with open(results_file) as f:
                data = json.load(f)
                results[method] = data.get("metrics", {})

    # Load fine-tuning results
    for model in ["qwen3-4b", "gemma-3-4b"]:
        inference_file = results_dir.glob(f"*{model}*inference*.json")
        for f in inference_file:
            with open(f) as file:
                data = json.load(file)
                results[f"{model}_lora"] = data.get("metrics", {})

    return results


def create_comparison_table(results: dict) -> pd.DataFrame:
    """Create comparison table of all models."""
    metrics_to_compare = ["accuracy", "f1", "precision", "recall", "auc_roc"]

    rows = []
    for model_name, metrics in results.items():
        row = {"Model": model_name}
        for metric in metrics_to_compare:
            row[metric] = metrics.get(metric, "N/A")
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def plot_comparison(results: dict, output_dir: Path) -> None:
    """Create comparison visualizations."""
    metrics_to_plot = ["accuracy", "f1", "precision", "recall"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.Set3(range(len(results)))

    for ax, metric in zip(axes, metrics_to_plot):
        models = list(results.keys())
        values = [results[m].get(metric, 0) for m in models]

        bars = ax.bar(models, values, color=colors)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.capitalize()} Score")
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, values):
            if isinstance(val, (int, float)):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "model_comparison.png", dpi=150)
    logger.info("Saved: model_comparison.png")


def generate_report(results: dict, output_dir: Path) -> None:
    """Generate comprehensive evaluation report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report = f"""# Community Notes Effectiveness Prediction - Evaluation Report

Generated: {timestamp}

## Model Comparison

| Model | Accuracy | F1-Score | Precision | Recall | AUC-ROC |
|-------|----------|----------|-----------|--------|---------|
"""

    for model_name, metrics in results.items():
        acc = metrics.get("accuracy", "N/A")
        f1 = metrics.get("f1", "N/A")
        prec = metrics.get("precision", "N/A")
        rec = metrics.get("recall", "N/A")
        auc = metrics.get("auc_roc", "N/A")

        if isinstance(acc, (int, float)):
            acc = f"{acc:.4f}"
        if isinstance(f1, (int, float)):
            f1 = f"{f1:.4f}"
        if isinstance(prec, (int, float)):
            prec = f"{prec:.4f}"
        if isinstance(rec, (int, float)):
            rec = f"{rec:.4f}"
        if isinstance(auc, (int, float)):
            auc = f"{auc:.4f}"

        report += f"| {model_name} | {acc} | {f1} | {prec} | {rec} | {auc} |\n"

    report += """
## Baseline Models

### TF-IDF + Logistic Regression
- Traditional ML approach using bag-of-words features
- Fast training and inference
- Good baseline for comparison

### BERT Fine-Tuning
- Standard transformer baseline
- Captures contextual information
- Requires more resources than TF-IDF

## Prompting Approaches

### Zero-Shot Prompting
- No fine-tuning required
- Direct instruction following
- Quick to deploy

### Few-Shot Prompting
- In-context learning with examples
- Better than zero-shot typically
- More tokens required

### Chain-of-Thought (CoT)
- Includes reasoning in output
- More interpretable predictions
- Longer generation time

## Fine-Tuning Approaches

### Qwen3-4B-Instruct + LoRA
- Primary method for this project
- Efficient parameter updates
- Good balance of performance and resources

### Gemma-3-4B-IT + LoRA
- Architecture comparison
- Alternative model family
- Similar performance expected

## Recommendations

Based on the evaluation results:
1. Compare fine-tuned models against baselines
2. Consider trade-offs between accuracy and inference time
3. Analyze error patterns for improvement opportunities
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "evaluation_report.md", "w") as f:
        f.write(report)

    logger.info("Saved: evaluation_report.md")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    logger.info("Starting model evaluation...")

    # Load all results
    results = load_results(results_dir)

    if not results:
        logger.warning("No evaluation results found. Run models first.")
        return

    logger.info(f"Found {len(results)} model results")

    # Create comparison table
    df = create_comparison_table(results)
    print("\n=== Model Comparison ===")
    print(df.to_string(index=False))

    # Save comparison table
    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # Plot comparison
    plot_comparison(results, output_dir)

    # Generate report
    generate_report(results, output_dir)

    # Save full results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
