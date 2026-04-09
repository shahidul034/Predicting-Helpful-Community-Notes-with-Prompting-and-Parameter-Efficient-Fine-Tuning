"""
TF-IDF + Logistic Regression Baseline.

Traditional ML baseline for Community Notes effectiveness prediction.
"""

import os
import sys
import argparse
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging
from src.utils.metrics import compute_all_metrics, print_classification_report

logger = setup_logging()
config = Config()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train TF-IDF + Logistic Regression baseline"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed parquet files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/models",
        help="Directory to save model",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10000,
        help="Maximum number of TF-IDF features",
    )
    parser.add_argument(
        "--ngram-range", type=int, nargs=2, default=[1, 2], help="N-gram range"
    )
    return parser.parse_args()


def load_data(input_dir: Path) -> tuple:
    """Load train and test data."""
    train_df = pd.read_parquet(input_dir / "community_notes_cleaned_train.parquet")
    test_df = pd.read_parquet(input_dir / "community_notes_cleaned_test.parquet")

    X_train = train_df["summary"].values
    y_train = train_df["label"].values
    X_test = test_df["summary"].values
    y_test = test_df["label"].values

    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    return X_train, y_train, X_test, y_test


def train_pipeline(X_train, y_train, max_features: int, ngram_range: tuple) -> Pipeline:
    """Train TF-IDF + Logistic Regression pipeline."""
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    stop_words="english",
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", solver="lbfgs", n_jobs=-1
                ),
            ),
        ]
    )

    logger.info("Training TF-IDF + Logistic Regression...")
    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_pipeline(pipeline: Pipeline, X_test, y_test) -> dict:
    """Evaluate pipeline on test set."""
    logger.info("Evaluating on test set...")

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    metrics = compute_all_metrics(y_test.tolist(), y_pred.tolist(), y_prob.tolist())

    print("\n=== Classification Report ===")
    print(print_classification_report(y_test.tolist(), y_pred.tolist()))

    print("\n=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    return metrics


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting TF-IDF + Logistic Regression baseline...")

    # Load data
    X_train, y_train, X_test, y_test = load_data(input_dir)

    # Train pipeline
    pipeline = train_pipeline(
        X_train, y_train, args.max_features, tuple(args.ngram_range)
    )

    # Evaluate
    metrics = evaluate_pipeline(pipeline, X_test, y_test)

    # Save model
    model_path = output_dir / "tfidf_lr_baseline.pkl"
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved model to {model_path}")

    # Save metrics
    import json

    with open(output_dir / "tfidf_lr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("TF-IDF + Logistic Regression baseline complete!")


if __name__ == "__main__":
    main()
