"""
BERT Fine-Tuning Baseline.

Standard transformer baseline for Community Notes effectiveness prediction.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.config import Config
from src.utils.logging_utils import setup_logging

logger = setup_logging()
config = Config()


def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT baseline")
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
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--max-length", type=int, default=512, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-5, help="Learning rate"
    )
    return parser.parse_args()


def load_data(input_dir: Path) -> tuple:
    """Load train and test data as HuggingFace datasets."""
    train_df = pd.read_parquet(input_dir / "community_notes_cleaned_train.parquet")
    test_df = pd.read_parquet(input_dir / "community_notes_cleaned_test.parquet")

    train_dataset = Dataset.from_pandas(train_df[["summary", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["summary", "label"]])

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")

    return train_dataset, test_dataset


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting BERT baseline training...")

    # Load data
    train_dataset, test_dataset = load_data(input_dir)

    # Load tokenizer and model
    logger.info(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )

    # Tokenize datasets
    def tokenize_function(example):
        return tokenizer(
            example["summary"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir / "bert_baseline"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=str(output_dir / "bert_baseline" / "logs"),
        logging_steps=50,
        report_to="none",
        torch_compile=False,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Evaluate
    logger.info("Evaluating on test set...")
    metrics = trainer.evaluate(tokenized_test)

    print("\n=== BERT Baseline Results ===")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Save model
    model_save_path = output_dir / "bert_baseline"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    logger.info(f"Saved model to {model_save_path}")

    logger.info("BERT baseline complete!")


if __name__ == "__main__":
    main()
